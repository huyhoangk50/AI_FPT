import cv2
import numpy as np
import torch 
import os
from PIL import Image

from shapely.geometry import Polygon
import libs.yolo_darknet.darknet as darknet
from libs.yolo_darknet.yolov4 import predict_yolov4 # format tlwh

from get_mask import predict_img, mask_to_image
from libs.unet.unet import UNet
from configparser import ConfigParser
from sort import *

def make_polygon_segment_zone(output_image):
    image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image_gray, 30, 200)
    ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []

    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    hull = np.array(hull, np.int32)
    hull = hull.reshape((-1, 1, 2))

    M = cv2.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])



    leftmost = tuple(hull[hull[:,:,0].argmin()][0])
    rightmost = tuple(hull[hull[:,:,0].argmax()][0])
    topmost = tuple(hull[hull[:,:,1].argmin()][0])
    bottommost = tuple(hull[hull[:,:,1].argmax()][0])
    
    x_topleft = int((topmost[0] - leftmost[0]) / 2)
    y_topleft = topmost[1]

    x_topright = int((rightmost[0] - topmost[0]) / 2)
    y_topright = topmost[1]

    x_bottomleft = leftmost[0]
    y_bottomleft = leftmost[1]

    x_bottomright = rightmost[0]
    y_bottomright = rightmost[1]

    # polygon = np.array([[leftmost[0], leftmost[1]], [rightmost[0], rightmost[1]], [955, topmost[1]], [855, topmost[1]]], np.int32)
    # polygon = polygon.reshape((-1,1,2))

    polygon = [[leftmost[0], leftmost[1]], [rightmost[0], rightmost[1]], [955, topmost[1]], [855, topmost[1]]]

    return polygon #topleft, topright, bottomright, bottomleft

def draw_polygon(image, polygon):
    image = cv2.drawContours(image, [np.array(polygon)], -1, (0, 255, 0), thickness=2)
    return image

def get_mask_polygon(image, net, device):
    mask = predict_img(image, net, device)
    mask_cv2 = cv2.cvtColor(np.asarray(mask_to_image(mask)), cv2.COLOR_RGB2BGR)
    mask_polygon = make_polygon_segment_zone(mask_cv2)

    return mask_polygon

# find orientation 
def orientation(p, q, r):        
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        return 1    # counter-clockwise

    elif (val < 0): 
        return 2    # clockwise 

    else: 
        return 0    # alignment

def detect_boxes(image, NETWORK, CLASS_NAMES, CLASS_COLORS, mask_polygon):
    list_bboxes, list_scores, list_labels = predict_yolov4(image, NETWORK, CLASS_NAMES, CLASS_COLORS)
    # mask_polygon = Polygon(mask_polygon)
    topleft_mask, topright_mask, bottomright_mask, bottomleft_mask = mask_polygon[:]

    new_list_bboxes = []
    new_list_scores = []
    new_list_labels = []

    for bbox, score, label in zip(list_bboxes, list_scores, list_labels):
        # tl = [bbox[0], bbox[1]]
        # tr = [bbox[0]+bbox[2], bbox[1]]
        br = [bbox[0]+bbox[2], bbox[1]+bbox[3]]
        bl = [bbox[0], bbox[1]+bbox[3]]
        bot_center = [(br[0]+bl[0])//2, (br[1]+bl[1])//2]

        # bbox_polygon = Polygon([tl, tr, br, bl])
        if orientation(topleft_mask, bottomleft_mask, bot_center) == 2 and \
            orientation(topright_mask, bottomright_mask, bot_center) == 1:

            new_list_bboxes.append(bbox)
            new_list_scores.append(score)
            new_list_labels.append(label)

    return new_list_bboxes, new_list_scores, new_list_labels

if __name__ == '__main__':
    video_path = os.getenv("VIDEO_PATH")
    if not video_path:
        raise 'must set VIDEO_PATH'

    CFG = ConfigParser()
    CFG.read("configs/config.ini")

    model_unet_path = CFG["unet"]["unet_model"]

    NETWORK, CLASS_NAMES, CLASS_COLORS = darknet.load_network(
        CFG["yolo"]["yolo_cfg"],
        CFG["yolo"]["obj_data"],
        CFG["yolo"]["yolo_weight"],
        batch_size=8
    )

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NET = UNet(n_channels=3, n_classes=1)
    NET.to(device=DEVICE)
    NET.load_state_dict(torch.load(model_unet_path, map_location=DEVICE))

    # Sort 
    SORT_TRACKER = Sort()

    cap = cv2.VideoCapture(video_path)
    while True :
        _, frame = cap.read()

        # convert pil image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        # main process
        mask_polygon = get_mask_polygon(im_pil, NET, DEVICE)
        list_bboxes, list_scores, list_labels = detect_boxes(frame, NETWORK, CLASS_NAMES, CLASS_COLORS, mask_polygon)

        # if len(list_bboxes) > 0:
        #     list_bboxes = np.array(list_bboxes)
        #     # update SORT
        #     track_bbs_ids = SORT_TRACKER.update(list_bboxes)
        
        #     print("track: ", track_bbs_ids)

        # draw
        for bbox in list_bboxes:
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
        
        frame = draw_polygon(frame, mask_polygon)
        cv2.imwrite("output.jpg", frame)
        
    cap.stop()
    cv2.destroyAllWindows()  

    