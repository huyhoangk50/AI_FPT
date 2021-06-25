import cv2
import numpy as np
import torch 
import os
from PIL import Image

from shapely.geometry import Polygon
import libs.yolo_darknet.darknet as darknet
from libs.yolo_darknet.yolov4 import predict_yolov4 # format max-min

from get_mask import predict_img, mask_to_image
from libs.unet.unet import UNet
from configparser import ConfigParser
from sort import *
import os

def make_polygon_segment_zone(output_image):
    image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image_gray, 30, 200)
    ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.array([])

    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull = np.append(hull, np.array(cv2.convexHull(contours[i], False)))
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
    mask_polygon = Polygon(mask_polygon)

    new_list_bboxes = []
    new_list_scores = []
    new_list_labels = []

    for bbox, score, label in zip(list_bboxes, list_scores, list_labels):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        p1 = (x_min, y_min)
        p2 = (x_max, y_min)
        p3 = (x_max, y_max)
        p4 = (x_min, y_max)
        bb = Polygon([p1, p2, p3, p4])
        if mask_polygon.intersects(bb):
            new_list_bboxes.append(bbox)
            new_list_scores.append(score)
            new_list_labels.append(label)

    return new_list_bboxes, new_list_scores, new_list_labels

def draw_bbox_maxmin(image, bbox, view_id=False, track_id=None):
    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    if view_id:
        track_id = str(track_id)
        cv2.putText(image, "box_id: " + track_id, (int(bbox[0])+2, int(bbox[1])-1), 0, 1, (255, 0, 0), 1)

    return image 

if __name__ == '__main__':
    video_path = os.getenv("VIDEO_PATH")
    if not video_path:
        raise 'must set VIDEO_PATH'
    listVideos= []
    for file in os.listdir(video_path):
        if file.endswith(".MP4") or file.endswith(".mp4"):
            listVideos.append([file, os.path.join(video_path, file)])
    if len(listVideos) == 0:
        raise 'Empty MP4 file in VIDEO_PATH'
    print ("List of all files: ", listVideos)
    output_video = os.getenv("OUTPUT_VIDEO")
    if not output_video:
        raise 'must set OUTPUT_VIDEO'

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


    for videoCount in range(len(listVideos)):
        print ("Video: " + str(videoCount) + "th in total: " + str(len(listVideos)))
        record = listVideos[videoCount]
        videoName = record[0]
        videoPath = record[1]
        outVideoPath = output_video + videoName[0: -3] + "avi"
        print("outVideoPath: ", outVideoPath)
        frameCount = 0
        # exit()
        cap = cv2.VideoCapture(videoPath)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(5)
        frameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # outputVideo = 
        size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
        
        while True :
            _, frame = cap.read()
            if frame is None:
                print ("End of video")
                break
            print("Reading frame: " + str(frameCount) + " in total: " + str(frameNum))
            frameCount += 1
            # try:
            # convert pil image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            # main process
            mask_polygon = get_mask_polygon(im_pil, NET, DEVICE)
            list_bboxes, list_scores, list_labels = detect_boxes(frame, NETWORK, CLASS_NAMES, CLASS_COLORS, mask_polygon)

            if len(list_bboxes) > 0:
                list_bboxes = np.array(list_bboxes)
                # update SORT
                track_bbs_ids = SORT_TRACKER.update(list_bboxes)

                for track in track_bbs_ids:
                    frame = draw_bbox_maxmin(frame, track[:4], True, int(track[4]))
            
            # write the flipped frame
            out.write(frame)
            # except:
            #     print("error occurs")
            #     pass

            # draw
            # for bbox in list_bboxes:
            #     frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
            
            # frame = draw_polygon(frame, mask_polygon)
            # cv2.imwrite("output.jpg", frame)
            
        cap.stop()
        cv2.destroyAllWindows()  

    
