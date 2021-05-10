import cv2
import numpy as np 
import argparse
import os
import glob
import random
import time
import numpy as np
import darknet_fastapi.darknet as darknet
from darknet_fastapi.yolov4 import predict_yolov4 # format tlwh

NETWORK, CLASS_NAMES, CLASS_COLORS = darknet.load_network(
    "yolo_model/yolov4-custom.cfg",
    "yolo_model/obj.data",
    "yolo_model/yolov4-custom.weights",
    batch_size=8
)

def detect_boxes(image):
    list_bboxes, list_scores, list_labels = predict_yolov4(image, NETWORK, CLASS_NAMES, CLASS_COLORS)

    return list_bboxes, list_scores, list_labels

if __name__ == "__main__":
    image = cv2.imread("images/demo.jpg")
    list_bboxes, list_scores, list_labels = detect_boxes(image)

    for bbox in list_bboxes:
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 1)
    
    cv2.imwrite("results.jpg", image)
