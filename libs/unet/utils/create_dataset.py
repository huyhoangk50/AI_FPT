import os 
import cv2 
import glob2

import numpy as np
from tqdm import tqdm 
from PIL import Image

def create_mask_image(save_dir, images_dir, annotates_dir):
    list_of_annotates = os.listdir(annotates_dir)
    for annotate in list_of_annotates:
        name = annotate.split('.')[0]
        image_name = name +'.jpg'
        image_path = os.path.join(images_dir, image_name)
        annotate_path = os.path.join(annotates_dir, annotate)

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        mask_image = np.zeros((1080, 1920), np.int32)

        f = open(annotate_path, 'r')
        content = f.read()
        content = content.split('\n')[:4]
        arr = []
        for pt in content:
            pt = pt.split(' ')[:2]
            for i in range(0, len(pt)):
                pt[i] = int(pt[i])
            arr.append(pt)
        pts = np.array(arr, np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask_image = cv2.fillPoly(mask_image, pts = [pts], color = (255, 255, 255))

        annotate_image_name = name + '_mask' +'.jpg'
        new_image_dir = os.path.join(save_dir, annotate_image_name)

        cv2.imwrite(new_image_dir, mask_image)

if __name__ == "__main__":
    # create_mask()
    create_dataset()