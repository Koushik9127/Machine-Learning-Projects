import os
import numpy as np
import cv2

def mask_to_polygons(mask_path):
    mask = cv2.imread(mask_path, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.squeeze().tolist() for cnt in contours if len(cnt) >= 3]
