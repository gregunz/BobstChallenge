import cv2
import numpy as np


def crop_contour(img):
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(img_gray,127,255,0)
    
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)
    
    x, y, width, height = cv2.boundingRect(cnt)
    roi = img[y:y+height, x:x+width]
    
    return roi


def get_min_width_height(images):
    return tuple([np.min([img.shape[i] for img in images]) for i in range(2)])