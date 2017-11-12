import cv2
import os
import numpy as np
from split import splits
from impl import *

images_dir = "/mnt/bobstchallenge/"
images_paths = sorted(["{d}/{f}".format(d=images_dir, f=f) for f in os.listdir(images_dir)])

for cat in range(1, 26):
    from_, to = splits[cat]
    images = [cv2.imread(p, 0) for p in images_paths[from_:to+1]]
    aligned_images = [old_align(img, images[0]) for img in images]
    
    mean = np.mean(aligned_images, axis=0)
    std = np.std(aligned_images, axis=0)
    
    mask = mask_contour(aligned_images[0])
    kernel = np.ones((2,2), np.uint8)
    
    for idx, img in enumerate(aligned_images):
        img_dif = (np.abs(img - mean) - 2*std).clip(min=0)
        img_dif = contrast(img_dif, 15)
        img_dif = img_dif * mask
        img_dif = cv2.morphologyEx(img_dif.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        cnts = contours(img_dif)
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 8]
        if len(cnts) > 0:
            or_img = cv2.imread(images_paths[from_ + idx])
            or_img = old_align(or_img, images[0])
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                r = 8
                cv2.rectangle(or_img, (x-r,y-r), (x+w+r,y+h+r), (0,0,255), 2)
            print(cv2.imwrite('output/ID{}.jpg'.format(from_ + idx + 1), or_img))