import cv2
import os
import numpy as np
from split import splits
from impl import *

images_dir = "/path/to/images"
images_paths = sorted(["{d}/{f}".format(d=images_dir, f=f) for f in os.listdir(images_dir)])

for cat in range(1, 26):
    # inclusive - inclusive
    from_, to = splits[cat]
    
    color_images = [cv2.imread(p) for p in images_paths[from_:to+1]]
    # todo: should not load it again from disk just for grayscale 
    gray_images = [cv2.imread(p, 0) for p in images_paths[from_:to+1]]
    
    template = images[0]
    gray_aligned_images = []
    color_aligned_images = []
    for gray_img, color_img in zip(gray_images, color_images):
        warp_matrix = find_affine_transform_matrix(gray_img, template)
        
        gray_aligned_images.append(affine_transform(gray_img, warp_matrix))
        color_aligned_images.append(affine_transform(color_img, warp_matrix))
        
    mean = np.mean(gray_aligned_images, axis=0)
    std = np.std(gray_aligned_images, axis=0)
    
    mask = mask_contour(template)
    kernel = np.ones((2,2), np.uint8)
    
    for idx, img in enumerate(gray_aligned_images):
        img_dif = (np.abs(img - mean) - 2*std).clip(min=0)
        img_dif = contrast(img_dif, 15)
        img_dif = img_dif * mask
        img_dif = cv2.morphologyEx(img_dif.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        cnts = find_contours(img_dif)
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 8]
        if len(cnts) > 0:
            color_image = color_aligned_images[idx]
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                r = 8
                cv2.rectangle(color_image, (x-r,y-r), (x+w+r,y+h+r), (0,0,255), 2)
            print(cv2.imwrite('output/ID{}.jpg'.format(from_ + idx + 1), color_image))
            
