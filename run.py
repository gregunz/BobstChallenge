import cv2
import os
import numpy as np
from tqdm import tqdm

from split import splits
from impl import *

images_dir = "/mnt/bobst-lauzhack"
images_paths = sorted(["{d}/{f}".format(d=images_dir, f=f) for f in os.listdir(images_dir)])

for cat in range(1, 26):
    from_, to = splits[cat]
    images = [cv2.imread(p, 0) for p in tqdm(images_paths[from_:to])]
    aligned_images = [old_align(img, images[0]) for img in tqdm(images)]
    
    mean = np.mean(aligned_images, axis=0)
    std = np.std(aligned_images, axis=0)
    
    mask = impl.mask_contour(np_imgs[0])
    kernel = np.ones((2,2), np.uint8)
    
    for idx, img in tqdm(enumarate(aligned_images)):
        img = (np.abs(img - mean) - 2*std).clip(min=0)
        img = contrast(img, 15)
        img = img * mask
        img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cv2.imwrite('output/dif_{c}_{i}.jpg'.format(c=cat, i=idx), img)