
# coding: utf-8

# In[1]:

import cv2
import os
import numpy as np
import impl
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
#get_ipython().magic('matplotlib inline')


# In[2]:

images = os.listdir('Images')
images_path = [os.path.join("Images", i) for i in images]
IMG_NAME = 'Images/ID0001.tif'
images_path = sorted(images_path)


# In[3]:

def show(img):
    plt.imshow(img, cmap="gray")
    plt.show()


# In[4]:

template_img = cv2.imread(IMG_NAME, 0)
#show(template_img)
print(template_img.shape)
template_1 = template_img[1100:1225, 320:440]
template_3 = template_img[1100:1225, 1350:1650]
template_2 = template_img[500:600, 1300:1400]
#show(template_1)
#show(template_2)
#show(template_3)


# In[7]:

from split2 import *


# In[9]:

def_pos = np.array([320, 1098])
def_pos2 = np.array([1350, 1097])
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))
def proccess_split(s):
    imgs = []
    i=0
    for IMG_NAME in tqdm(images_path[split[s][0][0] : split[s][0][1] + 1]):
        i+=1
        img = cv2.imread(IMG_NAME, 0)

        rows,cols = img.shape
        res_1_ = cv2.matchTemplate(img,template_1,cv2.TM_CCOEFF)
        res_3_ = cv2.matchTemplate(img,template_3,cv2.TM_CCOEFF)

        _, _, _, max_loc_1_ = cv2.minMaxLoc(res_1_)
        _, _, _, max_loc_3_ = cv2.minMaxLoc(res_3_)
        del res_1_
        del res_3_
        pos_1 = np.array([max_loc_1_[0], max_loc_1_[1]])
        pos_3 = np.array([max_loc_3_[0], max_loc_3_[1]])

        #translate
        trans = def_pos - pos_1
        M = np.float32([[1,0,trans[0]],[0,1,trans[1]]])
        dst = cv2.warpAffine(img,M,(cols,rows))
        #rotation
        angle = angle_between(pos_3-pos_1, def_pos2-def_pos)
        M = cv2.getRotationMatrix2D((def_pos[0],def_pos[1]),angle,1)
        dst = cv2.warpAffine(dst,M,(cols,rows))

        imgs.append(dst)
    np_imgs = np.array(imgs)
    del imgs
    mean = np_imgs.mean(axis=0)
    std = np_imgs.std(axis=0)
    cv2.imwrite('output/centred_for_split_'+str(s)+'-mean.jpg',mean)
    cv2.imwrite('output/centred_for_split_'+str(s)+'-std.jpg',std)
    mask = impl.mask_contour(np_imgs[0])
    mask2 = np.zeros(np_imgs[0].shape)
    mask2[500:1200, 200:1700] = 1
    kernel = np.ones(split[s][1],np.uint8)

    j=0
    for img in tqdm(np_imgs):
        j+=1
        img_t = (np.abs(img-mean)-2.5*std).clip(min=0)
        img_t = impl.contrast(img_t, 15)
        img_t = img_t*mask*mask2
        img_t = cv2.morphologyEx(img_t.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        if np.sum(img_t != 0) > split[s][2]:

            cv2.imwrite('output_diff/diff_for_split_'+str(s)+'-'+str(j)+'-'+str(split[s][0][0] + j)+'.jpg',img_t)
    return np_imgs


# In[10]:
my_split = proccess_split(19)
