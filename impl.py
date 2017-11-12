import cv2
import numpy as np
from matplotlib import pyplot as plt


def contour(img):
    
    _, thresh = cv2.threshold(img,127,255,0)
    
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)
    
    return cnt

def crop(img, area):
    
    x, y, width, height = cv2.boundingRect(area)
    roi = img[y:y+height, x:x+width]
    
    return roi


def align(img, def_pos, templates):
    rows,cols = img.shape
    res = [cv2.matchTemplate(img, t, cv2.TM_CCOEFF) for t in templates]
    max_loc = [cv2.minMaxLoc(r) for r in res]
    pos = [np.array([m_l[0], m_l[1]]) for m_l in max_loc]
    
    #translate
    trans = def_pos[0] - pos[0]
    M = np.array([[1,0,trans[0]],[0,1,trans[1]]])
    dst = cv2.warpAffine(img, M, img.shape)
    #rotation
    angle = angle_between(pos[1] - pos[0], def_pos[1] - def_pos[0])
    M = cv2.getRotationMatrix2D((def_pos[0][0], def_pos[0][1]), angle, 1)
    dst2 = cv2.warpAffine(dst, M, img.shape)
    return dst2

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def old_align(img, template):
    
    # Find size of image1
    sz = template.shape

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (_, warp_matrix) = cv2.findTransformECC(template, img, warp_matrix, warp_mode)

    img_aligned = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return img_aligned

def contrast(img, threshold):
    return (img > threshold) * 255
    
def get_min_width_height(images):
    return tuple([np.min([img.shape[1-i] for img in images]) for i in range(2)])

def show(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    
def mask_contour(img, perc=0.98):
    cnt = contour(img)
    full_mask = np.zeros(img.shape, dtype=np.uint8)
    tmp1 = cv2.drawContours(full_mask, [cnt], 0, 255, -1)
    w, h = tmp1.shape
    tmp2 = cv2.resize(tmp1, (int(h * perc), int(w * perc)), interpolation=cv2.INTER_AREA)
    tmp3 = np.zeros_like(tmp1)
    smaller_w, smaller_h = tmp2.shape
    w_dif = w - smaller_w
    h_dif = h - smaller_h
    tmp3[w_dif // 2:smaller_w + w_dif // 2, h_dif // 2:smaller_h + h_dif // 2] = tmp2
    mask = tmp3 // 128
    return mask