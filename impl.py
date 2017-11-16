import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_show(img):
    """
    Plot a grayscale image using matplotlib
    """
    
    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
    plt.show()

def crop(img, area):
    """
    Crop the image given an area
    """
    
    x, y, width, height = cv2.boundingRect(area)
    roi = img[y:y+height, x:x+width]
    return roi


def contrast(img, threshold):
    """
    Constrast all pixels of an image given a threshold.
    All pixels smaller or equal will be 0 and the other will be 255
    """
    return (img > threshold) * 255

def angle_between(p1, p2):
    """
    Compute the angle between two points
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def fast_align(img, def_pos, templates):
    """
    Align an image given templates and their positions
    """
    
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

def find_affine_transform_matrix(img, template):
    """
    Compute the affine transformation matrix to go from img to template
    """
    
    if(len(img.shape) == 3):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (_, warp_matrix) = cv2.findTransformECC(template, img_gray, warp_matrix, warp_mode)
    return warp_matrix
    
def affine_transform(img, warp_matrix):
    """
    Apply affine transformation to an image
    """

    sz = img.shape
    return = cv2.warpAffine(img, warp_matrix, (sz[1], sz[0])), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
def accurate_align(img, template):
    """
    Align an image given a template image using affine transformations (found with the ECC algorithm)
    """
    warp_matrix = find_affine_transform_matrix(img, template)
    return affine_transform(img, warp_matrix)
    
def find_contours(img):
    """
    Find all contours in the image
    """
    
    _, thresh = cv2.threshold(img,127,255,0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_biggest_contour(img):
    """
    Find the biggest contour (biggest by area) in the image
    """
    
    contours = find_contours(img)
    cnt = max(contours, key = cv2.contourArea)
    return cnt

def mask_contour(img, perc=0.99):
    """
    Create a mask on the area outside the biggest contour of an image.
    A shrink of the contour area can be done to be sure the whole contour is masked.
    """
    
    cnt = biggest_contour(img)
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