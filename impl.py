import cv2
import numpy as np
from matplotlib import pyplot as plt


def roi(img):
    
    _, thresh = cv2.threshold(img,127,255,0)
    
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)
    
    area = cv2.contourArea(cnt)
    
    return area

def crop(img, area):
    
    x, y, width, height = cv2.boundingRect(area)
    roi = img[y:y+height, x:x+width]
    
    return roi
    
def align(img, template):
    
    # Find size of image1
    sz = template.shape

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 20

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(template, img, warp_matrix, warp_mode, criteria)

    img_aligned = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return img_aligned

def contrast(img, threshold):
    return (img > threshold) * 255
    
def get_min_width_height(images):
    return tuple([np.min([img.shape[1-i] for img in images]) for i in range(2)])

def show(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap='gray')
    plt.show()