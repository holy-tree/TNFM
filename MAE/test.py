import numpy as np
import cv2

# from skimage.metrics import structural_similarity as ssim

ck20 = cv2.imread('./mask1.jpg')
ck40 = cv2.imread('./mask.jpg')

chayi = ck20 - ck40

cv2.imwrite('./chayi.jpg', chayi)