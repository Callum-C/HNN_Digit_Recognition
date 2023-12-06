from PIL import Image
import cv2
import numpy as np

fname = 'Digits/0.png'

img = cv2.imread(fname, 0)

img_norm = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

print(img_norm)

cv2.imshow('Normalized Image', img_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()