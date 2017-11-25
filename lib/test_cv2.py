import cv2
import numpy as np

pixels=np.zeros((10,10))
pixels[1:5,1:5]=0.4*255
cv2.imwrite('test.jpg',pixels)
