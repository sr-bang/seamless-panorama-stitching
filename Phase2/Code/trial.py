import cv2
import numpy as np
from random import random

img = cv2.imread('/home/takud/Downloads/WPI_Homework/RBE549/rmnagwekar_p1/Phase2/Data/Train/1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x1 = int((img.shape[1] - 150) * random()) + 10
x2 = x1 + 128
y1 = int((img.shape[0] - 150) * random()) + 10
y2 = y1 + 128

pts = np.array([[x1, y1],[x2, y1],[x2, y2], [x1, y2]], np.float32)

h4pt = []
for _ in range(4):
    h4pt.append([int((random()*20)-10), int((random()*20)-10)])
h4pt = np.array(h4pt, np.float32)

Hab = cv2.getPerspectiveTransform(pts, pts+h4pt)
Hba = np.linalg.inv(Hab)
out = cv2.warpPerspective(img, Hba, (700, 700))

patch_1 = img[y1:y2, x1:x2]
patch_2 = out[y1:y2, x1:x2]

f_img = np.dstack((patch_1, patch_2))
print(f_img.shape)

cv2.imshow('img3', patch_1)
cv2.imshow('img4', patch_2)

# img_pt = cv2.rectangle(img, np.int32(pts[0]), np.int32(pts[2]), color=(255,0,0))
# img_pt = cv2.polylines(img_pt, np.int32([pts+h4pt]), color=(0, 255, 0), isClosed=True)

# out = cv2.polylines(out, np.int32([pts]), color=(255, 0, 0), isClosed=True)
# # out = cv2.polylines(out, np.int32([pts+h4pt]), color=(0, 255, 0), isClosed=True)

# cv2.imshow('img', img_pt)
# cv2.imshow('img2', out)
# print(patch.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
