import cv2
import numpy as np
from random import random
import torch
from Network.Network import dlt

def stack_A(u, v, up, vp):

    arr = np.array([[0, 0, 0, -u, -v, -1, u*vp, v*vp],
                    [u, v, 1, 0, 0, 0, -u*up, -v*up]])

    return arr

img = cv2.imread('/home/takud/Downloads/WPI_Homework/RBE549/rmnagwekar_p1/Phase2/Data/Train/1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x1 = int((img.shape[1] - 150) * random()) + 10
x2 = x1 + 128
y1 = int((img.shape[0] - 150) * random()) + 10
y2 = y1 + 128

Ca = np.array([[x1, y1],[x2, y1],[x2, y2], [x1, y2]], np.float32)

h4pt = []
for _ in range(4):
    h4pt.append([int((random()*20)-10), int((random()*20)-10)])
h4pt = np.array(h4pt, np.float32)

Cb = Ca + h4pt

Hab = cv2.getPerspectiveTransform(Ca, Cb)
Hba = np.linalg.inv(Hab)
out = cv2.warpPerspective(img, Hba, (700, 700))

print(Hab)

Pa = img[y1:y2, x1:x2]
Pb = out[y1:y2, x1:x2]

f_img = np.dstack((Pa, Pb))
# print(f_img.shape)

Hpts = h4pt.flatten()
# print(h4pt, Hpts)

pred = dlt(Hpts, Ca)

print(pred)

# A = []
# B = []

# for i in range(4):
#     Ai = stack_A(Ca[i][0], Ca[i][1], Cb[i][0], Cb[i][1])
#     A.append(Ai)

#     Bi = [-Cb[i][1], Cb[i][0]]
#     B.append(Bi)

# A = np.array(A).reshape((-1, 8))
# B = np.array(B).reshape(8, 1)

# H_new = np.linalg.pinv(A) @ B

# H_new = np.append(H_new, 1)
# H_new = H_new.reshape((3,3))

# print(H_new)

# out2 = cv2.warpPerspective(out, H_new, (700, 700))

# cv2.imshow('img3', Pa)
# cv2.imshow('img4', Pb)

# img_pt = cv2.rectangle(img, np.int32(pts[0]), np.int32(pts[2]), color=(255,0,0))
# img_pt = cv2.polylines(img_pt, np.int32([pts+h4pt]), color=(0, 255, 0), isClosed=True)

# out = cv2.polylines(out, np.int32([pts]), color=(255, 0, 0), isClosed=True)
# # out = cv2.polylines(out, np.int32([pts+h4pt]), color=(0, 255, 0), isClosed=True)

# cv2.imshow('img', img)
# cv2.imshow('img2', out)
# cv2.imshow('img3', out2)
# # print(patch.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
