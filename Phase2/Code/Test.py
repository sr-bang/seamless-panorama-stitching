#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import HomographyModel, LossFn, DLT, photo_loss
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import torch
import tensorflow as tf


# Don't generate pyc codes
sys.dont_write_bytecode = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()
    
    x1 = int((I1.shape[1] - 150) * random.random()) + 10
    x2 = x1 + 128
    y1 = int((I1.shape[0] - 150) * random.random()) + 10
    y2 = y1 + 128

    Ca = np.array([[x1, y1],[x2, y1],[x2, y2], [x1, y2]], np.float32)

    h4pt = []
    for _ in range(4):
        h4pt.append([int((random.random()*20)-10), int((random.random()*20)-10)])
    h4pt = np.array(h4pt, np.float32)

    Cb = Ca + h4pt

    Hab = cv2.getPerspectiveTransform(Ca, Cb)
    Hba = np.linalg.inv(Hab)
    out = cv2.warpPerspective(I1, Hba, (700, 700))

    patch_1 = I1[y1:y2, x1:x2]
    patch_2 = out[y1:y2, x1:x2]

    f_img = np.float32(np.dstack((patch_1, patch_2)))

    f_img = torch.from_numpy(f_img)
    f_img = f_img.view((1, 128, 128, 2))

    # print(f_img.shape)

    return f_img.to(device), torch.tensor(h4pt.flatten()).to(device), Hab, Ca


def TestOperation(ModelPath, TestSet, BasePath, ModelType):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyModel().to(device)
    model.eval()

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    preds = []
    labels = []
    loss_total = 0
    for count in tqdm(range(len(TestSet))):

        img = cv2.imread(BasePath + TestSet[count] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Img, h4pt_og, Hab, Ca = ReadImages(img)

        h4pt_pred = model(Img)
        if ModelType == 'Unsup':
            H_mat = DLT(h4pt_pred, Ca)
            loss = photo_loss(H_mat, Img)
        else:
            labels.append(h4pt_og)
            preds.append(h4pt_pred.clone().detach())
            loss = LossFn(h4pt_pred, h4pt_og).item()
        loss_total += loss
        print('Loss_{}:{}'.format(count+1, loss))
        print('Loss_avg:', loss_total/(count+1))
        torch.cuda.empty_cache()

    return h4pt_og.cpu().detach().numpy(), h4pt_pred.cpu().detach().numpy(), Hab, Ca

def showImg(img_no, ModelPath, BasePath, ModelType):

    img = cv2.imread('../Data/Test/{}.jpg'.format(img_no))
    h4pt_og, h4pt_pred, Hab, Ca = TestOperation(ModelPath, ['Test/{}'.format(img_no)], BasePath, ModelType)

    # print(h4pt_og.reshape((4,2)).shape, Ca.shape)
    h4pt_og = h4pt_og.reshape((4,2))
    h4pt_pred = h4pt_pred.reshape((4,2))

    # img_og = cv2.warpPerspective(img, Hab, (700, 700))
    # img_pred = cv2.warpPerspective(img, pred, (700, 700))
    img = cv2.polylines(img, np.int32([np.add(Ca, h4pt_og)]), color=(255, 0, 0), isClosed=True)
    img = cv2.polylines(img, np.int32([np.add(Ca, h4pt_pred)]), color=(0, 0, 255), isClosed=True)

    cv2.imshow('img', img)
    # cv2.imshow('img1', img_og)
    # cv2.imshow('img2', img_pred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../../Checkpoints/Sup2.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="/home/takud/Downloads/WPI_Homework/RBE549/rmnagwekar_p1/Phase2/Data/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    ModelType = Args.ModelType

    DirNamesTest = [('Test/'+ str(x)) for x in range(1,1001)]
    TestOperation(ModelPath, DirNamesTest, BasePath, ModelType)
    # showImg(317, ModelPath, BasePath, ModelPath)

if __name__ == "__main__":
    main()
