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
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import Adam, lr_scheduler
from Network.Network import HomographyModel, LossFn, photo_loss, DLT
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def GenerateBatch(BasePath, DirNamesTrain, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # print(RandImageName, RandIdx)
        I1 = cv2.imread(RandImageName)
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        # Coordinates = TrainCoordinates[RandIdx]

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

        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(f_img))
        CoordinatesBatch.append(torch.tensor(h4pt.flatten()))

        # print(torch.stack(I1Batch).size())

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device), Ca


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel().to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    if ModelType == 'Unsup':
        Optimizer = Adam(model.parameters(), lr=0.0001)
    else:
        Optimizer = Adam(model.parameters(), lr=0.01)

    sched = lr_scheduler.MultiStepLR(Optimizer, milestones=[2, 4, 6, 8], gamma=0.5)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    DirNamesVal = [('Val/'+ str(x)) for x in range(1,1001)]

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch, Ca = GenerateBatch(
                BasePath, DirNamesTrain, MiniBatchSize
            )
            model.train()
            # Predict output with forward pass
            # torch.onnx.export(model, I1Batch, 'model.onnx')
            PredicatedCoordinatesBatch = model(I1Batch)
            h = PredicatedCoordinatesBatch

            if ModelType == 'Unsup':
                H_mat = DLT(h, Ca)
                LossThisBatch = photo_loss(H_mat, I1Batch)
                # print(LossThisBatch)
                # LossThisBatch = Variable()

            else:
                LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            print("Loss:", LossThisBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            model.eval()
            with torch.no_grad():
                # print(validation_set_image_names)
                val_batch, val_labels, Ca_val = GenerateBatch(
                    BasePath, DirNamesVal, MiniBatchSize
                )

            if ModelType == 'Unsup':
                h_pred = model(val_batch)
                H_val = DLT(h_pred, Ca_val)
                val_loss = photo_loss(H_val, val_batch)
                print('Val Loss:', val_loss)
            else:
                result = model.validation_step(val_batch, val_labels)

            # Tensorboard
            Writer.add_scalar(
                "ValLossEveryIter",
                (val_loss if ModelType == 'Unsup' else result["val_loss"]),
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.add_scalar(
                "TrainLossEveryIter",
                LossThisBatch.item(),
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        sched.step()
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/takud/Downloads/WPI_Homework/RBE549/rmnagwekar_p1/Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="./Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=10,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=20,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
