"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import cv2
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    criterion = nn.MSELoss()
    loss = criterion(out, labels)
    return loss

def LossFnUnsup(H, Pa, Pb):

    wPa = cv2.warpPerspective(Pa, H, Pa.shape)
    loss = np.linalg.norm(wPa-Pb)
    return loss

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net_Sup()
        # torch.save(self.model.state_dict(), 'model.pt')

    def forward(self, a):
        return self.model(a)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, imgs, labels):
        # img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(imgs)
        loss = LossFn(delta, labels)
        print("Validation loss:", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class UnsupModel(pl.LightningModule):
    def __init__(self):
        super(UnsupModel, self).__init__()
        # self.hparams = hparams
        self.model = Net_Unsup()
        # torch.save(self.model.state_dict(), 'model.pt')

    def forward(self, a):
        return self.model(a)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFnUnsup(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, H, Pa, Pb):
        # img_a, patch_a, patch_b, corners, gt = batch
        # delta = self.model(imgs)
        loss = LossFnUnsup(H, Pa, Pb)
        print("Validation loss:", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

def conv(in_c, out_c):

    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

def stack_A(u, v, up, vp):

    arr = np.array([[0, 0, 0, -u, -v, -1, u*vp, v*vp],
                    [u, v, 1, 0, 0, 0, -u*up, -v*up]])

    return arr

def dlt(h4pt, Ca):

    Cb = np.add(Ca, h4pt.reshape(4, 2))

    A = []
    B = []

    for i in range(4):
        Ai = stack_A(Ca[i][0], Ca[i][1], Cb[i][0], Cb[i][1])
        A.append(Ai)

        Bi = [-Cb[i][1], Cb[i][0]]
        B.append(Bi)

    A = np.array(A).reshape((-1, 8))
    B = np.array(B).reshape(8, 1)

    H_new = np.linalg.pinv(A) @ B

    H_new = np.append(H_new, 1)
    H_new = H_new.reshape((3,3))

    return H_new

class Net_Sup(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1 = conv(2, 64)
        self.conv2 = conv(64, 64)
        self.conv3 = conv(64, 128)
        self.conv4 = conv(128, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 128, 1024)
        self.fc2 = nn.Linear(1024, 8)
        self.flat = nn.Flatten()

    def forward(self, x):

        #############################
        # Fill your network structure of choice here!
        #############################

        mini_batch_size = x.shape[0]
        dim_x = x.shape[1]
        dim_y = x.shape[2]
        depth = x.shape[3]

        x = x.view(torch.Size([mini_batch_size, depth, dim_x, dim_y]))
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.dropout(x)

        x = self.flat(x)
        x = self.fc1(x)
        out = self.fc2(x)

        return out

class Net_Unsup(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):

        #############################
        # Fill your network structure of choice here!
        #############################

        mini_batch_size = x.shape[0]
        dim_x = x.shape[1]
        dim_y = x.shape[2]
        depth = x.shape[3]

        x = x.view(torch.Size([mini_batch_size, depth, dim_x, dim_y]))


        return x