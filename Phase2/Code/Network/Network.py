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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def photo_loss(H, Ia):

    img_ht = Ia.shape[1]
    img_wth = Ia.shape[2]

    M = torch.tensor([[img_wth/2, 0, img_wth/2],
                  [0, img_ht/2, img_ht/2],
                  [0, 0, 1]]).to(device)
    
    Hinv = torch.inverse(M) @ torch.inverse(H) @ M
    Hinv = Hinv.cpu().detach().numpy()

    wPa_f = []
    Pb_f = []
    for i in range(Ia.size()[0]):
        Pa = Ia[i, :, :, 0].cpu().detach().numpy()
        Pb = Ia[i, :, :, 1].cpu().detach().numpy()
        wPa = cv2.warpPerspective(Pa, Hinv[i], Pb.shape)
        wPa_f.append(wPa)
        Pb_f.append(Pb)

    wPa_f = torch.tensor(wPa_f, requires_grad=True).to(device)
    Pb_f = torch.tensor(Pb_f, requires_grad=True).to(device)

    criterion = nn.L1Loss()
    loss = criterion(wPa_f, Pb_f)
    return loss

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()
        # torch.save(self.model.state_dict(), 'model.pth')

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

def conv(in_c, out_c):

    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

def stack_A(u, v, up, vp):

    arr = torch.tensor([[0, 0, 0, -u, -v, -1, u*vp, v*vp],
                    [u, v, 1, 0, 0, 0, -u*up, -v*up]])

    return arr

def DLT(h4pt_batch, Ca):
    # print(h4pt_batch.size())

    # h4pt_batch = h4pt_batch.detach().numpy()
    H_mat = []

    for i in range(h4pt_batch.shape[0]):

        h4pt = h4pt_batch[i, :]

        Cb = torch.add(torch.tensor(Ca).to(device), h4pt.view(torch.Size([4, 2])))

        A = []
        B = []

        for i in range(4):
            Ai = stack_A(Ca[i][0], Ca[i][1], Cb[i][0], Cb[i][1])
            A.append(Ai)

            Bi = [-Cb[i][1], Cb[i][0]]
            B.extend(Bi)

        A = torch.stack(A).view(torch.Size([8, 8])).to(device)
        B = torch.tensor(B).to(device)

        H_new = torch.inverse(A) @ B

        H_new = torch.cat((H_new, torch.tensor([1]).to(device))).to(device)
        H_new = H_new.view(torch.Size([3, 3]))

        H_mat.append(H_new)

    return torch.stack(H_mat).to(device)

class Net(nn.Module):
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