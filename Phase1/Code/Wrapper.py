#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import math
import random
from skimage.feature import peak_local_max
import os

# Add any python libraries here


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    # """
    # Read a set of images for Panorama stitching
    # """

# Add path for images (All sets)

	images = []
	images_path = []
	for i in range(9):

		images_path.append("../Data/Train/Set1/"+str(i+1)+".jpg")
		# images_path.append("../Data/Train/Set2/"+str(i+1)+".jpg")
		# images_path.append("../Data/Train/Set3/"+str(i+1)+".jpg")

		# images_path.append("../Data/Custom/Set1/"+str(i+1)+".jpg")
		# images_path.append("../Data/Custom/Set2/"+str(i+1)+".jpg")

		# images_path.append("../Data/Test/Set1/"+str(i+1)+".jpg")
		# images_path.append("../Data/Test/Set2/"+str(i+1)+".jpg")
		# images_path.append("../Data/Test/Set3/"+str(i+1)+".jpg")
		# images_path.append("../Data/Test/Set4/"+str(i+1)+".jpg")

	for n in images_path:
		temp = cv2.imread(n)
		images.append(temp)
	return images

    # """
	# Corner Detection
	# Save Corner detection output as corners.png
	# """

def corner_detect(image):
	temp=image.copy()
	g_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img = np.float32(g_image)
	corner_d = cv2.cornerHarris(img, 2, 3, 0.04)
	corner = cv2.dilate(corner_d, None)
	threshold1 = corner < 0.001*corner.max()
	threshold2 = corner > 0.001*corner.max()
	corner[threshold1] = 0
	corners_ = np.where(threshold2)
	temp[corners_]=[255,0,0]
	cv2.imshow('img' ,temp)
	cv2.waitKey(0)
	cv2.imwrite("corers.png",temp)
	return corner 



    # """
	# Perform ANMS: Adaptive Non-Maximal Suppression
	# Save ANMS output as anms.png
	# """

def ANMS(image, Cimg, Nbest):
	
	temp_i = image.copy()
	Local_Maxima = peak_local_max(Cimg, min_distance=3)
	m,n = np.shape(Local_Maxima)
	Nstrong = m
	r = [np.inf for i in range(Nstrong)]
	x=np.zeros((Nstrong,))  
	y=np.zeros((Nstrong,)) 
	ED = 0
	best_coordinate=[]
	for i in range(Nstrong):
		for j in range(Nstrong):
			if(Cimg[Local_Maxima[j][0],Local_Maxima[j][1]] > Cimg[Local_Maxima[i][0],Local_Maxima[i][1]]):
				ED = np.square((Local_Maxima[j][0]-Local_Maxima[i][0])) + np.square((Local_Maxima[j][1]-Local_Maxima[i][1]))
				if r[i] > ED:
					r[i] = ED
					x[i] = Local_Maxima[j][0]     ###inverted xj
					y[i] = Local_Maxima[j][1]     ###inverted yj
	index = np.array(np.flip(np.argsort(r)))    ##index for max value of r descending order
	best_index = index[0:Nbest]              ##taking Nbest from descending indices
	if Nbest> Nstrong:
		Nbest = Nstrong
	best_x=np.zeros((Nbest,))
	best_y=np.zeros((Nbest,))  
	for i in range(Nbest):
		best_x[i] = np.int0(x[best_index[i]])        ## y Coordinates of best index
		best_y[i] = np.int0(y[best_index[i]])        ## x Coordinates of best index 
		cv2.circle(temp_i, (int(best_y[i]), int(best_x[i])), 3, 255, -1)
	best_coordinate = np.zeros((Nbest,2))
	best_coordinate[:,0] = best_x
	best_coordinate[:,1] = best_y
	cv2.imshow('img' ,temp_i)
	cv2.waitKey(0)
	cv2.imwrite("anms.png",temp_i)
	return best_coordinate,best_index,Nbest           #club best_x and best_y 


    # """
	# Feature Descriptors
	# Save Feature Descriptor output as FD.png
	# """

def feature(image,Nbest,best_coordinate,patch_size):
	features_des=[]
	s = int((patch_size-1)/2)
	gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	for i in range(Nbest):
		x = int(best_coordinate[i,0])    
		y = int(best_coordinate[i,1])    
		gr_img = (gray_img).copy()
		gr_img = np.pad(gr_img, ((patch_size,patch_size), (patch_size,patch_size)), mode='constant', constant_values=0)
		patch = gr_img[x-s+patch_size:x+s+1+patch_size , y-s+patch_size:y+s+1+patch_size]
		Patch = cv2.GaussianBlur(patch, (5,5), sigmaX=1.0,sigmaY=1.0)  #cv2.BORDER_DEFAULT
		Patch = cv2.resize(Patch, (8, 8)) #,interpolation = cv2.INTER_AREA)
		vector = np.reshape(Patch,(64,))
		var = vector.std()
		if (var==0):
			var=0.0000001
		vector = (vector-vector.mean())/var
		features_des.append(vector)
	cv2.imshow('img' ,features_des)
	cv2.waitKey(0)
	cv2.imwrite("FD.png",features_des)
	return features_des       
  


    # """
	# Feature Matching
	# Save Feature Matching output as matching.png
	# """
def feature_matching(features_des1, features_des2, best_coordinate1, best_coordinate2, req_ratio):

	pair1=[]
	pair2=[]
	Match = []
	min_distance=[]
	for i in range(len(features_des1)):
		distance=[]
		for j in range(len(features_des2)):
			dst=np.sum(np.square(features_des1[i]-features_des2[j]))
			distance.append(dst)
		d=(np.argsort(distance))
		ratio = distance[d[0]]/distance[d[1]]
		if (ratio < req_ratio):
			min_distance.append(distance[d[0]])
			pair1.append(best_coordinate1[i,:])
			pair2.append(best_coordinate2[d[0],:])
			Match.append([pair1,pair2])  		#pair1+pair2
	return Match, pair1,pair2,min_distance

def keypoint(pair):
	keypt = []
	for i in range(len(pair)):
		keypt.append(cv2.KeyPoint(int(pair[i][1]), int(pair[i][0]), 2))
	return keypt

def matches(pair):
	m = []
	for i in range(len(pair)):
		m.append(cv2.DMatch(int(pair[i][1]), int(pair[i][0]), 3))
	return m

def draw_matches(img1, img2 ,pair1, pair2, Match):
	temp_img1 = img1.copy()
	temp_img2 = img2.copy()
	keypoints1 = keypoint(pair1)
	keypoints2 = keypoint(pair2)
	match_index = [(i,i) for i,j in enumerate(Match)]
	matches1to2 = matches(match_index,)
	output = cv2.drawMatches(temp_img1, keypoints1, temp_img2, keypoints2, matches1to2, None,matchColor=(255,255,0))
	return(output)



    # """
	# Refine: RANSAC, Estimate Homography
	# """

def perspective(pair1, pair2):
	pair_1=np.float32(pair1)
	pair_2=np.float32(pair2)
	homo_matrix  = cv2.getPerspectiveTransform(pair_1, pair_2)
	return homo_matrix
  


def RANSAC(Match, threshold, Nmax, pair1, pair2):
    pair1= np.array(pair1)
    pair2 = np.array(pair2)
    Match = np.array(Match)
    inliers = []
    num = []
   
    for i in range(Nmax):
        random_list = random.sample(range(0, len(pair1)), 4)  #4 feature pairs randmoly
        p1 = [pair1[index] for index in random_list]        #accessing 4 pairs
        p2 = [pair2[index] for index in random_list]
        homography=perspective(p1,p2) 
        pts = []
        num_inliers = 0
        for j in range(len(pair1)):
            set2_ = np.array(pair2[j])
            set2 = np.expand_dims(set2_, 1)
            set1_ = np.expand_dims(pair1[j], 1)
            set1 = np.vstack([set1_, 1])
            response = np.dot(homography, set1)
            if (response[2]==0):
                response[2]=0.0000001
            pred = response/response[2]             # runtime error (to ger 3rd element back to 1)
            ssd = np.linalg.norm(set2 - pred[0:2,:])      #(3,1)
            
            if ssd < threshold:
                num_inliers += 1
                pts.append((pair1[j],pair2[j]))
            num.append(num_inliers)
            inliers.append((homography, pts))
    max_idx = np.flip(np.argsort(num))
    max_idx = max_idx[0]
    new_Match = inliers[max_idx][1]
    new_pair1 = np.float32([x[0] for x in new_Match])
    new_pair2 = np.float32([x[1] for x in new_Match])
    
    h_best,_= cv2.findHomography(np.float32(new_pair1),np.float32(new_pair2))
    return h_best, new_Match, new_pair1, new_pair2


    # """
	# Image Warping + Blending
	# Save Panorama output as mypano.png
	# """


def stitch_img(image1 ,image2, h_best):
    new_h = np.array([h_best[1,:],h_best[0,:],h_best[2,:]]).T
    neww = np.array([new_h[1,:],new_h[0,:],new_h[2,:]]).T
    h_best=neww
    height_1, width_1,_= np.shape(image1)
    height_2,width_2,_ = np.shape(image2)
    pt1 = np.array([[0, width_1, width_1, 0], [0, 0, height_1, height_1], [1, 1, 1, 1]])
    p_prime = np.dot(h_best, pt1)
    p_primexy = p_prime / p_prime[2]
    x_row = p_primexy[0]
    y_row = p_primexy[1]
    ymin = min(y_row)
    xmin = min(x_row)
    ymax = max(y_row)
    xmax = max(x_row)
    newh = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])    # removes offset and multiply by homography

    size1=int(xmax-xmin+width_2)
    size2=int(ymax-ymin+height_2) 
    
    size = (size1,size2)
    warped_image = cv2.warpPerspective(image1, newh.dot(h_best), dsize=size)    #Warpigng
    w_x,w_y,_ = warped_image.shape
    xs = 0
    ys = 0 
    if xmin>0:
      xs = int(xmin)
      xmin = 0
    if ymin>0:
      ys = int(ymin)
      ymin = 0
    new_warped = np.zeros((ys + w_x,xs + w_y,3))
    new_warped = new_warped.astype(np.uint8)
    new_warped[ys:ys + w_x,xs:xs + w_y,:] = warped_image
    return new_warped, int(xmin), int(ymin)
  

def blending(images):
    img1 = images[0]
    for i in images[1:]:
        H,_,_,_ = Find_Homography(img1,i)
        temp_save,y_min,x_min = stitch_img(img1,i,H)
        new2 = np.zeros(temp_save.shape)
        new2[abs(x_min):abs(x_min)+i.shape[0],abs(y_min):i.shape[1]+abs(y_min)] = i
        final = temp_save*0.5 + new2*0.5
        final=final.astype(np.uint8)
        
        x = temp_save.shape[0]
        y = temp_save.shape[1]
        for i in range(x):
          for j in range(y):
            if np.sum(temp_save[i,j,:])==0:
              final[i,j,:]=new2[i,j,:]
            if np.sum(new2[i,j,:])==0:
              final[i,j,:]=temp_save[i,j,:]
        img1 = final
        cv2.imshow('img' ,img1)
        cv2.waitKey(0)
    panorama= img1
    cv2.imshow('img' ,panorama)
    cv2.waitKey(0)
    return panorama


#######################################

##Without Blending

def blending(images):
    img1 = images[1]
    for i in images[2:9]:
        H,_,_,_ = Find_Homography(img1,i)
        temp_save,y_min,x_min = stitch_img(img1,i,H)

        temp_save[abs(x_min):abs(x_min)+i.shape[0],abs(y_min):i.shape[1]+abs(y_min)] = i
        img1 = temp_save
        cv2.imshow('img' ,img1)
        cv2.waitKey(0)
    panorama= img1
    cv2.imshow('img' ,panorama)
    cv2.waitKey(0)
    cv2.imwrite("panorama.png", panorama)
    return panorama

########################################


def Find_Homography(img1,img2):
	corner= corner_detect(img1)
	best_coordinate1,best_index,Nbest= ANMS(img1,corner,500)
	features_des1=feature(img1,Nbest,best_coordinate1,41)	
	corner= corner_detect(img2)
	best_coordinate2,best_index,Nbest= ANMS(img2,corner,500) 
	features_des2=feature(img2,Nbest,best_coordinate2,41)	
	Match, pair1,pair2,min_distance=feature_matching(features_des1, features_des2, best_coordinate1, best_coordinate2, 0.70) 

	see = draw_matches(img1, img2 ,pair1, pair2, Match)
	cv2.imshow('img' ,see)
	cv2.waitKey(0)
	cv2.imwrite("matching.png",see)
	if (len(pair1)<4 or len(pair2)<4):
		print("Number of matched pairs is less than 4, insufficient to compute homography matrix.")
		return
	h_best,new_match,new_pair1, new_pair2 = RANSAC(Match, 15, 3000, pair1, pair2) 
	Ransac_see = draw_matches(img1, img2 ,new_pair1, new_pair2, new_match)
	cv2.imshow('img' ,Ransac_see)
	cv2.waitKey(0)
	cv2.imwrite("ransac.png",Ransac_see)
	if (len(new_pair1)<4 or len(new_pair2)<4):
		print("Number of matched pairs is less than 4, insufficient to compute homography matrix.")
		return
	return h_best,new_match,new_pair1, new_pair2
	
images=main()
autopano=blending(images)

if __name__ == "__main__":
    main()

 