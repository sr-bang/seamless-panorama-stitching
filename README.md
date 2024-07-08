# Seamless Panorama Stitching (MyAutopano)

## Phase 1 : Geometrical computer vision
### The code can be executed using the wrapper.py from the Phase1 folder.

#### 1. Input Data: Sequence of images
<p float="left">
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/e3cf413f-fc69-43b7-84e3-764243d69a66" width="210" />
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/0345f923-d870-4dee-9034-3a3418a20f61" width="210" />
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/32f0bfe3-df91-42b3-a0bc-b9a6a8c6bb8f" width="210" />
</p>

#### 2. Corner Detection and Adaptive Non-Maximal Suppression (ANMS):

<p float="left">
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/004757be-cc01-4ed9-9c81-465349112c7b" width="320" />
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/6ac85493-4b62-4875-ba8f-63174c92bcd8" width="320" />
</p>
<p float="left">
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/f5b50a09-8aa3-4036-8c3b-747c51eb89a2" width="320" />
  <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/f4b01346-27fb-4b54-9b0c-3cb60db81787" width="320" />
</p>
  
#### 3. Feature Matching

<p float="left">
    <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/602c9db7-9870-4566-8f4f-093faf910c35"/> 
</p>

##### After RANSAC

<p float="left">
    <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/cfe0c54d-7bc7-47bd-99c8-2945add88d09"/> 
</p>

##### 4. Panorama Stiching

<p float="left">
    <img src="https://github.com/takud1/seamless-panorama-stitching/assets/124741847/4127b0ad-fa2d-4260-9bbc-63c3e12242bd" width="800" /> 
</p>

## Phase2

### The models can be trained using the train.py file and can be tested using the test.py file. Both the files default to the Supervised version of the algorithm and must be provided the modeltype argument to change to the unsupervised version. Similarly, both the files must be given the path for checkpoints.
