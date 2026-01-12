# CryoET-Membrane-Seg
Membrane segmentation for Cryo Electron Tomograms using UNet

## Overview
This research project features a 2.5D UNet for membrane segmentation in cryo-electron tomography (cryo ETs). Traditionally, cryo ETS are very large, complicated, 3D images of cells used to visualize cell structures including membranes. However, membrane annotations can take a very long time and are sparse and expensive. This research project seeks to learn where membranes exist in a cryo ET to better visualize cell membranes.



This project uses patching from a cryo ET to obtain useful data where membrane annotations exist. Then this data is fed through a UNet built using PyTorch Convolutional layers. Then the model is trained using randomly sampled patches measured by loss function with both BCE and dice components. Finally, inference is done on a full tomogram by tiling and stitching to manage large data and visualized.


## Data processing and patching
Cryo ET data are too large to process, so we create patches that contain useful information to train from.

Preprocessing
-	Tomograms and masks are loaded as ```.mrc``` files
-	The tomogram is normalized robustly focusing between 5th and 95th percentile bounds
-	Masks are binarized, if not already

Data splitting
-	To prevent data leak, split the tomogram into layers and ensure that layers are split among training, validation, and testing groups via z-slabs

Patching
-	Since annotations are sparse, and training data has 512 x 1024 x 1024 voxels, it is not efficient to train on the entire dataset. Most of these voxels only contain background information. Therefore, we train where membrane annotations exist
-	Each patch has size 256 x 256 with 7 channels
-	With probability 0.8, we sample a patch where the patch is anchored on some membrane, else we sample a random background patch.


## Model

This model is a 2D UNet modified for 2.5D analysis. 2.5D was used as this was the most computational feasible way to train large cryo ET files.

<img width="341" height="255" alt="Screenshot 2026-01-13 at 7 21 50 AM" src="https://github.com/user-attachments/assets/c209f931-6aea-47f0-b567-a20d45bd58b9" />


Standard UNet was used with 4 down (encoding) layers and 4 up (decoding layers) with 3x3 convolutions. Each convolution block consists of 2 convolution layers as seen in the diagram (first adjusting channels, then convolution on new channels). MaxPool of factor 2 was used, and base channel of 32 was used. For now, this is just a standard UNet. Skip connections were constructed via torch.cat.

## Training

Loss function

<img width="588" height="35" alt="Screenshot 2026-01-13 at 7 22 17 AM" src="https://github.com/user-attachments/assets/35f98333-d5d8-4e65-9ac4-8b8ba1a6761d" />

This loss function contains both elements of BCE and Dice loss. BCE encourages stable per voxel probability estimates, while Dice loss directly optimizes overlap on sparse, thin membrane annotations, making their combination well suited for membrane segmentation.

Training
-	First load data using DataLoader from PyTorch
-	Fix number of randomly sampled patches per Epoch
-	Validation loss is used to select the best loss and state
-	Test evaluation is conducted once training is completed

Loss

<img width="400" height="300" alt="Screenshot 2026-01-13 at 7 23 04 AM" src="https://github.com/user-attachments/assets/91657d73-a536-4f3f-a905-90cf08a57b66" />



<img width="277" height="60" alt="Screenshot 2026-01-13 at 7 23 16 AM" src="https://github.com/user-attachments/assets/c6951739-5567-403f-aa89-a81fc5bd925c" />

For an initial run without GPU with low sampling and less epochs, loss was acceptable during training. However, testing loss is a bit higher, which indicates a bit of overfitting. Please see further considerations to address this issue.


## Inferences
After saving the weights from training, we obtain a mask of membrane 

Tiling + stitching
-	For each layer, we tile into PATCH_SIZE by PATCH_SIZE tiles. Using our model, we output the probability distribution of membrane existence. 
-	We stitch each of these tiles back together for each layer, and end up with a giant 3D tensor of probabilities
-	Create mask where probability > 0.5 are considered membranes and less are not.
-	Using MatPlotLib, visualizations are created

Below are before slices of the annotation

<img width="556" height="172" alt="Screenshot 2026-01-13 at 7 24 36 AM" src="https://github.com/user-attachments/assets/d4692eb1-f399-4fb3-8b55-b76ea36ddacd" />
   
Below are slices of mask obtained through training model of the same tomogram
   
<img width="557" height="175" alt="Screenshot 2026-01-13 at 7 25 03 AM" src="https://github.com/user-attachments/assets/5ed7692a-7e8a-45f6-9f94-43a7b18ea5dc" />


Below is a 3D visual of the annotation

<img width="319" height="285" alt="Screenshot 2026-01-13 at 7 25 16 AM" src="https://github.com/user-attachments/assets/92e7612f-03ff-471b-9556-c792629e0a3e" />


Below is a 3D visual of mask obtained through model of the same tomogram
 
<img width="299" height="244" alt="Screenshot 2026-01-13 at 7 25 38 AM" src="https://github.com/user-attachments/assets/cf8b8bd9-add2-4641-b331-a535333154c3" />


Below is inference on a separate tomogram
  
<img width="555" height="263" alt="Screenshot 2026-01-13 at 7 25 51 AM" src="https://github.com/user-attachments/assets/56c6ee3b-1aa3-4a30-bee3-ca619f172b42" />


As observed, even after training, there seems to be pretty sparse membrane predictions. Below are possible ways to better the model.

## Further considerations
-	Data augmentation
-	Transition to 3D convolutions
o	2.5D was used to manage the computational load, but it would be interesting to see how 3D convolutions could change results
-	Multi tomogram training
-	Random offset in patching
o	Each patch is centered off some membrane voxel. However, this may not help because convolution does not take location into account.
-	There is some overfitting. Perhaps adding nn.Dropout or nn.BatchNorm or nn.GroupNorm to the UNet architecture.


## How to run
First, ensure that everything in config.py is up to par. Download the training data from below, and put it in the data directory. Make sure to update config.py with any new file paths.

Then run

```python train.py```

To inference and visualize, run
```python inference.py --tomo [DATA FILE PATH]```

(I have already saved the weights in output, so there is no need to train again, if inference is the goal.)


## Data
All data was obtained from the public database CryoET Data Portal. The Chlamydomonas Reinhardtii was used, and specifically, 01082023_BrnoKrios_Arctis_WebUI_Position_20 was used as training data.

Training data: https://cryoetdataportal.czscience.com/runs/14122?table-tab=Tomograms0 

Download tomogram for the tomogram, and download membrane annotations under annotions.

 

