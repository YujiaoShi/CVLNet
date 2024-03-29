# CVLNet
ACCV2022 oral "CVLNet: Cross-View Feature Correspondence Learning for Video-based Camera Localization"
![alt text](./VBL_Framework.png)

# Abstract
In this paper, we tackle cross-view video-based camera localization. The task is to determine the pose of a ground-based camera from a continuous sequence of images by matching them to a large overhead satellite image. 
To this end, a Cross-view Video-based Localization Deep Neural Network, dubbed CVLNet, is developed to establish semantic correspondences between the ground-view video sequence and the satellite image using cross-view geometry between the two views. 
Specifically, we devise a Geometry-driven View Projection (GVP) module, a Photo-consistency Constrained Sequence Fusion (PCSF) module, and a Scene-prior driven Similarity Matching (SSM) strategy in the CVLNet.  The GVP is designed to align ground-view deep features to overhead-view satellite features in the same domain.  The PCSF module takes a continuous video as input and outputs a fused global representation for the sequential ground-view observations. 
This significantly improves the discriminate power of the learned ground-view descriptors. 
Moreover, our SSM strategy estimates the displacement between a query camera location and the center of the retrieved satellite image, leading to accurate localization. To train the CVLNet, we collect satellite images from Google Map for the KITTI dataset and construct a novel cross-view video-based localization benchmark dataset, named KITTI-CVL. Extensive experiments have confirmed the effectiveness of our method. 
### Experiment Dataset
Our experiments is conducted on the KITTI dataset. 
Please first download the raw data (ground images) from http://www.cvlibs.net/datasets/kitti/raw_data.php, and store them according to different date (not category). 
For our collected satellite images for both datasets, please first fill this [Google Form](https://forms.gle/61V7fVR2FE5emHDM6), we will then send you the link for download. 

Your dataset folder structure should be like: 

KITTI:

  raw_data:
  
    2011_09_26:
    
      2011_09_26_drive_0001_sync:
      
        image_00:
	
	image_01:
	
        image_02:
	
        image_03:
	
        oxts:
	
      ...
      
    2011_09_28:
    
    2011_09_29:
    
    2011_09_30:
    
    2011_10_03:
  
  
  satmap:
  
    train_10mgap:
    
      2011_09_26:
      
      2011_09_29:
      
      2011_09_30:
      
      2011_10_03:
      
    test:
    
      2011_09_26_drive_0002_sync
      
      2011_09_26_drive_0005_sync
      
      2011_09_26_drive_0015_sync
      
      2011_09_26_drive_0036_sync
      
      2011_09_26_drive_0046_sync
      
      2011_09_30_drive_0016_sync
      
      2011_09_30_drive_0034_sync

### Codes

#### Training 

python train.py 

#### Testing 
python test.py 

In the test.py, "test_wo_destractors=True" indicates testing on both Test1 and Tes2; "test_wo_destractors=False" indicates only test on Test2. 

### Models:
Our trained model is available [here](https://anu365-my.sharepoint.com/:u:/g/personal/u6293587_anu_edu_au/EVVeze8yhfpGpsrpuAJDpP0BqS86odkmeeQyu1rdDBJOSA?e=I9Hsk7). 



### Publications
This work is published in ACCV 2022.  
[CVLNet: Cross-View Semantic Correspondence Learning for Video-based Camera Localization]
