---
title: Image_Manipulation_detection
emoji: 🚀
colorFrom: pink
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Image manipulation detection

Image manipulation detection is different from traditional semantic object detection because it pays more attention to tampering artifacts than to image content, which suggests that richer features need to be learned. In order to detect image manipulation, we have used different models and since they are trained on a variety of different data sources we have to build a robust system to detect image manipulation and forgery. 

We have used two models: ManTraNet and BusterNet. Both models have a very different architecture and when we upload an image to our UI we can see the predictions for both of the models. Now we will be describing each model in detail.

## ManTraNet

ManTraNet is an end-to-end image forgery detection and localization solution, which means it takes a testing image as input and predicts pixel-level forgery likelihood map as output. Compared to existing methods, the proposed ManTraNet has the following advantages:

- Simplicity: ManTraNet needs no extra pre-and/or post-processing
- Fast: ManTraNet puts all computations in a single network, and accepts an image of arbitrary size.
- Robustness: ManTraNet does not rely on working assumptions other than the local manipulation assumption, i.e. some region in a testing image is modified differently from the rest.

### Dataset

ManTraNet is pretrained with all synthetic data. To prevent overfitting, we
Pre train the Image Manipulation Classification (385 classes) task to obtain the Image Manipulation Trace Feature Extractor
Train ManTraNet with four types of synthetic data, i.e. copy-move, splicing, removal, and enhancement
To extend the provided ManTraNet, one may introduce the new manipulation either to the IMC pre-train task, to the end-to-end ManTraNet task, or both. It is also worth noting that the IMC task can be a self-supervised task.

### Model Architecture

![image](https://user-images.githubusercontent.com/49101362/153304277-bbb6a852-df1b-41e7-b486-4cda6e0e3c30.png)

Technically speaking, ManTraNet is composed of two sub-networks as shown below:
Image Manipulation Trace Feature Extractor: the feature extraction network for the image manipulation classification task, which is sensitive to different manipulation types, and encodes the image manipulation in a patch into a fixed dimension feature vector.
Local Anomaly Detection Network: the anomaly detection network to compare a local feature against the dominant feature averaged from a local region, whose activation depends on how far a local feature deviates from the reference feature instead of the absolute value of a local feature.

![image](https://user-images.githubusercontent.com/49101362/153304401-225e1d08-734b-4f28-9ed8-173e16ad894c.png)


## BusterNet

We introduce a novel deep neural architecture for image copy-move forgery detection (CMFD), code-named BusterNet. Unlike previous efforts, BusterNet is a pure, end-to-end trainable, deep neural network solution. It features a two-branch architecture followed by a fusion module. The two branches localize potential manipulation regions via visual artifacts and copy-move regions via visual similarities, respectively. To the best of our knowledge, this is the first CMFD algorithm with discernibility to localize source/target regions.

### Datasets

In BusterNet they have used custom-made CASIA-CMFD, CoMoFoD-CMFD, and USCISI-CMFD datasets. 

#### CASIA-CMFD Dataset

This copy-move forgery detection(CMFD) dataset is made upon the original CASIA-Tide-V2 dataset by

select copy-move samples only
obtain the target copy by thresholding the difference between the manipulated image and its original
obtain the source copy by matching the target copy on the manipulated image using SIFT/ORB/SURF features
manually verify all obtained masks

In the end, we have 1313 positive CMFD samples (i.e. with copy-move forgery). The corresponding original images are used as negative samples. It is worth to mention that there are 991 unique negative samples, because some positive samples point to the same original image.

#### CoMoFoD-CMFD Dataset

This copy-move forgery detection(CMFD) dataset is made upon the original CoMoFoD dataset by

select copy-move samples only
obtain the target copy by thresholding the difference between the manipulated image and its original
obtain the source copy by matching the target copy on the manipulated image using SIFT/ORB/SURF features
manually verify all obtained masks

#### USCISI-CMFD Dataset

This copy-move forgery detection(CMFD) dataset relies on

MIT SUN2012 Database
MS COCO Dataset
More precisely, we synthesize a copy-move forgery sample using the following steps

select a sample in the two above dataset
select one of its object polygon
use both sample image and polgyon mask to synthesize a sample

### Model

![image](https://user-images.githubusercontent.com/49101362/153304435-17e313d7-08ce-43a9-9ca4-835cc765751e.png)
![image](https://user-images.githubusercontent.com/49101362/153304459-b364e2c0-408b-45c8-95d7-dbe50be195ac.png)




### Model architecture 

![image](https://user-images.githubusercontent.com/49101362/153304475-3495369e-67cd-4512-ae9b-0242accb828c.png)


## Examples and Demonstration 

Demo can be found here: https://huggingface.co/spaces/gagan3012/Image_Manipulation_detection









citation
```
@inproceedings{wu2018eccv,
  title={BusterNet: Detecting Image Copy-Move Forgery With Source/Target Localization},
  author={Wu, Yue, and AbdAlmageed, Wael and Natarajan, Prem},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018},
  organization={Springer},
}

  @inproceedings{Wu2019ManTraNet,
      title={ManTra-Net: Manipulation Tracing Network For Detection And Localization of Image ForgeriesWith Anomalous Features},
      author={Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan},
      journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019}
  }
```




