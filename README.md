# DGFRCNN-RS

This is the PyTorch implementation of the improved version of our paper [Domain Generalized Object Detection for Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/10223771)

## Abstract

Building roof type detection from remotely sensed images is a crucial task for many remote sensing applications, including urban planning and disaster management. In recent years, deep learning-based object detection approaches have demonstrated outstanding performance in this field. However, most of these approaches assume that the training and testing data are sampled from the same distribution. When there are differences between the distributions of training and test data, known as domain shift, the performance significantly degrades. In this paper, we proposed a domain generalization method to address domain shift at the instance and image level for roof type detection from remote sensing images. We incorporated consistency regularization to enforce uniformity between these two levels of domain generalization. Furthermore, we evaluated our proposed method with IEEE Data Fusion Contest 2023 dataset. The proposed approach is the first of its kind in terms of domain generalization for remote sensing object detection.

## Method

![GitHub Logo](/images/figure.png)

## Dataset

In the context of the [IEEE Data Fusion Contest 2023](https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification), a
large-scale remote sensing image dataset has been provided for
the detection of building roof types. The images were collected
from the SuperView-1 and Gaofen-2 satellites, with spatial
resolutions of 0.5m and 0.8m, respectively. The images in the
dataset have a spatial size of 512×512 and were gathered from
seventeen different cities across six continents and are labeled
based on twelve distinct categories of roof types. We selected
three cities in Asia as the validation set, four cities in South
America as the test set, and ten cities from the remaining four
continents as the training set. Each city represents a different
domain. The roof type distribution in the dataset is highly
imbalanced. In order to avoid dealing with the imbalanced
class problem, we utilized this dataset for building detection, 
treating all these twelve distinct categories as a single class 
representing buildings.

## Expriments & Results

|               | image   | instance | consistency | mAP     | mAP@50   | 
|---------------|:-------:|:--------:|:-----------:|:-------:|:--------:|
| Faster R-CNN  |         |          |             |  0.106  |  0.223   |
| DGFRCNN-RS    |   ✓     |          |             |  0.108  |  0.225   | 
| DGFRCNN-RS    |         |     ✓    |             |  0.111  |  0.229   |
| DGFRCNN-RS    |   ✓     |     ✓    |             |  0.112  |  0.231   |
| DGFRCNN-RS    |   ✓     |     ✓    |       ✓     |  **0.115**  |  **0.234**   |
