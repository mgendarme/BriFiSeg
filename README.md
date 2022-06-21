# BriFiSeg
BriFiSeg is a method developped to perform semantic and instance segmentation of nuclei in brightfield images. 

The method is based on deep convolutional neural network architecture (U-Net mainly, but FPN, PSPNet and Deeplab v3+ can also be found) using deep pre-trained encoders. We provide plenty of encoders compatible with our architectures. Also ResNeXt, and Squeeze-and-Excitation networks were written in R and tested here. Code not available in R before!

Two different instance segmentation strategies were tested here to further post-processed the semantic maps into individual instances. One is based on watershed, the other one on connected component analysis. To opt for the watershed-based instance segmentation full nuclear mask are necessary (2 semantic classes: background and nuclei), for connected component analysis center and border of nuclei are required (3 semantic classes, background, border of nuclei, center of nuclei).

Nifti format were used to develop the method. Stick to it to run the method as it is or you will have to modify it accordingly.

Set [parameters](https://github.com/mgendarme/BriFiSeg/FunctionCompilation/Params.r) for size of images, number of semantic classes, batch size, epochs, choice of architecture, encoder, etc.:
`~/FunctionCompilation/Params.r`

For training and ruinning inference of [semantic segmentation](https://github.com/mgendarme/BriFiSeg/BriFiSeg_Semantic_Segmentation.r) task run:
`BriFiSeg_Semantic_Segmentation.r`

For running [instance segmentation](https://github.com/mgendarme/BriFiSeg/BriFiSeg_Instance_Segmentation.r) derived from semantic maps and getting metrics on instances:
`BriFiSeg_Instance_Segmentation.r`

Example of semantic segmentation of nuclei in brightfield images using U-Net SE ResNeXt 101.
![plot](https://github.com/mgendarme/BriFiSeg/blob/main/Example/Segmentation/gt_pred_bf_test_class_2_4.png)
(from left to right, input image, ground truth, prediction)

Example of instance segmentation using watershed based instance segmentation with semantic maps as input. 
![plot](https://github.com/mgendarme/BriFiSeg/blob/main/Example/Segmentation/Montage_2.png)
(from left to right, input image, ground truth, prediction)

List of compatible pre-trained encoders:
  - ResNet 50, ResNet 101, ResNet 152,
  - ResNet 50 v2, ResNet 101 v2, ResNet 152 v2
  - ResNeXt 50, ResNeXt 101
  - SE ResNeXt 50, SE ResNeXt 101, SENet 154
  - Xception
  - Inception ResNet v2
  - EfficientNet B0 to EfficientNet B7
  - NASNet mobile, NASNet large
  - DenseNet 121, DenseNet 169, DenseNet 201
