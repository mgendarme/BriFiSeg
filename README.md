# BriFiSeg
BriFiSeg is a method developped to perform semantic and instance segmentation of nuclei in brightfield images. 

The method is based on deep convolutional neural network architecture (U-Net mainly, but you can find FPN, PSPNet and Deeplab v3+ too here) using deep pre-trained encoders. We provide plenty of encoders compatible with our archtiectures (right layer sizes). Also ResNeXt, and Squeeze-and-Excitation networks are provided here too! Code not available in R before!

Two different instance segmentation strategies are then offered here to further post-processed the semantic maps into individual instances. One is based on watershed, the other one on connected component analysis. To opt for the watershed-based instance segmentation full nuclear mask are necessary (2 semantic classes: background and nuclei), for connected component analysis center and border of nuclei are required (3 semantic classes, background, border of nuclei, center of nuclei).

Nifti format were used to develop the method. Stick to it to run the method as it is or you will have to modify it accordingly.

Set parameters for size of images, number of semantic classes, batch size, epochs, etc.:
`~/FunctionCompilation/Params.r`

For training and ruinning inference of semantic segmentation task run:
`BriFiSeg_Semantic_Segmentation.r`

For derinving instances and getting metrics on instances run:
`BriFiSeg_Instance_Segmentation.r`

Example of semantic segmentation of nuclei in brightfield images using U-Net and different pre-trained encoders.
![plot](https://github.com/mgendarme/BriFiSeg/blob/main/Example/Segmentation/gt_pred_bf_test_class_2_4.png)

Example of instance segmetation using watershed based instance segmentation with semantic maps as input. 
![plot](https://github.com/mgendarme/BriFiSeg/blob/main/Example/Segmentation/Montage_2.png)

List of compatible pre-trained encoders:
  - resnet50, resnet101, resnet152,
  - resnet 50 v2, resnet101 v2, resnet152v2,
  - resnext50, resnext101
  - seresnext 50, seresnext 101, senet154
  - xception
  - inception_resnet_v2
  - efficientnet B0 to efficientnet B7
  - nasnet_mobile, "nasnet_large
  - densenet121, "densenet169, densenet201
