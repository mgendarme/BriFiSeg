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
