DEFAULT_ENCODER_LAYER <<- list(
	# resnet ------------------------------------------------------------------------------------------------------------------------

	# resnet v1
	"resnet50" = list("conv1_relu",	"conv2_block3_out",	"conv3_block4_out",	"conv4_block6_out",	"conv5_block3_out"),
	"resnet101" = list("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"),
	"resnet152" = list("conv1_relu", "conv2_block3_out", "conv3_block8_out", "conv4_block36_out", "conv5_block3_out"),
	# resnet v2
	"resnet50_v2" = list("conv1_conv", "conv2_block3_1_relu",	"conv3_block4_1_relu", "conv4_block6_1_relu", "post_relu"),
	"resnet101_v2" = list("conv1_conv",	"conv2_block3_1_relu", "conv3_block4_1_relu",	"conv4_block23_1_relu",	"post_relu"),
	"resnet152_v2" = list("conv1_conv",	"conv2_block3_1_relu", "conv3_block8_1_relu", "conv4_block36_1_relu", "post_relu"),
  
	# ResNeXt --------------------------------------------------------------------------------------------------------------------------
	'resnext50' = rev(list('stage5_unit3_relu2', 'stage5_unit1_relu1', 'stage4_unit1_relu1', 'stage3_unit1_relu1', 'relu0')),
	'resnext101' = rev(list('stage5_unit3_relu', 'stage5_unit1_relu1', 'stage4_unit1_relu1', 'stage3_unit1_relu1', 'relu0')),
	
	# SE models ------------------------------------------------------------------------------------------------------------------------
	
	## SE-ResNet
	# 'seresnet18' = rev(list('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')),  			# add 5th layer
	# 'seresnet34' = rev(list('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')),  			# add 5th layer`
	# 'seresnet50' = rev(list(247, 137, 63, 5)),            															# add 5th layer
	# 'seresnet101' =  rev(list(553, 137, 63, 5)),          															# add 5th layer
	# 'seresnet152' = rev(list(859, 209, 63, 5)),           															# add 5th layer
	
	## SE-ResNeXt
	'seresnext50' = rev(list(1327, 1079, 585, 255, 5)),
	'seresnext101' = rev(list(2721, 2473, 585, 255, 5)),
	
	## SE-Net
	'senet154' = rev(list("activation_252", "activation_237", "activation_57", "activation_17", "activation_2")),          															# add 5th layer
	
	# inception_resnet_v2 --------------------------------------------------------------------------------------------------------------
	## requires InceptionResNetV2Same because of padding, can skip this part
	
	# Xception -------------------------------------------------------------------------------------------------------------------------
	# with padding
	"xception" = list("block2_sepconv2_bn",	"block3_sepconv2_bn", "block4_sepconv2_bn",	"block14_sepconv2_bn", "block15_sepconv2_act"),

	# efficientnets --------------------------------------------------------------------------------------------------------------------
	## layers true for efficientnet b0 to b7
	"efficientnet_B0" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B1" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B2" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B3" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B4" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B5" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B6" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
  	"efficientnet_B7" = as.list(paste0('block', c(2, 3, 4, 6, 7),'a_expand_activation')),
	
	## NASNET -------------------------------------------------------------------------------------------------------------------------
	'nasnet_large' = list('activation_3', 'activation_14', 'activation_97', 'activation_180', 'activation_259'),
	'nasnet_mobile' = list('activation_3', 'activation_14', 'activation_73', 'activation_136', 'activation_187'),

	## DenseNet -----------------------------------------------------------------------------------------------------------------------
	'densenet121' = list("conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"),
	'densenet169' = list("conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"),
	'densenet201' = list("conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu")

	## Dual Path Networks -------------------------------------------------------------------------------------------------------------
	# https://github.com/titu1994/Keras-DualPathNetworks/blob/master/dual_path_network.py
	# dpn92 = list("", "", "", "", ""),
	# dpn98 = list("", "", "", "", ""),
	# dpn107 = list("", "", "", "", ""),
	# dpn137 = list("", "", "", "", "")

)
