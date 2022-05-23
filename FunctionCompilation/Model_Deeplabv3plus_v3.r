## source code used to build our DeepLab V3 +
## https://github$com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/DeepLabV3plus$py
## https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py


# library(keras)
# library(tidyverse)
require(tensorflow)
# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
# source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_ResNet.r"))
# source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_Zoo.r"))

################################################################################
# Layers
################################################################################
ConvolutionBnActivation = function(input_tensor,
                                   filters,
                                   kernel_size,
                                   strides=c(1L, 1L),
                                   padding="same",
                                   data_format=NULL,
                                   dilation_rate=c(1L, 1L),
                                   groups=1L,
                                   activation=NULL,
                                   kernel_initializer="glorot_uniform",
                                   bias_initializer="zeros",
                                   kernel_regularizer=NULL,
                                   bias_regularizer=NULL,
                                   activity_regularizer=NULL,
                                   kernel_constraint=NULL,
                                   bias_constraint=NULL,
                                   use_batchnorm=FALSE, 
                                   axis=-1L,
                                   momentum=0.99,
                                   epsilon=0.001,
                                   center=TRUE,
                                   scale=TRUE,
                                   trainable=TRUE,
                                   post_activation="relu",
                                   block_name=NULL){
                    

    # 2D Convolution Arguments
    args_conv = list(
                    filters = filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    padding = padding,
                    data_format = data_format,
                    dilation_rate = dilation_rate,
                    activation = activation,
                    use_bias = !use_batchnorm, ##### check if correct
                    kernel_initializer = kernel_initializer,
                    bias_initializer = bias_initializer,
                    kernel_regularizer = kernel_regularizer,
                    bias_regularizer = bias_regularizer,
                    activity_regularizer = activity_regularizer,
                    kernel_constraint = kernel_constraint,
                    bias_constraint = bias_constraint
    )

    # Batch Normalization Arguments
    args_bn = list(
                   axis = axis,
                   momentum = momentum,
                   epsilon = epsilon,
                   center = center,
                   scale = scale,
                   trainable = trainable
    )
    
    # Generic arguments
    block_name = block_name
    post_activation = tf$keras$layers$Activation(post_activation)
    
    x = do.call(tf$keras$layers$Conv2D,
                c(args_conv, list(name = ifelse(!is.null(block_name),
                                  paste0(block_name, "_conv"), ""))
                ))(input_tensor)
    x = do.call(tf$keras$layers$BatchNormalization,
                c(args_bn, list(name = ifelse(!is.null(block_name),
                            paste0(block_name, "_bn"), ""))
                ))(x)
    x = post_activation(x)

    return(x)
}


## AtrousSeparableConvolutionBnReLU was replaced by the colution from bonlime
## source
## https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py


AtrousSeparableConvolutionBnReLU = function(input_tensor, 
                                            filters,
                                            kernel_size=3,
                                            stride=1, 
                                            padding="same",
                                            dilation=1, 
                                            depth_layer_activation=FALSE,
                                            epsilon=1e-5,
                                            trainable,
                                            block_name){
    
    # input_tensor = input
    # filters = 256
    # kernel_size= 3
    # stride = 1
    # padding = "same"
    # dilation = 1
    # post_activation = NULL
    # epsilon = 1e-3
    # block_name = "test"

    stride=as.integer(stride) 
    kernel_size=as.integer(kernel_size)
    dilation=as.integer(dilation)
    
    if(stride == 1){
        depth_padding = 'same'
    } else {
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = floor(pad_total / 2)
        pad_end = pad_total - pad_beg
        x = layer_zero_padding_2d(x, list(pad_beg, pad_end))
        depth_padding = 'valid'
    }

    x = input_tensor
    
    if(depth_layer_activation == FALSE){
        x = tf$keras$layers$Activation("relu")(x)
    }
    
    x = tf$keras$layers$DepthwiseConv2D(
            kernel_size=c(kernel_size, kernel_size),
            strides=c(stride, stride),
            dilation_rate=c(dilation, dilation),
            padding=depth_padding,
            use_bias=FALSE,
            name=paste0(block_name, '_depthwise'))(x)
    x = tf$keras$layers$BatchNormalization(name=paste0(block_name, '_depthwise_BN'),
                                  epsilon=epsilon)(x)
    if(depth_layer_activation == TRUE){
        x = tf$keras$layers$Activation("relu")(x)
    }

    x = tf$keras$layers$Conv2D(filters, c(1L, 1L), padding='same',
                         use_bias=FALSE, name=paste0(block_name, '_pointwise'))(x)
    x = tf$keras$layers$BatchNormalization(name=paste0(block_name, '_pointwise_BN'),
                                            epsilon=epsilon)(x)
    if(depth_layer_activation == TRUE){
        x = tf$keras$layers$Activation("relu")(x)
    }
    
    return(x)
}

AtrousSpatialPyramidPoolingV3 = function(input_tensor, atrous_rates, filters, depth_training=FALSE){

    #for testing
    # input_tensor = x
    # atrous_rates = c(6L, 12L, 18L)
    # filters = 256L
    # depth_training = TRUE

    # # global average pooling input_tensor
    # glob_avg_pool = tf$keras$layers$Lambda(function(x) x = tf$reduce_mean(x,
    #                                                                       axis=c(2L, 3L),
    #                                                                       keepdims=TRUE)
    #                                       )(input_tensor)
    # glob_avg_pool = ConvolutionBnActivation(glob_avg_pool,
    #                                         filters=filters,
    #                                         kernel_size=1L,
    #                                         trainable=training)
    # glob_avg_pool = tf$keras$layers$Lambda(function(x) x = tf$image$resize(x,
    #                                                                        c(input_tensor$shape[2],
    #                                                                          input_tensor$shape[3]))
    #                                         )(glob_avg_pool)

    shape_before = tf$keras$backend$int_shape(input_tensor)
  	v = tf$keras$layers$GlobalAveragePooling2D()(input_tensor)
  	v_shape = tf$keras$backend$int_shape(v)
  	
  	# from (b_size, channels)->(b_size, 1, 1, channels)
  	v = tf$keras$layers$Reshape(list(1L, 1L, as.integer(v_shape[[2]])))(v)
  	v = tf$keras$layers$Conv2D(256L, c(1L, 1L), padding='same', use_bias=FALSE, name='image_pooling')(v)
  	v = tf$keras$layers$BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(v)
  	v = tf$keras$layers$Activation("relu")(v)
  	
  	# upsample. have to use compat because of the option align_corners
  	size_before = k_int_shape(input_tensor)
  	v = tf$keras$layers$experimental$preprocessing$Resizing(height = size_before[[2]],
  															 width = size_before[[3]],
  															 interpolation="bilinear")(v)
  	
    # process with atrous
    w = ConvolutionBnActivation(input_tensor,
                                filters=filters,
                                kernel_size=1L,
                                # depth_layer_activation=training,
                                block_name = "aspp_w")
    
    x = AtrousSeparableConvolutionBnReLU(input_tensor,
                                            dilation=atrous_rates[1],
                                            filters=filters,
                                            kernel_size=3L,
                                            depth_layer_activation=depth_training,
                                            block_name = "aspp_x")

    y = AtrousSeparableConvolutionBnReLU(input_tensor,
                                            dilation=atrous_rates[2],
                                            filters=filters,
                                            kernel_size=3L,
                                            depth_layer_activation=depth_training,
                                            block_name = "aspp_y")

    z = AtrousSeparableConvolutionBnReLU(input_tensor,
                                            dilation=atrous_rates[3],
                                            filters=filters,
                                            kernel_size=3L,
                                            depth_layer_activation=depth_training,
                                            block_name = "aspp_z")

    # concatenation
    net = tf$concat(list(v, w, x, y, z), axis=-1L)
    net = tf$keras$layers$Conv2D(filters=256L,
                                 kernel_size=1L,
                                 use_bias=FALSE,
                                 padding='same',
                                 name='concat_projection'
                                 )(net)
    
  	# net = tf$keras$layers$BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(net)
  	# net = tf$keras$layers$Activation("relu")(net)
  	# net = tf$keras$layers$Dropout(rate=0.1)(net)
        
    return(net)
}

# AtrousSpatialPyramidPoolingV3(input, c(6L, 12L, 18L), filters)

################################################################################
# DeepLabV3+
################################################################################
DeepLabV3plus = function(input_tensor,
                        classes,
                        backbone,
                        input_shape=c(512L, 512L, 3L),
                        filters=256L,
                        final_activation="sigmoid",
                        backbone_trainable=TRUE,
                        output_stride=16L,
                        depth=NULL,
                        depth_training=TRUE,
                        dilations=c(6L, 12L, 18L),
                        dropout){
    
    #for testing
    # backbone='resnet101'
    # input_shape=c(512, 512, 3)
    # classes = 3
    # filters=256L
    # final_activation="softmax"
    # backbone_trainable=TRUE
    # output_stride=16L
    # depth_training=TRUE
    # dilations=c(6L, 12L, 18L)

    img_input = layer_input(shape = input_shape)

    ## make correct backbone name
    backbone_name = ifelse(str_detect(backbone, "^resnet") == TRUE,
                            str_replace(backbone, "resnet", "ResNet"),
                            backbone) %>%
        ifelse(str_detect(., "efficientnet") == TRUE,
            str_replace(., "efficientnet", "EfficientNet") %>% str_replace(., "_", ""),
            . ) %>%
        ifelse(str_detect(., "inception_resnet_v2") == TRUE,
            str_replace(., "inception_resnet_v2", "InceptionResNetV2Same"),
            . ) %>%
        ifelse(str_detect(., "xception") == TRUE,
            str_replace(., "xception", "Xception"),
            . ) %>%
        ifelse(str_detect(., "seresnext") == TRUE, 
            str_replace(., "seresnext", "SEResNeXt"),
            . ) %>%
        ifelse(str_detect(., "resnext") == TRUE, 
            str_replace(., "resnext", "ResNeXt"),
            . )

    ## get code for building backbone
  if(str_detect(backbone, "^resnet") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_ResNet.r"))
  } else if(str_detect(backbone, "efficientnet") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_EfficientNet.r"))
  } else if(str_detect(backbone, "inception_resnet_v2") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_InceptionResnetv2_padding_same.r"))
  } else if(str_detect(backbone, "seresnext") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_SeNet.r"))
  } else if(str_detect(backbone, "resnext") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_ResNeXt.r"))
  } else if(str_detect(backbone, "Xception") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_Xception_padding_same.r"))
  }

    ### fetch backbone and layer to extract
    ## build backbone
    backbone_model = do.call(backbone_name, list(input_shape = input_shape,
                                                include_top = ifelse(backbone == "inception_resnet_v2", FALSE, TRUE),
                                                weights = "imagenet")) ## finalize this
    
    ## get list of layers
    backbone_layers = DEFAULT_ENCODER_LAYER[[backbone]]
    output_layers = list()
    for(i in 1:length(backbone_layers)){
        if(backbone_name != "InceptionResNetV2Same"){
                if(is.numeric(backbone_layers[[i]]) == TRUE){
                    output_layers[[i]] = get_layer(backbone_model, index = backbone_layers[[i]])$output
                } else {
                    output_layers[[i]] = get_layer(backbone_model, name = backbone_layers[[i]])$output
                } 
            } else {
                output_layers[[i]] = backbone_model$output[[i]]
            } 
            if(i == 1 & str_detect(backbone_name, "nasnet") == TRUE){
                output_layers[[i]] = layer_zero_padding_2d(output_layers[[i]], padding = list(list(1,0),list(1,0)))
                int_shape = k_int_shape(convs[[i]])
                output_layers[[i]] = conv_bn(output_layers[[i]], rev(int_shape)[[1]],
                                        kernel_size = 3L, stride = c(1,1), padding = "same", name = "zeropad")
                output_layers[[i]] = conv_relu(output_layers[[i]], rev(int_shape)[[1]],
                                        kernel_size = 3L, stride = c(1,1), padding = "same", name = "zeropad")
        }
    }
    
    if(output_stride == 8){
        output_layers = output_layers[1:3]
        for(rate in 1:length(dilations)){
            dilations[rate] = 2 * dilations[rate]
        }
    } else if(output_stride == 16){
        output_layers = output_layers[1:4]
        dilations = dilations
    }
    
    backbone_model$trainable = backbone_trainable
    backbone_simple = tf$keras$Model(inputs=backbone_model$input, outputs=output_layers)

    x = rev(backbone_simple(img_input))[[1]]
    low_level_features = backbone_simple(img_input)[[2]]
    
    # Encoder Module
    ## 1st step too much?
    encoder = x
    # encoder = AtrousSeparableConvolutionBnReLU(x, dilation=2L, filters=filters,
    # 					 kernel_size=3L, depth_training=depth_training, block_name = "encoder_1")
    encoder = AtrousSpatialPyramidPoolingV3(encoder, dilations, filters, depth_training=depth_training)
    encoder = ConvolutionBnActivation(encoder, filters, 1L, block_name = "encoder_2")
    encoder = tf$keras$layers$Dropout(rate=dropout)(encoder)
       
    ## Decoder Module
    
    # upsampling #1
    # size_before_1 = k_int_shape(encoder)
    # encoder = tf$keras$layers$experimental$preprocessing$Resizing(height = size_before_1[[2]],
    #                                                               width = size_before_1[[3]],
    #                                                               interpolation="bilinear")(encoder)
    upsampling_size = ifelse(output_stride == 8, 2L, 4L)
    encoder = tf$keras$layers$UpSampling2D(size = c(upsampling_size, upsampling_size), interpolation = "bilinear")(encoder)
    
    decoder_low_level_features = ConvolutionBnActivation(encoder, filters, 1L, block_name = "decoder_low_1")
    decoder = tf$keras$layers$Concatenate(axis = -1L)(list(decoder_low_level_features, encoder))
    decoder = AtrousSeparableConvolutionBnReLU(decoder, filters, depth_layer_activation=depth_training, block_name = "atrous_decoder_1")
    decoder = AtrousSeparableConvolutionBnReLU(decoder, filters, depth_layer_activation=depth_training, block_name = "atrous_decoder_2")
    decoder = ConvolutionBnActivation(decoder, classes, 1L, post_activation=NULL, trainable=TRUE)
    
    # upsampling #1
    # size_before_2 = k_int_shape(decoder)
    # decoder = tf$keras$layers$experimental$preprocessing$Resizing(height = size_before_2[[2]],
    #                                                               width = size_before_2[[3]],
    #                                                               interpolation="bilinear")(decoder)
    decoder = tf$keras$layers$UpSampling2D(size = as.integer(c(upsampling_size, upsampling_size)), interpolation = "bilinear")(decoder)
    decoder = tf$keras$layers$Activation(final_activation)(decoder)

    model = tf$keras$Model(inputs=img_input, outputs=decoder)
    return(model)
}


###############################################################################################
##########                                   test model                              ##########
###############################################################################################

# backbonemod = create_base_model(name="resnet101", weights="imagenet", height=512, width=512,
#                                   include_top=FALSE, pooling=NULL, alpha=1.0, depth_multiplier=1.0)
# base_model = backbonemod$base_model
# layers = backbonemod$layers
# layer_names = backbonemod$layer_names

# testmod = DeepLabV3plus(input_tensor=NULL,
#                         classes=3L,
#                         backbone="resnet101",
#                         input_shape=c(512L, 512L, 3L),
#                         filters=256L,
#                         final_activation="sigmoid",
#                         backbone_trainable=TRUE,
#                         output_stride=16L,
#                         depth_training=TRUE,
#                         dilations=c(6L, 12L, 18L),
#                         dropout = 0.5)
# summary(testmod)

########### get NASnet large to run

# NAS = tf$keras$applications$NASNetLarge(input_shape = c(513L, 513L, 3L),
#                                         include_top = FALSE,
#                                         weights = NULL)
# # NAS2 = tf$keras$applications$NASNetMobile(input_shape = c(513L, 513L, 3L),
# #                                         include_top = FALSE,
# #                                         weights = NULL)
# BASE_WEIGHTS_PATH = 'https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/'
# NASNET_MOBILE_WEIGHT = 'NASNet-mobile.h5'
# NASNET_MOBILE_WEIGHT_NO_TOP = 'NASNet-mobile-no-top.h5'
# NASNET_LARGE_WEIGHT = 'NASNet-large.h5'
# NASNET_LARGE_WEIGHT_NO_TOP = 'NASNet-large-no-top.h5'
# file_name = NASNET_LARGE_WEIGHT_NO_TOP
# # file_name = NASNET_MOBILE_WEIGHT_NO_TOP
# weights_path = get_file(file_name, paste0(BASE_WEIGHTS_PATH, file_name))
# NAS %>% load_model_weights_hdf5(filepath = weights_path)
# NAS2 %>% load_model_weights_hdf5(filepath = weights_path)
# sink("~/UNet/NASNET_summary.txt")
# summary(NAS2)
# sink()
