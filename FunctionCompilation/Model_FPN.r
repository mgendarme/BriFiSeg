# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
# library(tensorflow)
# library(keras)
# library(tidyverse)
# source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_Zoo.r"))

# testlist = DEFAULT_ENCODER_LAYER[['resnext101']]
# for(i in testlist){
#   print(get_layer(resnext101, i)$output)
# }

## for testing
 # input_shape = c(512, 512, 3)
#  img_input = layer_input(shape = input_shape)
# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')

conv_relu = function(input,
                     num_channel,
                     kernel_size,
                     stride,
                     name,
                     padding='same',
                     use_bias=TRUE,
                     activation='relu'){
  x = layer_conv_2d(input,
                    filters=num_channel,
                    kernel_size=c(kernel_size, kernel_size),
                    strides=stride,
                    padding=padding,
                    kernel_initializer = "he_normal",
                    use_bias=use_bias)
  x = layer_activation(x, activation)
}

prediction_fpn_block= function(x, name, upsample = NULL){
  x = conv_relu(x, 128, 3, stride=c(1,1), name = paste0("prediction_", name ,"_1"))
  x = conv_relu(x, 128, 3, stride=c(1,1), name = paste0("prediction_", name ,"_2"))
  if(!is.null(upsample)){
    x = layer_upsampling_2d(x, size = upsample)
  } else {
    x
  }
}

create_pyramid_features_with_residuals = function(convs,
                                                  feature_size=256,
                                                  decoder_skip = FALSE,
                                                  nlevels = nlevels){
    #  feature_size=256
    # C1 = conv1
    # C2 = conv2
    # C3 = conv3
    # C4 = conv4
    # C5 = conv5
    ps = list()
    up_ps = list()

    ps[[nlevels]] = layer_conv_2d(convs[[nlevels]], 64, kernel_size=1, strides=c(1,1), padding='same', kernel_initializer="he_normal")
    ps[[nlevels]] = layer_conv_2d(convs[[nlevels]], feature_size, kernel_size=1, strides=c(1,1),
                                        padding='same', kernel_initializer="he_normal")
    if(decoder_skip == TRUE){
        ps[[nlevels]] = residual_block(ps[[nlevels]], feature_size, activation = "relu")
    }
    ## extra activation?
    up_ps[[nlevels]] = layer_upsampling_2d(ps[[nlevels]])
    
    for(i in (nlevels-1):1){
        ps[[i]] = layer_conv_2d(convs[[i]], feature_size, kernel_size=1, strides=c(1,1), padding='same', kernel_initializer="he_normal")
        ## dropout?
        ps[[i]] = layer_add(list(up_ps[[i+1]], ps[[i]]))
        ## dropout?
        ps[[i]] = layer_conv_2d(ps[[i]], feature_size, kernel_size=3, strides=c(1,1), padding='same', kernel_initializer="he_normal")
        if(decoder_skip == TRUE){
            ps[[i]] = residual_block(ps[[i]], feature_size, activation = "relu")
        }
        ## extra activation?
        if(i > 1){
            up_ps[[i]] = layer_upsampling_2d(ps[[i]])
        }
    }
    return(ps)
}

conv_bn_relu = function(input, num_channel, kernel_size, stride,
                        name, padding='same', bn_axis=-1, bn_momentum=0.99,
                        bn_scale=TRUE, use_bias=TRUE){
  x = layer_conv_2d(input, filters=num_channel, kernel_size=c(kernel_size, kernel_size),
                    strides=stride, padding=padding,
                    kernel_initializer="he_normal",
                    use_bias=use_bias,
                    name=paste0(name, "_conv"))
  x = layer_batch_normalization(x, name=paste0(name, '_bn'), scale=bn_scale,
                                axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5)
  x = layer_activation(x, 'relu', name=paste0(name, '_relu'))
  x
}

conv_bn = function(input,
                   num_channel,
                   kernel_size,
                   stride,
                   name, 
                   padding='same',
                   bn_axis=-1,
                   bn_momentum=0.99,
                   bn_scale=TRUE,
                   use_bias=TRUE){
  
  x = layer_conv_2d(input, 
                    filters=num_channel, kernel_size=c(kernel_size, kernel_size),
                    strides=stride, padding=padding,
                    kernel_initializer="he_normal",
                    use_bias=use_bias,
                    name=paste0(name, "_conv"))
  x = layer_batch_normalization(x, name=paste0(name, '_bn'),
                                scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5)
  x
  
}

decoder_block = function(input, filters, skip, block_name){
  x = layer_upsampling_2d(input)
  x = conv_bn_relu(x, filters, 3, stride=c(1,1), padding='same', name=paste0(block_name, '_conv1'))
  x = layer_concatenate(list(x, skip), axis=-1, name=paste0(block_name, '_concat'))
  x = conv_bn_relu(x, filters, 3, stride=c(1,1), padding='same', name=paste0(block_name, '_conv2'))
  x
}

decoder_block_no_bn = function(input, filters, skip, block_name, activation='relu'){
  x = layer_upsampling_2d(input)
  x = conv_relu(x, filters, 3, stride=c(1,1), padding='same', name=paste0(block_name, '_conv1'), activation=activation)
  x = layer_concatenate(list(x, skip), axis=-1, name=paste0(block_name, '_concat'))
  x = conv_relu(x, filters, 3, stride=c(1,1), padding='same', name=paste0(block_name, '_conv2'), activation=activation)
  x  
}

# conv2d_bn = function(x,
#                      filters,
#                      kernel_size,
#                      strides=c(1,1),
#                      padding='same',
#                      activation='relu',
#                      use_bias=F,
#                      name=NULL) {
#   x = layer_conv_2d(x,
#                     filters,
#                     kernel_size,
#                     strides=strides,
#                     padding=padding,
#                     use_bias=use_bias,
#                     name=name)
#   if(use_bias == F){
#     bn_axis = ifelse(k_image_data_format() == 'channels_first', 1, -1)
#     bn_name = ifelse(is.null(name), "", paste0(name, "_bn"))
#     x = layer_batch_normalization(x, axis=bn_axis, scale=F, name=bn_name)
#     if(!is.null(activation)){
#       ac_name = ifelse(is.null(name), "", paste0(name, "_ac"))
#       x = layer_activation(x, activation, name=ac_name)
#       x
#     } else {
#       x
#     }
#   } else {
#     x
#   }
# }

# residual_block = function(blockInput, num_filters=16, activation = "relu"){
#   x = layer_activation(blockInput, activation)
#   x = layer_batch_normalization(x)
#   blockInput = layer_batch_normalization(blockInput)
#   x = conv2d_bn(x,num_filters, c(3L,3L), strides=c(1,1), padding='same', activation='relu',
#                 use_bias=F, name=NULL) 
#   x = conv2d_bn(x,num_filters, c(3L,3L), strides=c(1,1), padding='same', activation=NULL,
#                 use_bias=F, name=NULL) 
#   # x = convolution_block(x, num_filters, c(3L,3L), use_bias = T)
#   # x = convolution_block(x, num_filters, c(3L,3L), use_bias = F)
#   x = layer_add(list(x, blockInput))
#   x
# }

######################
## build fpn

build_fpn = function(input_shape,
                    backbone,
                    nlevels=NULL,
                    output_channels=1,
                    output_activation="softmax",
                    decoder_skip = FALSE,
                    dropout = NULL){
  ## for testing
  # input_shape = c(512L, 512L, 3L)
  # output_channels = 3L
  # output_activation = "sigmoid"
  # nlevels = NULL
  # decoder_skip = F
  # backbone = "seresnext101"#""#"efficientnet_B4" #"inception_resnet_v2" #"resnet101"
  
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
           . ) %>%
    ifelse(str_detect(., "senet") == TRUE & str_detect(., "densenet") == FALSE, 
           str_replace(., "senet", "SENet"),
           . ) %>%
    ifelse(str_detect(., "densenet") == TRUE, 
           str_replace(., "densenet", "DenseNet"),
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
  } else if(str_detect(backbone, "senet") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_SeNet.r"))
  } else if(str_detect(backbone, "resnext") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_ResNeXt.r"))
  } else if(str_detect(backbone, "Xception") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_Xception_padding_same.r"))
  }

  ### fetch backbone and layer to extract
  ## build backbone
  if(str_detect(backbone, "nasnet") == TRUE){
      BASE_WEIGHTS_PATH = 'https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/'
      # NASNET_MOBILE_WEIGHT = 'NASNet-mobile.h5'
      NASNET_MOBILE_WEIGHT_NO_TOP = 'NASNet-mobile-no-top.h5'
      # NASNET_LARGE_WEIGHT = 'NASNet-large.h5'
      NASNET_LARGE_WEIGHT_NO_TOP = 'NASNet-large-no-top.h5'
      
      if(str_detect(backbone, "large") == TRUE){
        backbone_model = tf$keras$applications$NASNetLarge(input_shape = input_shape,
                                                           include_top = FALSE,
                                                           weights = NULL)
        file_name = NASNET_LARGE_WEIGHT_NO_TOP
        
      } else if(str_detect(backbone, "mobile") == TRUE){
        backbone_model = tf$keras$applications$NASNetMobile(input_shape = input_shape,
                                                            include_top = FALSE,
                                                            weights = NULL)
        file_name = NASNET_MOBILE_WEIGHT_NO_TOP
        
      } 
      weights_path = get_file(file_name, paste0(BASE_WEIGHTS_PATH, file_name))
      backbone_model %>% load_model_weights_hdf5(filepath = weights_path)
  
  } else if(str_detect(backbone, "densenet") == TRUE){
      if(str_detect(backbone, "121") == TRUE){
        backbone_model = tf$keras$applications$densenet$DenseNet121(input_shape = input_shape,
                                                                    include_top = FALSE)
      }
      if(str_detect(backbone, "169") == TRUE){
        backbone_model = tf$keras$applications$densenet$DenseNet169(input_shape = input_shape,
                                                                    include_top = FALSE)
      }
      if(str_detect(backbone, "201") == TRUE){
        backbone_model = tf$keras$applications$densenet$DenseNet201(input_shape = input_shape,
                                                                    include_top = FALSE)
      }
  } else if(str_detect(backbone_name, "nasnet") == FALSE & str_detect(backbone_name, "densenet") == FALSE){
      backbone_model = do.call(backbone_name, list(input_shape = input_shape,
                                                 include_top = ifelse(backbone == "inception_resnet_v2", FALSE, TRUE),
                                                 weights = "imagenet")) ## finalize this
  }

  # sink("~/Desktop/densenet201.txt")
  # summary(backbone_model)
  # sink()
  
  ## get list of layers
  backbone_layers = DEFAULT_ENCODER_LAYER[[backbone]]
  
  ## Set custom depth for the model
  if(is.null(nlevels) & !is.null(backbone_name)){
    if(backbone != "inception_resnet_v2"){
      nlevels = length(backbone_layers)
    } else {
      nlevels = 5
    }     
  }

  message(paste0("\n########################################################################\n",
                 "Constructing --FPN-- using --", backbone, "-- as encoder with --", nlevels, "--levels",
                 "\n########################################################################"))

  ## extract layers
  convs = list()
  for(i in 1:(nlevels)){
    if(backbone_name != "InceptionResNetV2Same"){
      if(is.numeric(backbone_layers[[i]]) == TRUE){
        convs[[i]] = get_layer(backbone_model, index = backbone_layers[[i]])$output
      } else {
        convs[[i]] = get_layer(backbone_model, name = backbone_layers[[i]])$output
      } 
    } else {
      convs[[i]] = backbone_model$output[[i]]
    } 
    if(i == 1 & str_detect(backbone_name, "nasnet") == TRUE){
      convs[[i]] = layer_zero_padding_2d(convs[[i]], padding = list(list(1,0),list(1,0)))
      int_shape = k_int_shape(convs[[i]])
      convs[[i]] = conv_bn(convs[[i]], rev(int_shape)[[1]], kernel_size = 3L, stride = c(1,1), padding = "same", name = "zeropad")
      convs[[i]] = conv_relu(convs[[i]], rev(int_shape)[[1]], kernel_size = 3L, stride = c(1,1), padding = "same", name = "zeropad")
    }
  }
  
  ## build pyramid features
  ps = create_pyramid_features_with_residuals(convs,
                                              decoder_skip = decoder_skip,
                                              nlevels = nlevels)
  upsample_size = list(c(1,1), c(2,2), c(4,4), c(8,8))

  ## preds
  pred_fpn = list()
  for(i in nlevels:2){
     pred_fpn[[i-1]] = prediction_fpn_block(ps[[i]], paste0("P",i), unlist(upsample_size[i-1]))
     # print(ps[[i]])
  }
  x = layer_concatenate(pred_fpn)                            # n*128*128*128

  ## final steps
  x = conv_bn_relu(x, 256, 3, c(1, 1), name="aggregation")   # n*128*128*256
  
  ## add dropout if necessary  
  if(!is.null(dropout)){
    x = layer_spatial_dropout_2d(x, dropout, name="pyramid_dropout")
  }
  
  # get back the original resolution
  x = decoder_block_no_bn(x, 128, convs[[1]], 'up4')          # n*256*256*128
  x = layer_upsampling_2d(x)                                  # n*512*512*64
  x = conv_relu(x, 64, 3, c(1, 1), name="up5_conv1")          # n*512*512*64
  x = conv_relu(x, 64, 3, c(1, 1), name="up5_conv2")          # n*512*512*64
  output = layer_conv_2d(x,
                         output_channels,
                         kernel_size = c(1,1),
                         strides = c(1,1),
                         padding = 'same')

  if(output_activation == "softmax"){
    output = layer_activation_softmax(output, name = "output_ac_softmax")
    # output = tf$keras$activations$softmax(output)
  } else {
    output = layer_activation(output, output_activation, name = paste0("output_ac_", output_activation))
  }

  # output
  model = keras_model(inputs = backbone_model$input, outputs = output)
  return(model)
}

# test = build_fpn(input_shape = c(512L, 512L, 3L),
#                  backbone = "resnext101",#"inception_resnet_v2",
#                  nlevels=NULL,
#                  output_channels=1,
#                  output_activation="softmax",
#                  decoder_skip = FALSE,
#                  dropout = 0.5)
# 
# sink("~/Desktop/summary_fpn_seresnext101.txt")
# summary(test)
# sink()
