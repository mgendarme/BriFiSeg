# library(devtools)
# install.packages("devtools")
# devtools::install_github("rstudio/keras")
# library(keras)
# library(tidyverse)
# install_keras()
# tensorflow::tf_config()
## based on segmentation models
## https://github.com/qubvel/segmentation_models/blob/master/

RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/BFSeg', '~/Documents/BFSeg')
# RelPath = "D:/WRK/DL/UNet10"

# from keras_applications import get_submodules_from_kwargs
# 
# from ._common_blocks import Conv2dBn
# from ._utils import freeze_model, filter_keras_submodules
# from ..backbones.backbones_factory import Backbones

backend = NULL
layers = NULL
models = NULL
keras_utils = NULL

## common blocks

# Conv2dBn = function(
#   input_tensor,
#   filters,
#   kernel_size,
#   strides=c(1, 1),
#   padding='valid',
#   # data_format=NULL,
#   # dilation_rate=c(1L, 1L),
#   activation=NULL,
#   kernel_initializer='glorot_uniform',
#   # bias_initializer='zeros',
#   # kernel_regularizer=NULL,
#   # bias_regularizer=NULL,
#   # activity_regularizer=NULL,
#   # kernel_constraint=NULL,
#   # bias_constraint=NULL,
#   use_batchnorm=FALSE
#   # **kwargs
#   ){
  
#   # """Extension of Conv2D layer with batchnorm"""
  
#   ## for testing
#   # input_tensor_test = x
#   # filters = 512
#   # kernel_size=3
#   # strides=c(1, 1)
#   # padding='valid'
#   # dilation_rate=c(1, 1)
#   # kernel_initializer='glorot_uniform'
#   # bias_initializer='zeros'
#   # kernel_regularizer=NULL
#   # bias_regularizer=NULL
#   # activity_regularizer=NULL
#   # kernel_constraint=NULL
#   # bias_constraint=NULL
#   # use_batchnorm=FALSE

#   conv_name = NULL
#   act_name = NULL
#   bn_name = NULL
#   block_name = paste0("name", NULL)
  
#   if(!is.null(block_name)){
#     conv_name = paste0(block_name, '_conv')
#   }
  
#   if(!is.null(block_name) & !is.null(activation)){
#     act_str = activation
#     act_name = paste0(block_name, '_', act_str)
#   }
    
  
#   if(!is.null(block_name) & use_batchnorm == TRUE){
#     bn_name = paste0(block_name, '_bn')
#   }
  
#   bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  
#   # wrapper = function(input_tensor){
#     x = layer_conv_2d(
#       input_tensor,
#       filters=filters,
#       kernel_size=kernel_size,
#       strides=strides,
#       padding=padding,
#       # data_format=data_format,
#       # dilation_rate=dilation_rate,
#       activation=NULL,
#       use_bias=FALSE,
#       kernel_initializer=kernel_initializer,
#       # bias_initializer=bias_initializer,
#       # kernel_regularizer=kernel_regularizer,
#       # bias_regularizer=bias_regularizer,
#       # activity_regularizer=activity_regularizer,
#       # kernel_constraint=kernel_constraint,
#       # bias_constraint=bias_constraint,
#       name=conv_name)
    
#     if(use_batchnorm == TRUE){
#       x = layer_batch_normalization(x, axis = bn_axis, name = bn_name)
#     } else {
#       x
#     }
      
      
#     if(activation == TRUE){
#       x = layer_activation(x, activation, name = act_name)
#     } else {
#       x
#     }
        
#     return(x)
    
#   # }
  
#   # return(wrapper)
  
# }


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------
# 
# def get_submodules():
#   return {
#     'backend': backend,
#     'models': models,
#     'layers': layers,
#     'utils': keras_utils,
#   }
# 
# 
# def check_input_shape(input_shape, factor):
#   if input_shape is NULL:
#   raise ValueError("Input shape should be a tuple of 3 integers, not NULL!")
# 
# h, w = input_shape[:2] if backend.image_data_format() == 'channels_last' else input_shape[1:]
# min_size = factor * 6
# 
# is_wrong_shape = (
#   h % min_size != 0 or w % min_size != 0 or
#   h < min_size or w < min_size
# )
# 
# if is_wrong_shape:
#   raise ValueError('Wrong shape {}, input H and W should '.format(input_shape) +
#                      'be divisible by `{}`'.format(min_size))


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

# Conv1x1BnReLU = function(input_tensor, filters, use_batchnorm, name=NULL){
#   # wrapper = function(input_tensor){
#     return(
#       Conv2dBn(
#         input_tensor,
#         filters,
#         kernel_size=1,
#         activation='relu',
#         kernel_initializer='he_uniform',
#         padding='same',
#         use_batchnorm,
#         name#,
#         # **kwargs
#       )
#     )
#   # }
    
#   # return(wrapper)
  
# }


conv1bnRelu = function(input_tensor, filters, use_batchnorm, activation="relu", name=NULL){
  conv_name = NULL
  act_name = NULL
  bn_name = NULL
  block_name = ifelse(!is.null(name), name, "name")
  
  if(!is.null(block_name)){ conv_name = paste0(block_name, '_conv') }
  
  if(!is.null(block_name) & !is.null(activation)){
    act_str = activation
    act_name = paste0(block_name, '_', act_str)
  }
    
  
  if(!is.null(block_name) & use_batchnorm == TRUE){ bn_name = paste0(block_name, '_bn') }
  
  bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  
  x = layer_conv_2d(
      input_tensor,
      filters=filters,
      kernel_size=1,
      strides=1,
      padding='same',
      use_bias=FALSE,
      kernel_initializer='he_normal',
      name=conv_name)
    
    if(use_batchnorm == TRUE){
      x = layer_batch_normalization(x, axis = bn_axis, name = bn_name)
    } else {
      x
    }
      
      
    if(activation == TRUE){
      x = layer_activation(x, activation, name = act_name)
    } else {
      x
    }
        
    return(x)
}

# conv1bnRelu(x, 512, use_batchnorm=TRUE, name="test")

SpatialContextBlock = function(input_tensor,
                               level,
                               conv_filters=512,
                               pooling_type='avg',
                               use_batchnorm=TRUE){
  # if(!(pooling_type %in% c('max', 'avg'))){
    ## stop and return error message
    # ('Unsupported pooling type - `{}`.'.format(pooling_type) +
    #                    'Use `avg` or `max`.')
  # }
  # 
  ## for testing 
  #  input_tensor = x

  # level = 3
  # conv_filters=512
  # pooling_type='avg'
  # use_batchnorm=TRUE

  # pooling_type = 'max'
  Pooling2D = ifelse(pooling_type == 'max', layer_max_pooling_2d, layer_average_pooling_2d)
  
  pooling_name = paste0('psp_level', level,'_pooling')
  conv_block_name = paste0('psp_level', level)
  upsampling_name = paste0('psp_level', level, '_upsampling')
  
  # wrapper = function(input_tensor){
    # extract input feature maps size (h, and w dimensions)
    input_shape = unlist(k_int_shape(input_tensor))
    if(backend()$image_data_format() == "channels_last"){
      spatial_size = input_shape[1:3]
    } else {
      spatial_size = input_shape[2:4]
    }
    
    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel factor and strides are equal, then we can compute the final feature map factor
    # by simply dividing the current factor by the kernel or stride factor
    # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6.
    pool_size = list(floor(spatial_size[1] / level), floor(spatial_size[2] / level))
    up_size = pool_size
    
    x = Pooling2D(input_tensor, pool_size, strides=pool_size, padding='same', name=pooling_name)
    # x = Conv1x1BnReLU(x, conv_filters, use_batchnorm=use_batchnorm, name=conv_block_name)
    x = conv1bnRelu(x, conv_filters, use_batchnorm = use_batchnorm, name = conv_block_name)
    x = layer_upsampling_2d(x, up_size, interpolation='bilinear', name=upsampling_name)
    
    return(x)
  # }
    
  # return(wrapper)
  
}

# input_tensor = layer_input(shape = c(512, 512, 3))

# ---------------------------------------------------------------------
#  PSP Decoder
# ---------------------------------------------------------------------

build_psp = function(
  backbone,
  psp_layer_idx,
  pooling_type='avg',
  conv_filters=512,
  use_batchnorm=TRUE,
  final_upsampling_factor=8,
  classes=21,
  activation='softmax',
  dropout=NULL){
  
  ## for testing 
  # psp_layer_idx
  # pooling_type='avg'
  # conv_filters=512
  # use_batchnorm=TRUE
  # final_upsampling_factor=4
  # classes=21
  # activation='softmax'
  # dropout=NULL

  input_ = backbone$input
  x = get_layer(backbone, name=psp_layer_idx[[1]])$output 
  
  # build spatial pyramid
  x1 = SpatialContextBlock(x, 1, conv_filters, pooling_type, use_batchnorm)
  x2 = SpatialContextBlock(x, 2, conv_filters, pooling_type, use_batchnorm)
  x3 = SpatialContextBlock(x, 3, conv_filters, pooling_type, use_batchnorm)
  x6 = SpatialContextBlock(x, 6, conv_filters, pooling_type, use_batchnorm)
  
  # aggregate spatial pyramid
  concat_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  x = layer_concatenate(list(x, x1, x2, x3, x6), axis=concat_axis, name='psp_concat')
  # x = Conv1x1BnReLU(x, conv_filters, use_batchnorm, name='aggregation')
  x = conv1bnRelu(x, conv_filters, use_batchnorm, name='aggregation')
  # model regularization
  if(!is.null(dropout)){
    x = layer_spatial_dropout_2d(x, dropout, name='spatial_dropout')
  }
    
  # model head
  x = layer_conv_2d(
    x,
    filters=classes,
    kernel_size=c(3, 3),
    padding='same',
    kernel_initializer='glorot_uniform',
    name='final_conv')
  
  x = layer_upsampling_2d(x, final_upsampling_factor, name='final_upsampling', interpolation='bilinear')
  x = layer_activation(x, activation, name=activation)
  
  model = keras_model(input_, x)
  
  return(model)
  
}



# ---------------------------------------------------------------------
#  PSP Model
# ---------------------------------------------------------------------

PSPNet = function(
  backbone_name='resnet101',
  input_shape=c(384, 384, 3),
  classes=21,
  activation='softmax',
  # weights=NULL,
  # encoder_weights='imagenet',
  encoder_freeze=FALSE,
  downsample_factor=8,
  psp_conv_filters=512,
  psp_pooling_type='avg',
  psp_use_batchnorm=TRUE,
  psp_dropout=NULL,
  nlevels=NULL
  # **kwargs
  ){
  # """PSPNet_ is a fully convolution neural network for image semantic segmentation
  #   Args:
  #       backbone_name: name of classification model used as feature
  #               extractor to build segmentation model.
  #       input_shape: shape of input data/image ``(H, W, C)``.
  #           ``H`` and ``W`` should be divisible by ``6 * downsample_factor`` and **NOT** ``NULL``!
  #       classes: a number of classes for output (output shape - ``(h, w, classes)``).
  #       activation: name of one of ``keras.activations`` for last model layer
  #               (e.g. ``sigmoid``, ``softmax``, ``linear``).
  #       weights: optional, path to model weights.
  #       encoder_weights: one of ``NULL`` (random initialization), ``imagenet`` (pre-training on ImageNet).
  #       encoder_freeze: if ``TRUE`` set all layers of encoder (backbone model) as non-trainable.
  #       downsample_factor: one of 4, 8 and 16. Downsampling rate or in other words backbone depth
  #           to construct PSP module on it.
  #       psp_conv_filters: number of filters in ``Conv2D`` layer in each PSP block.
  #       psp_pooling_type: one of 'avg', 'max'. PSP block pooling type (maximum or average).
  #       psp_use_batchnorm: if ``TRUE``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
  #               is used.
  #       psp_dropout: dropout rate between 0 and 1.
  #   Returns:
  #       ``keras.models.Model``: **PSPNet**
  #   .. _PSPNet:
  #       https://arxiv.org/pdf/1612.01105.pdf
  #   """
  
  # global backend, layers, models, keras_utils
  # submodule_args = filter_keras_submodules(kwargs)
  # backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)
  
  # control image input shape
  # check_input_shape(input_shape, downsample_factor)
  
  ## FOR TESTING
  # backbone_name = "resnet101"
  # input_shape = c(384, 384, 3)
  # nlevels = NULL
  # downsample_factor = 4
  # psp_conv_filters=512
  # psp_pooling_type='avg'
  # psp_use_batchnorm=TRUE
  # psp_dropout=NULL
  
  ## get list of layers
  backbone_layers = DEFAULT_ENCODER_LAYER[[backbone_name]]#[(nlevels-2):nlevels]
  
  ## get code for building backbone
  if(str_detect(backbone_name, "^resnet") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/ResNets_keras_MG.r"))
  } else if(str_detect(backbone_name, "efficientnet") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/EfficientNets_keras_MG.r"))
  } else if(str_detect(backbone_name, "inception_resnet_v2") == TRUE){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/InceptionResnetv2_MG.r"))
  } ## add line for xception
  
  # ## make correct backbone name
  backbone_name = ifelse(str_detect(backbone_name, "^resnet") == TRUE,
                         str_replace(backbone_name, "resnet", "ResNet"),
                         backbone_name) %>%
    ifelse(str_detect(., "efficientnet") == TRUE,
           str_replace(., "efficientnet", "EfficientNet") %>% str_replace(., "_", ""),
           . ) %>%
    ifelse(str_detect(., "inception_resnet_v2") == TRUE,
           str_replace(., "inception_resnet_v2", "InceptionResNetV2Same"),
           . ) %>%
    ifelse(str_detect(., "xception") == TRUE,
           str_replace(., "xception", "Xception"),
           . )
  # backbone = Backbones.get_backbone(
  #   backbone_name,
  #   input_shape=input_shape,
  #   weights=encoder_weights,
  #   include_top=FALSE,
  #   **kwargs
  # )
  
  backbone = do.call(backbone_name, list(input_shape = input_shape,
                                         include_top = ifelse(backbone_name == "inception_resnet_v2", FALSE, TRUE),
                                         weights = "imagenet"))
  
  ## Set custom depth for the model
  if(is.null(nlevels) & !is.null(backbone_name)){
    if(backbone_name != "inception_resnet_v2"){
      nlevels = length(backbone_layers) -2
    } else {
      nlevels = 3
    }     
  }
  # nlevels
  
  message(paste0("\n########################################################################\n",
                 "Constructing --PSPnet-- using --", backbone_name, "--as encoder with --", nlevels, "--levels",
                 "\n########################################################################"))
  
  if(downsample_factor == 16){
    psp_layer_idx = backbone_layers[3+1]
  } else if (downsample_factor == 8){
    psp_layer_idx = backbone_layers[2+1]
  } else if( downsample_factor == 4){
    psp_layer_idx = backbone_layers[1+1]
  } else {
    ##   error message
    # raise ValueError('Unsupported factor - `{}`, Use 4, 8 or 16.'.format(downsample_factor))
  }
    
  model = build_psp(
    backbone,
    psp_layer_idx,
    pooling_type=psp_pooling_type,
    conv_filters=psp_conv_filters,
    use_batchnorm=psp_use_batchnorm,
    final_upsampling_factor=downsample_factor,
    classes=classes,
    activation=activation,
    dropout=psp_dropout
  )
  
  # lock encoder weights for fine-tuning
  # if encoder_freeze:
  #   freeze_model(backbone, **kwargs)
  # 
  # # loading model weights
  # if weights is not NULL:
  #   model.load_weights(weights)
  
  return(model)
  
}

# testpsp = PSPNet(backbone_name='resnet101',
#                  input_shape=c(384*1.5, 384*1.5, 3),
#                  classes=3,
#                  activation='sigmoid',
#                  # weights=NULL,
#                  # encoder_weights='imagenet',
#                  encoder_freeze=FALSE,
#                  downsample_factor=4, # c(4, 8, 16)
#                  psp_conv_filters=512,
#                  psp_pooling_type='avg',
#                  psp_use_batchnorm=TRUE,
#                  psp_dropout=0.5)

# summary(testpsp)
