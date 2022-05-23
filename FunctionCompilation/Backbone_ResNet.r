# library(keras)
# library(tidyverse)
 
BASE_WEIGHTS_PATH = list(
  'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = list(
  'resnet50' = c('2cb95161c43110f7111970584f804107',
                '4d473c1dd8becc155b73f8504c6f6626'),
  'resnet101'= c('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                '88cf7a10940856eca736dc7b7e228a21'),
  'resnet152'= c('100835be76be38e30d865e96f2aaae62',
                'ee4c566cf9a93f14d82f913c2dc6dd0c'),
  'resnet50v2'= c('3ef43a0b657b3be2300d5770ece849e0',
                 'fac2f116257151a9d068a22e544a4917'),
  'resnet101v2'= c('6343647c601c52e1368623803854d971',
                  'c0ed64b8031c3730f411d2eb4eea35b5'),
  'resnet152v2'= c('a49b44d1979771252814e80f8ec446f9',
                  'ed17cf2e0169df9d443503ef94b23b33'),
  'resnext50'= c('67a5b30d522ed92f75a1f16eef299d1a',
                '62527c363bdd9ec598bed41947b379fc'),
  'resnext101'= c('34fb605428fcc7aa4d62f44404c11509',
                  '0f678c91647380debd923963594981b3')
)

layers = NULL

ResNet = function(stack_fn,
       preact, #T/F
       use_bias, #T/F
       model_name='resnet',
       include_top=TRUE,
       Weights='imagenet',
       input_tensor=NULL,
       input_shape=NULL,
       pooling=NULL, # "avg" or "max" or NULL
       classes=1000,
       classifier_activation='softmax'
       # **kwargs
       ){
  
  # global layers
  # if 'layers' in kwargs:
  #   layers = kwargs.pop('layers')
  # else:
  #   layers = VersionAwareLayers()
  # if kwargs:
  #   raise ValueError('Unknown argument(s): %s' % (kwargs,))
  # if not (weights in {'imagenet', NULL} or tf.io.gfile.exists(weights)):
  #   raise ValueError('The `weights` argument should be either '
  #                    '`NULL` (random initialization), `imagenet` '
  #                    '(pre-training on ImageNet), '
  #                    'or the path to the weights file to be loaded.')
  # 
  # if weights == 'imagenet' and include_top and classes != 1000:
  #   raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
  #                    ' as TRUE, `classes` should be 1000')
  
  
  
  ## for testing
  # input_shape=c(256, 256, 3)
  # preact=NULL
  # use_bias=TRUE
  # model_name='resnet152'
  # include_top=TRUE
  # weights='imagenet'
  # input_tensor=NULL
  # pooling=NULL
  # classes=1000
  # classifier_activation='softmax'
  
  # stack_fn
  # preact = FALSE
  # use_bias = TRUE
  # model_name = 'resnet152'
  # include_top = F
  # Weights = "imagenet"
  # input_tensor = NULL
  # input_shape = c(256, 256, 3)
  # pooling = NULL 
  # classes = 1000
  
  # Determine proper input shape
  
  # input_shape = imagenet_utils.obtain_input_shape(
  #   input_shape,
  #   default_size=224,
  #   min_size=32,
  #   data_format=backend.image_data_format(),
  #   require_flatten=include_top,
  #   weights=weights)
  
  if(is.null(input_tensor)){
    img_input = layer_input(shape=input_shape)
  } else {
    if(is.null(backend()$is_keras_tensor(input_tensor))){
      img_input = layer_input(tensor=input_tensor, shape=input_shape)
    } else {
      img_input = input_tensor
    }
  } 
  
  bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  # img_input = layer_input(shape = input_shape)
  
  x = layer_zero_padding_2d(img_input, padding= c(3, 3), name='conv1_pad')
  x = layer_conv_2d(x, 64, 7, strides=2, use_bias=use_bias, name='conv1_conv')
  
  if(preact == FALSE){
    x = layer_batch_normalization(x, axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')
    x = layer_activation(x, 'relu', name='conv1_relu')
  }
  
  x = layer_zero_padding_2d(x, padding=c(1, 1), name='pool1_pad')
  x = layer_max_pooling_2d(x, 3, strides=2, name='pool1_pool')
  
  # x = block1(x, filters = 64, stride = 1, name=paste0("test", '_block1'))
  # for(i in 2:(3+1)){ ### triple check if indices are fine
  #   x = block1(x, filters = 64, conv_shortcut=FALSE, name=paste0("test", '_block', as.character(i)))
  #   print(i)
  #   x
  # }
  
  # stack_fn = function(x){
  #   x = stack1(x, filters = 64, blocks = 3, stride1=1, name='conv2')
  #   x = stack1(x, filters = 128, blocks = 8, name='conv3')
  #   x = stack1(x, filters = 256, blocks = 36, name='conv4')
  #   x = stack1(x, filters = 512, blocks = 3, name='conv5')
  #   return(x)
  # }
  x = stack_fn(x)
  
  if(preact == TRUE){
    x = layer_batch_normalization(x, axis=bn_axis, epsilon=1.001e-5, name='post_bn')
    x = layer_activation(x, 'relu', name='post_relu')
  }
  
  if(include_top == T){
    x = layer_global_average_pooling_2d(x, name='avg_pool')
    # imagenet_utils.validate_activation(classifier_activation, weights) ## check how to do that
    x = layer_dense(x, classes, activation=classifier_activation, name='probs')
  } else {
    if(!is.null(pooling)){
      if(pooling == 'avg'){
        x = layer_global_average_pooling_2d(x, name='avg_pool')
      } else if(pooling == 'max'){
        x = layer_global_max_pooling_2d(x, name='max_pool')
      } else {
        x
      }
    }
    x
  } 
  
  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  # if(!is.null(input_tensor)){
  #   # inputs = layer_utils.get_source_inputs(input_tensor) ## check
  # } else {
  #   inputs = img_input
  #   inputs
  # } 
  
  inputs = img_input

  # Create model.
  model = keras_model(inputs, x)
  # length(model$layers)
  # summary(model)
  
  # Load weights.
  if(Weights == 'imagenet' & model_name %in% names(WEIGHTS_HASHES)){
    if(include_top == TRUE){
      file_name = paste0(model_name, '_weights_tf_dim_ordering_tf_kernels.h5')
      file_hash = WEIGHTS_HASHES[model_name][[1]][1]
    } else {
      file_name = paste0(model_name, '_weights_tf_dim_ordering_tf_kernels_notop.h5')
      file_hash = WEIGHTS_HASHES[model_name][[1]][2]    
    }
    weights_path = get_file(file_name,
                            paste0(BASE_WEIGHTS_PATH, file_name))
    model %>% load_model_weights_hdf5(filepath = weights_path)
  } else if(!is.null(Weights)){
    model %>% load_model_weights_hdf5(Weights)
  }
  
  model
  
}

block1 = function(x, filters, kernel_size=3, stride=1, conv_shortcut=TRUE, name=NULL){
  # """A residual block
  # Arguments:
  #   x: input tensor.
  #   filters: integer, filters of the bottleneck layer.
  #   kernel_size: default 3, kernel size of the bottleneck layer.
  #   stride: default 1, stride of the first layer.
  #   conv_shortcut: default TRUE, use convolution shortcut if TRUE,
  #       otherwise identity shortcut.
  #   name: string, block label.
  # Returns:
  #   Output tensor for the residual block.
  # """
  bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)

  if(conv_shortcut == TRUE){
    shortcut = layer_conv_2d(x, 
      4 * filters, 1, strides=stride, name=paste0(name, '_0_conv'))
    shortcut = layer_batch_normalization(shortcut,
      axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_0_bn'))
  } else {
    shortcut = x
  }
  
  x = layer_conv_2d(x, filters, 1, strides=stride, name=paste0(name, '_1_conv'))
  x = layer_batch_normalization(x, 
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_1_bn'))
  x = layer_activation(x, 'relu', name=paste0(name, '_1_relu'))
  
  x = layer_conv_2d(x, filters, kernel_size, padding='SAME', name=paste0(name, '_2_conv'))
  x = layer_batch_normalization(x,
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_2_bn'))
  x = layer_activation(x, 'relu', name=paste0(name, '_2_relu'))
  
  x = layer_conv_2d(x, 4 * filters, 1, name=paste0(name, '_3_conv'))
  x = layer_batch_normalization(x, 
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_3_bn'))
  
  x = layer_add(list(shortcut, x), name=paste0(name, '_add'))
  x = layer_activation(x, 'relu', name=paste0(name, '_out'))
  return(x)
}

stack1 = function(x, filters, blocks, stride1=2, name=NULL){
  # """A set of stacked residual blocks
  # Arguments:
  #   x: input tensor.
  #   filters: integer, filters of the bottleneck layer in a block.
  #   blocks: integer, blocks in the stacked blocks.
  #   stride1: default 2, stride of the first layer in the first block.
  #   name: string, stack label.
  # Returns:
  #   Output tensor for the stacked blocks.
  # """
  # x = block1(x, filters=64, stride=1, name=paste0("test", '_block1'))
  x = block1(x, filters, stride=stride1, name=paste0(name, '_block1'))
  for(i in 2:(blocks)){ ### triple check if indices are fine
    x = block1(x, filters, conv_shortcut=FALSE, name=paste0(name, '_block', as.character(i)))
    x
    # print(i)
  }
    
  return(x)
}

block2 = function(x, filters, kernel_size=3, stride=1, conv_shortcut=FALSE, name=NULL){
  # """A residual block
  # Arguments:
  #     x: input tensor.
  #     filters: integer, filters of the bottleneck layer.
  #     kernel_size: default 3, kernel size of the bottleneck layer.
  #     stride: default 1, stride of the first layer.
  #     conv_shortcut: default FALSE, use convolution shortcut if TRUE,
  #       otherwise identity shortcut.
  #     name: string, block label.
  # Returns:
  #   Output tensor for the residual block.
  # """
  # x = layer_input(shape = c(512,512,3))
  # filters = 64
  # kernel_size=3
  # stride=2
  # conv_shortcut=T
  # type_D=F
  # name="block1"
  
  bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  
  preact = layer_batch_normalization(x,
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_preact_bn'))
  preact = layer_activation(preact, 'relu', name=paste0(name, '_preact_relu'))
  
  if(conv_shortcut == TRUE){
    shortcut = layer_conv_2d(preact,
      4 * filters, 1, strides=stride, name=paste0(name, '_0_conv'))
  } else {
    if(stride > 1){
      shortcut = layer_max_pooling_2d(x, 1, strides=stride)  
    } else {
      shortcut = x
    }
  }
  
  x = layer_conv_2d(preact,
    filters, 1, strides=1, use_bias=FALSE, name=paste0(name, '_1_conv'))
  x = layer_batch_normalization(x, 
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_1_bn'))
  x = layer_activation(x, 'relu', name=paste0(name, '_1_relu'))
  
  x = layer_zero_padding_2d(x, padding=c(1, 1), name=paste0(name, '_2_pad'))
  x = layer_conv_2d(x, 
    filters,
    kernel_size,
    strides=stride,
    use_bias=FALSE,
    name=paste0(name, '_2_conv'))
  x = layer_batch_normalization(x, 
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_2_bn'))
  x = layer_activation(x, 'relu', name=paste0(name, '_2_relu'))
  
  x = layer_conv_2d(x, 4 * filters, 1, name=paste0(name, '_3_conv'))
  x = layer_add(list(shortcut, x), name=paste0(name, '_out'))
  return(x)
}

stack2 = function(x, filters, blocks, stride1=2, name=NULL){
  # """A set of stacked residual blocks.
  # Arguments:
  #     x: input tensor.
  #     filters: integer, filters of the bottleneck layer in a block.
  #     blocks: integer, blocks in the stacked blocks.
  #     stride1: default 2, stride of the first layer in the first block.
  #     name: string, stack label.
  # Returns:
  #     Output tensor for the stacked blocks.
  # """
  
  # x = layer_input(c(512, 512, 3))
  # filters = 64
  # blocks = 3
  # stride1 = 2
  
  x = block2(x, filters, conv_shortcut=TRUE, name=paste0(name, '_block1'))
  for(i in 2:(blocks-1)){
    x = block2(x, filters, name=paste0(name, '_block', as.character(i)))
  }
  x = block2(x, filters, stride=stride1, name=paste0(name, '_block', as.character(blocks)))
  return(x)
}

block3 = function(x,
           filters,
           kernel_size=3,
           stride=1,
           groups=32,
           conv_shortcut=TRUE,
           name=NULL){
  # """A residual block.
  # Arguments:
  #   x: input tensor.
  #   filters: integer, filters of the bottleneck layer.
  #   kernel_size: default 3, kernel size of the bottleneck layer.
  #   stride: default 1, stride of the first layer.
  #   groups: default 32, group size for grouped convolution.
  #   conv_shortcut: default TRUE, use convolution shortcut if TRUE,
  #       otherwise identity shortcut.
  #   name: string, block label.
  # Returns:
  #   Output tensor for the residual block.
  # """
  
  ##for testing
  # filters = 256
  # groups = 32
  # conv_shortcut = T
  # kernel_size=3
  # stride=1
  # name = "test"
  # x = layer_input(shape = c(512, 512, 3))
  
  bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  
  if(conv_shortcut == TRUE){
    shortcut = layer_conv_2d(x,
                             floor(64 / groups) * filters,
                             1,
                             strides=stride,
                             use_bias=FALSE,
                             name=paste0(name, '_0_conv'))
    shortcut = layer_batch_normalization(shortcut,
                                         axis=bn_axis,
                                         epsilon=1.001e-5,
                                         name=paste0(name, '_0_bn'))
  } else {
    shortcut = x    
  }

  x = layer_conv_2d(x, filters, 1, use_bias=FALSE, name=paste0(name, '_1_conv'))
  x = layer_batch_normalization(x, 
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_1_bn'))
  x = layer_activation(x, 'relu', name=paste0(name, '_1_relu'))
  
  c = floor(filters / groups)
  x = layer_zero_padding_2d(x, padding=c(1, 1), name=paste0(name, '_2_pad'))
  x = layer_depthwise_conv_2d(x, 
                              kernel_size,
                              strides=stride,
                              depth_multiplier=c,
                              use_bias=FALSE,
                              name=paste0(name, '_2_conv'))
  
  kernel = array(0, dim = c(1, 1, filters * c, filters))
  for(i in 1:filters){
    start = floor((i-1) / c) * c * c + (i-1) %% c + 1
    end = start + c * c
    kernel[, , seq.int(start, end-1, c), i] = 1
  }
  CONV_KERNEL_INITIALIZER = list(
    'class_name' = 'Constant',
    'config' = list(
      'value' = kernel
    )
  )
  x = layer_conv_2d(x,
                    filters,
                    1,
                    use_bias=FALSE,
                    trainable=FALSE,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=paste0(name, '_2_gconv'))  
  
  x = layer_batch_normalization(x,
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_2_bn'))
  x = layer_activation(x, 'relu', name=paste0(name, '_2_relu'))
  
  x = layer_conv_2d(x, 
    floor(64 / groups) * filters, 1, use_bias=FALSE, name=paste0(name, '_3_conv'))
  x = layer_batch_normalization(x, 
    axis=bn_axis, epsilon=1.001e-5, name=paste0(name, '_3_bn'))
  
  x = layer_add(list(shortcut, x), name=paste0(name, '_add'))
  x = layer_activation(x, 'relu', name=paste0(name, '_out'))
  return(x)
}

stack3 = function(x, filters, blocks, stride1=2, groups=32, name=NULL){
  # """A set of stacked residual blocks.
  # Arguments:
  #   x: input tensor.
  #   filters: integer, filters of the bottleneck layer in a block.
  #   blocks: integer, blocks in the stacked blocks.
  #   stride1: default 2, stride of the first layer in the first block.
  #   groups: default 32, group size for grouped convolution.
  #   name: string, stack label.
  # Returns:
  #   Output tensor for the stacked blocks.
  # """
  x = block3(x, filters, stride=stride1, groups=groups, name=paste0(name, '_block1'))
  for(i in 2:(blocks)){
    x = block3(
      x,
      filters,
      groups=groups,
      conv_shortcut=FALSE,
      name=paste0(name, '_block', as.character(i)))
    x
  }
  return(x)
    
}

######################################################################################
###### ResNet v1
######################################################################################

# add @ before keras_export
# keras_export('keras.applications.resnet50.ResNet50',
#               'keras.applications.resnet.ResNet50',
#               'keras.applications.ResNet50')
ResNet50 = function(include_top=TRUE,
             weights='imagenet',
             input_tensor=NULL,
             input_shape=NULL,
             pooling=NULL,
             classes=1000#,
             # **kwargs
             ){
  # """Instantiates the ResNet50 architecture"""
  
  stack_fn = function(x){
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    return(stack1(x, 512, 3, name='conv5'))
  }
    
  
  return(ResNet(stack_fn, FALSE, TRUE, 'resnet50', include_top, weights,
                input_tensor, input_shape, pooling, classes#,
                # **kwargs
                ))
  
}
  
# add @ before keras_export
# keras_export('keras.applications.resnet.ResNet101',
#               'keras.applications.ResNet101')
ResNet101 = function(include_top=TRUE,
              weights='imagenet',
              input_tensor=NULL,
              input_shape=NULL,
              pooling=NULL,
              classes=1000#,
              # **kwargs
              ){
  # """Instantiates the ResNet101 architecture."""
  
  stack_fn = function(x){
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 23, name='conv4')
    x = stack1(x, 512, 3, name='conv5')
    return(x)
  }
  
  return(ResNet(stack_fn, FALSE, TRUE, 'resnet101', include_top, weights,
                input_tensor, input_shape, pooling, classes#, 
                # **kwargs
                ))
}
  

# add @ before keras_export
# keras_export('keras.applications.resnet.ResNet152',
#              'keras.applications.ResNet152')
ResNet152 = function(include_top=TRUE,
              weights='imagenet',
              input_tensor=NULL,
              input_shape=NULL,
              pooling=NULL,
              classes=1000#,
              # **kwargs
              ){
  
  # """Instantiates the ResNet152 architecture."""
  
  stack_fn = function(x){
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 8, name='conv3')
    x = stack1(x, 256, 36, name='conv4')
    return(stack1(x, 512, 3, name='conv5'))
  }
  
  return(ResNet(stack_fn, FALSE, TRUE, 'resnet152', include_top, weights,
                input_tensor, input_shape, pooling, classes#,
                # **kwargs
                ))
  
}

# RN152 = ResNet152(include_top = F,
#                   input_shape = c(256, 256, 3))
# sink('~/Summary_ResNet152.txt', append=TRUE)
# summary(RN152)
# sink()
# 
# 
# RN101 = ResNet101(include_top = F,
#                   input_shape = c(256, 256, 3))
# sink('~/Summary_ResNet101.txt', append=TRUE)
# summary(RN101)
# sink()
# 
# RN50 = ResNet50(include_top = F,
#                   input_shape = c(512, 512, 3))
# RN50top = ResNet50(include_top = T,
#                 input_shape = c(512, 512, 3))
# RN50v2 = ResNet50V2(include_top = T,
#                 input_shape = c(512, 512, 3))
# sink('~/Summary_ResNet50.txt', append=TRUE)
# summary(RN50)
# sink()

# add @ before keras_export
# keras_export('keras.applications.resnet50.preprocess_input',
#               'keras.applications.resnet.preprocess_input')
preprocess_input = function(x, data_format=NULL){
  return(imagenet_utils.preprocess_input(
    x, data_format=data_format, mode='caffe'))
}
  

# add @ before keras_export
# keras_export('keras.applications.resnet50.decode_predictions',
#               'keras.applications.resnet.decode_predictions')
decode_predictions = function(preds, top=5){
  return(
    imagenet_utils.decode_predictions(preds, top=top)
    )
}
  


# preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
#   mode='',
#   ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE,
#   error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC)
# decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

# DOC = """
#   Reference paper:
#   - [Deep Residual Learning for Image Recognition]
#   (https://arxiv.org/abs/1512.03385) (CVPR 2015)
#   Optionally loads weights pre-trained on ImageNet.
#   Note that the data format convention used by the model is
#   the one specified in your Keras config at `~/.keras/keras.json`.
#   Arguments:
#     include_top: whether to include the fully-connected
#       layer at the top of the network.
#     weights: one of `NULL` (random initialization),
#       'imagenet' (pre-training on ImageNet),
#       or the path to the weights file to be loaded.
#     input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#       to use as image input for the model.
#     input_shape: optional shape tuple, only to be specified
#       if `include_top` is FALSE (otherwise the input shape
#       has to be `(224, 224, 3)` (with `'channels_last'` data format)
#       or `(3, 224, 224)` (with `'channels_first'` data format).
#       It should have exactly 3 inputs channels,
#       and width and height should be no smaller than 32.
#       E.g. `(200, 200, 3)` would be one valid value.
#     pooling: Optional pooling mode for feature extraction
#       when `include_top` is `FALSE`.
#       - `NULL` means that the output of the model will be
#           the 4D tensor output of the
#           last convolutional block.
#       - `avg` means that global average pooling
#           will be applied to the output of the
#           last convolutional block, and thus
#           the output of the model will be a 2D tensor.
#       - `max` means that global max pooling will
#           be applied.
#     classes: optional number of classes to classify images
#       into, only to be specified if `include_top` is TRUE, and
#       if no `weights` argument is specified.
#   Returns:
#     A Keras model instance.
# """
# 
# setattr(ResNet50, '__doc__', ResNet50.__doc__ + DOC)
# setattr(ResNet101, '__doc__', ResNet101.__doc__ + DOC)
# setattr(ResNet152, '__doc__', ResNet152.__doc__ + DOC)

# """ResNet v2 models for Keras.
# Reference:
#   - [Identity Mappings in Deep Residual Networks]
#     (https://arxiv.org/abs/1603.05027) (CVPR 2016)
# """
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# 
# from tensorflow.python.keras.applications import imagenet_utils
# from tensorflow.python.keras.applications import resnet
# from tensorflow.python.util.tf_export import keras_export

# add @ before keras_export
# keras_export('keras.applications.resnet_v2.ResNet50V2',
#               'keras.applications.ResNet50V2')

######################################################################################
###### ResNet v2
######################################################################################

ResNet50V2 = function(
  include_top=TRUE,
  weights='imagenet',
  input_tensor=NULL,
  input_shape=NULL,
  pooling=NULL,
  classes=1000,
  classifier_activation='softmax'){
  
  # """Instantiates the ResNet50V2 architecture."""
  
  stack_fn = function(x){
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 4, name='conv3')
    x = stack2(x, 256, 6, name='conv4')
    return(stack2(x, 512, 3, stride1=1, name='conv5'))
  }
  
  return(
    ResNet(stack_fn,
           TRUE,
           TRUE,
           'resnet50v2',
           include_top,
           weights,
           input_tensor,
           input_shape,
           pooling,
           classes,
           classifier_activation=classifier_activation))
}
# RN50v2 = ResNet50V2(include_top = T,
#                       input_shape = c(512, 512, 3))
# sink("~/Desktop/application_resnet50_v2.txt")
# summary(RN50v2)
# sink()

# add @ before keras_export
# keras_export('keras.applications.resnet_v2.ResNet101V2',
#               'keras.applications.ResNet101V2')
ResNet101V2 = function(
  include_top=TRUE,
  weights='imagenet',
  input_tensor=NULL,
  input_shape=NULL,
  pooling=NULL,
  classes=1000,
  classifier_activation='softmax'){
  
  # """Instantiates the ResNet101V2 architecture."""
  
  x = layer_input(c(512,512,3))
  x = stack2(x, 64, 3, name='conv2')
  x = stack2(x, 128, 4, name='conv3')
  x = stack2(x, 256, 23, name='conv4')
  x = stack2(x, 512, 3, stride1=1, name='conv5')
  stack_fn = function(x){
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 4, name='conv3')
    x = stack2(x, 256, 23, name='conv4')
    return(stack2(x, 512, 3, stride1=1, name='conv5'))
  } 
  
  return(
    ResNet(stack_fn,
           TRUE,
           TRUE,
           'resnet101v2',
           include_top,
           weights,
           input_tensor,
           input_shape,
           pooling,
           classes,
           classifier_activation=classifier_activation))
  
}
# RN101v2 = ResNet101V2(include_top = T,
#                       input_shape = c(512, 512, 3))
# sink("~/Desktop/application_resnet101_v2.txt")
# summary(RN101v2)
# sink()


# add @ before keras_export
# keras_export('keras.applications.resnet_v2.ResNet152V2',
#               'keras.applications.ResNet152V2')
ResNet152V2 = function(
  include_top=TRUE,
  weights='imagenet',
  input_tensor=NULL,
  input_shape=NULL,
  pooling=NULL,
  classes=1000,
  classifier_activation='softmax'){
  
  # """Instantiates the ResNet152V2 architecture."""
  
  stack_fn = function(x){
    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 8, name='conv3')
    x = stack2(x, 256, 36, name='conv4')
    return(stack2(x, 512, 3, stride1=1, name='conv5'))
  }
    
  
  return(
    ResNet(stack_fn,
           TRUE,
           TRUE,
           'resnet152v2',
           include_top,
           weights,
           input_tensor,
           input_shape,
           pooling,
           classes,
           classifier_activation=classifier_activation))
  
}
# RN152v2 = ResNet152V2(include_top = T,
#                       input_shape = c(512, 512, 3))
# sink("~/Desktop/application_resnet152_v2.txt")
# summary(RN152v2)
# sink()

######################################################################################
###### ResNeXt
######################################################################################

ResNeXt50 = function(include_top=TRUE,
                     weights='imagenet',
                     input_tensor=NULL,
                     input_shape=NULL,
                     pooling=NULL,
                     classes=1000,
                     classifier_activation='softmax'
                     # **kwargs
                     ){
  
  # """Instantiates the ResNeXt50 architecture."""
  
  stack_fn = function(x){
    x = stack3(x, 128, 3, stride1=1, name='conv2')
    x = stack3(x, 256, 4, name='conv3')
    x = stack3(x, 512, 6, name='conv4')
    return(stack3(x, 1024, 3, name='conv5'))
  }
  
  return(
    ResNet(stack_fn,
           FALSE,
           FALSE,
           'resnext50',
           include_top,
           weights,
           input_tensor,
           input_shape,
           pooling,
           classes,
           classifier_activation='softmax'
           # **kwargs
    )
  ) 
}
# RNX50 = ResNeXt50(include_top = T,
#                     input_shape = c(512, 512, 3))
# summary(RNX50)


ResNeXt101 = function(include_top=TRUE,
                      weights='imagenet',
                      input_tensor=NULL,
                      input_shape=NULL,
                      pooling=NULL,
                      classes=1000,
                      classifier_activation='softmax'
                      # **kwargs
                      ){
  
  # """Instantiates the ResNeXt101 architecture."""
  
  stack_fn = function(x){
    x = stack3(x, 128, 3, stride1=1, name='conv2')
    x = stack3(x, 256, 4, name='conv3')
    x = stack3(x, 512, 23, name='conv4')
    return(stack3(x, 1024, 3, name='conv5'))
  }
  
  return(
    ResNet(stack_fn,
           FALSE,
           FALSE,
           'resnext101',
           include_top,
           weights,
           input_tensor,
           input_shape,
           pooling, 
           classes,
           classifier_activation=classifier_activation
           # **kwargs
    )
  ) 
}

# RNX101 = ResNeXt101(include_top = T,
#                       input_shape = c(512, 512, 3))
# summary(RNX101)




# # add @ before keras_export
# # keras_export('keras.applications.resnet_v2.preprocess_input')
# def preprocess_input(x, data_format=NULL):
#   return imagenet_utils.preprocess_input(
#     x, data_format=data_format, mode='tf')
# 
# # add @ before keras_export
# # keras_export('keras.applications.resnet_v2.decode_predictions')
# def decode_predictions(preds, top=5):
#   return imagenet_utils.decode_predictions(preds, top=top)
# 
# 
# preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
#   mode='',
#   ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
#   error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC)
# decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

# DOC = """
#   Reference:
#   - [Identity Mappings in Deep Residual Networks]
#     (https://arxiv.org/abs/1603.05027) (CVPR 2016)
#   Optionally loads weights pre-trained on ImageNet.
#   Note that the data format convention used by the model is
#   the one specified in your Keras config at `~/.keras/keras.json`.
#   Note: each Keras Application expects a specific kind of input preprocessing.
#   For ResNetV2, call `tf.keras.applications.resnet_v2.preprocess_input` on your
#   inputs before passing them to the model.
#   Arguments:
#     include_top: whether to include the fully-connected
#       layer at the top of the network.
#     weights: one of `NULL` (random initialization),
#       'imagenet' (pre-training on ImageNet),
#       or the path to the weights file to be loaded.
#     input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#       to use as image input for the model.
#     input_shape: optional shape tuple, only to be specified
#       if `include_top` is False (otherwise the input shape
#       has to be `(224, 224, 3)` (with `'channels_last'` data format)
#       or `(3, 224, 224)` (with `'channels_first'` data format).
#       It should have exactly 3 inputs channels,
#       and width and height should be no smaller than 32.
#       E.g. `(200, 200, 3)` would be one valid value.
#     pooling: Optional pooling mode for feature extraction
#       when `include_top` is `False`.
#       - `NULL` means that the output of the model will be
#           the 4D tensor output of the
#           last convolutional block.
#       - `avg` means that global average pooling
#           will be applied to the output of the
#           last convolutional block, and thus
#           the output of the model will be a 2D tensor.
#       - `max` means that global max pooling will
#           be applied.
#     classes: optional number of classes to classify images
#       into, only to be specified if `include_top` is TRUE, and
#       if no `weights` argument is specified.
#     classifier_activation: A `str` or callable. The activation function to use
#       on the "top" layer. Ignored unless `include_top=TRUE`. Set
#       `classifier_activation=NULL` to return the logits of the "top" layer.
#   Returns:
#     A `keras.Model` instance.
# """
# 
# setattr(ResNet50V2, '__doc__', ResNet50V2.__doc__ + DOC)
# setattr(ResNet101V2, '__doc__', ResNet101V2.__doc__ + DOC)
# setattr(ResNet152V2, '__doc__', ResNet152V2.__doc__ + DOC)