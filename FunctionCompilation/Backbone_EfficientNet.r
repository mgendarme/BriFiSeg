### ref website 
## https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py

require(keras)
require(tidyverse)
require(tensorflow)

backend = NULL
layers = NULL
models = NULL
keras_utils = NULL

BASE_WEIGHTS_PATH = list(
  'https://github.com/Callidior/keras-applications/releases/download/efficientnet/')

WEIGHTS_HASHES = list(
  'b0'= c('e9e877068bd0af75e0a36691e03c072c',
         '345255ed8048c2f22c793070a9c1a130'),
  'b1'= c('8f83b9aecab222a9a2480219843049a1',
         'b20160ab7b79b7a92897fcb33d52cc61'),
  'b2'= c('b6185fdcd190285d516936c09dceeaa4',
         'c6e46333e8cddfa702f4d8b8b6340d70'),
  'b3'= c('b2db0f8aac7c553657abb2cb46dcbfbb',
         'e0cf8654fad9d3625190e30d70d0c17d'),
  'b4'= c('ab314d28135fe552e2f9312b31da6926',
         'b46702e4754d2022d62897e0618edc7b'),
  'b5'= c('8d60b903aff50b09c6acf8eaba098e09',
         '0a839ac36e46552a881f2975aaab442f'),
  'b6'= c('a967457886eac4f5ab44139bdd827920',
         '375a35c17ef70d46f9c664b03b4437f2'),
  'b7'= c('e964fd6e26e9a4c144bcb811f2a10f20',
         'd55674cc46b805f4382d18bc08ed43c1')
)


DEFAULT_BLOCKS_ARGS = list(
  ##0
  list('kernel_size'= 3L,
       'repeats'= 1L,
       'filters_in'= 32, 
       'filters_out'= 16,
       'expand_ratio'= 1,
       'id_skip'= TRUE,
       'strides'= 1L,
       'se_ratio'= 0.25),
  ##1
  list('kernel_size'= 3L,
       'repeats'= 2L,
       'filters_in'= 16,
       'filters_out'= 24,
       'expand_ratio'= 6,
       'id_skip'= TRUE,
       'strides'= 2L,
       'se_ratio'= 0.25),
  ##2
  list('kernel_size'= 5L,
       'repeats'= 2L,
       'filters_in'= 24,
       'filters_out'= 40,
       'expand_ratio'= 6, 
       'id_skip'= TRUE,
       'strides'= 2L,
       'se_ratio'= 0.25),
  ##3
  list('kernel_size'= 3L, 
       'repeats'= 3L,
       'filters_in'= 40, 
       'filters_out'= 80,
       'expand_ratio'= 6,
       'id_skip'= TRUE, 
       'strides'= 2L, 
       'se_ratio'= 0.25),
  ##4
  list('kernel_size'= 5L, 
       'repeats'= 3L,
       'filters_in'= 80, 
       'filters_out'= 112,
       'expand_ratio'= 6,
       'id_skip'= TRUE, 
       'strides'= 1L, 
       'se_ratio'= 0.25),
  ##5
  list('kernel_size'= 5L,
       'repeats'= 4L,
       'filters_in'= 112,
       'filters_out'= 192,
       'expand_ratio'= 6,
       'id_skip'= TRUE,
       'strides'= 2L, 
       'se_ratio'= 0.25),
  ##6
  list('kernel_size'= 3L,
       'repeats'= 1L,
       'filters_in'= 192, 
       'filters_out'= 320,
       'expand_ratio'= 6,
       'id_skip'= TRUE, 
       'strides'= 1L,
       'se_ratio'= 0.25)
)

CONV_KERNEL_INITIALIZER = list(
  'class_name' = 'VarianceScaling',
  'config' = list(
    'scale' = 2.0,
    'mode' = 'fan_out',
    # EfficientNet actually uses an untruncated normal distribution for
    # initializing conv layers, but keras.initializers.VarianceScaling use
    # a truncated distribution.
    # We decided against a custom initializer for better serializability.
    'distribution' = 'normal'
  )
)

DENSE_KERNEL_INITIALIZER = list(
  'class_name' = 'VarianceScaling',
  'config' = list(
    'scale' = 1. / 3.,
    'mode' = 'fan_out',
    'distribution' = 'uniform'
  )
)

## use of swish-1 recommended instead of relu as it outperforms relu in almost any application
swish = function(x){
  # """Swish activation function.
  #   # Arguments
  #       x: Input tensor.
  #   # Returns
  #       The Swish activation: `x * sigmoid(x)`.
  #   # References
  #       [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
  #   """

  # if(){               #### find a rule if tf module not available
    x = tf$nn$swish(x)
  # } else {            #### find a rule if tf module not available   
  #   x = x * activation_sigmoid(x)
  # }
  
  return(x)

}
  

correct_pad = function(inputs, kernel_size){
  # """Returns a tuple for zero-padding for 2D convolution with downsampling.
  #   # Arguments
  #       input_size: An integer or tuple/list of 2 integers.
  #       kernel_size: An integer or tuple/list of 2 integers.
  #   # Returns
  #       A tuple.
  #   """
  
  # for testing
  # backend = k_backend()
  # inputs = x
  # inputs = layer_input(shape = c(NULL, 128, 128, 96))
  # kernel_size = 3L
  
  img_dim = ifelse(backend()$image_data_format() == "channels_first", 2, 1)
  # img_dim
  input_size = unlist(k_int_shape(inputs)[img_dim:(img_dim+2)])
  # input_size
  
  if(is.integer(kernel_size) & length(kernel_size) == 1){
    kernel_size = c(kernel_size, kernel_size)
  }
  
  if(is.null(input_size[1])){
    adjust = c(1, 1)
  } else {
    adjust = c(1 - input_size[1] %% 2, 1 - input_size[2] %% 2)
  }
  # adjust
  correct = c(floor(kernel_size[1] / 2), floor(kernel_size[2] / 2))
  # correct
  zero_pad = list(
    list(correct[1] - adjust[1], correct[1]),
    list(correct[2] - adjust[2], correct[2])
  )
  zero_pad
  # return(zero_pad)  
  
}


block = function(inputs,
                 activation_fn,#=swish
                 drop_rate=0.0,
                 name='',
                 filters_in=32,
                 filters_out=16,
                 kernel_size=3L,
                 strides=1L,
                 expand_ratio=1,
                 se_ratio=0.0,
                 id_skip=TRUE){
  # """A mobile inverted residual block.
  #   # Arguments
  #       inputs: input tensor.
  #       activation_fn: activation function.
  #       drop_rate: float between 0 and 1, fraction of the input units to drop.
  #       name: string, block label.
  #       filters_in: integer, the number of input filters.
  #       filters_out: integer, the number of output filters.
  #       kernel_size: integer, the dimension of the convolution window.
  #       strides: integer, the stride of the convolution.
  #       expand_ratio: integer, scaling coefficient for the input filters.
  #       se_ratio: float between 0 and 1, fraction to squeeze the input filters.
  #       id_skip: boolean.
  #   # Returns
  #       output tensor for the block.
  #   """
  bn_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
  
  # for testing
  # kernel_size = 3L
  # filters_in = 16
  # filters_out = 24
  # expand_ratio = 6
  # id_skip = TRUE
  # strides = 2
  # se_ratio = 0.25
  # activation_fn = "sigmoid"
  # drop_rate = 0.025
  # name = "block2b_"
  # inputs = x
  # 
  # x_backup = x
  
  # Expansion phase
  filters = filters_in * expand_ratio
  if(expand_ratio != 1){
    x = layer_conv_2d(inputs,
                      filters,
                      1,
                      padding='same',
                      use_bias=FALSE,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=paste0(name,  'expand_conv'))
    x = layer_batch_normalization(x, axis=bn_axis, name=paste0(name,  'expand_bn'))
    x = layer_activation(x, activation_fn, name=paste0(name,  'expand_activation'))
  } else {
    x = inputs
  }
  
  # Depthwise Convolution
  if(strides == 2){
    x = layer_zero_padding_2d(x, 
                              padding=correct_pad(x, kernel_size),
                              name=paste0(name,  'dwconv_pad'))
    conv_pad = 'valid'
  } else {
    conv_pad = 'same'
  }
    
  x = layer_depthwise_conv_2d(x,
                              kernel_size,
                              strides=strides,
                              padding=conv_pad,
                              use_bias=FALSE,
                              depthwise_initializer=CONV_KERNEL_INITIALIZER,
                              name=paste0(name, 'dwconv'))
  x = layer_batch_normalization(x, axis=bn_axis, name=paste0(name,'bn'))
  x = layer_activation(x, activation_fn, name=paste0(name,'activation'))
  
  # Squeeze and Excitation phase
  if(0 < se_ratio & se_ratio <= 1){
    filters_se = max(1, as.integer(filters_in * se_ratio))
    se = layer_global_average_pooling_2d(x, name=paste0(name, 'se_squeeze'))
    if(bn_axis == 1){
      se = layer_reshape(se, list(filters, 1, 1), name=paste0(name, 'se_reshape'))
    } else {
      se = layer_reshape(se, list(1, 1, filters), name=paste0(name, 'se_reshape'))  
    }
  
    se = layer_conv_2d(se, 
                       filters_se, 1,
                       padding='same',
                       activation=activation_fn,
                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                       name=paste0(name,  'se_reduce'))
    se = layer_conv_2d(se,
                       filters, 1,
                       padding='same',
                       activation='sigmoid',
                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                       name=paste0(name,  'se_expand'))
    if(k_backend() == 'theano'){
      # For the Theano backend, we have to explicitly make
      # the excitation weights broadcastable.
      # se = layer_lambda(
      #   lambda x: backend.pattern_broadcast(x, [TRUE, TRUE, TRUE, FALSE]),
      #   output_shape=lambda input_shape: input_shape,
      #   name=paste0(name,  'se_broadcast')(se)
    }
      
    x = layer_multiply(list(x, se), name=paste0(name,  'se_excite'))
  }
    
  # Output phase
  x = layer_conv_2d(x,
                    filters_out,
                    1,
                    padding='same',
                    use_bias=FALSE,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=paste0(name,  'project_conv'))
  x = layer_batch_normalization(x, axis=bn_axis, name=paste0(name,  'project_bn'))
  if(id_skip == TRUE & strides == 1 & filters_in == filters_out){
    if(drop_rate > 0){
      x = layer_dropout(x,
                        drop_rate,
                        noise_shape=c(NULL, 1, 1, 1),
                        name=paste0(name,  'drop'))
    }
    x = layer_add(list(x, inputs), name=paste0(name,  'add'))
  }
  
  return(x)
  
}
  
EfficientNet = function(width_coefficient,
                        depth_coefficient,
                        default_size,
                        dropout_rate=0.2,
                        drop_connect_rate=0.2,
                        depth_divisor=8,
                        activation_fn="sigmoid",
                        blocks_args=DEFAULT_BLOCKS_ARGS,
                        model_name='efficientnet',
                        include_top=TRUE,
                        weights='imagenet',
                        input_tensor=NULL,
                        input_shape=NULL,
                        pooling=NULL,
                        classes=1000#,
                        # **kwargs
                        ){
  # """Instantiates the EfficientNet architecture using given scaling coefficients.
  #   Optionally loads weights pre-trained on ImageNet.
  #   Note that the data format convention used by the model is
  #   the one specified in your Keras config at `~/.keras/keras.json`.
  #   # Arguments
  #       width_coefficient: float, scaling coefficient for network width.
  #       depth_coefficient: float, scaling coefficient for network depth.
  #       default_size: integer, default input image size.
  #       dropout_rate: float, dropout rate before final classifier layer.
  #       drop_connect_rate: float, dropout rate at skip connections.
  #       depth_divisor: integer, a unit of network width.
  #       activation_fn: activation function.
  #       blocks_args: list of dicts, parameters to construct block modules.
  #       model_name: string, model name.
  #       include_top: whether to include the fully-connected
  #           layer at the top of the network.
  #       weights: one of `NULL` (random initialization),
  #             'imagenet' (pre-training on ImageNet),
  #             or the path to the weights file to be loaded.
  #       input_tensor: optional Keras tensor
  #           (i.e. output of `layers.Input()`)
  #           to use as image input for the model.
  #       input_shape: optional shape tuple, only to be specified
  #           if `include_top` is FALSE.
  #           It should have exactly 3 inputs channels.
  #       pooling: optional pooling mode for feature extraction
  #           when `include_top` is `FALSE`.
  #           - `NULL` means that the output of the model will be
  #               the 4D tensor output of the
  #               last convolutional layer.
  #           - `avg` means that global average pooling
  #               will be applied to the output of the
  #               last convolutional layer, and thus
  #               the output of the model will be a 2D tensor.
  #           - `max` means that global max pooling will
  #               be applied.
  #       classes: optional number of classes to classify images
  #           into, only to be specified if `include_top` is TRUE, and
  #           if no `weights` argument is specified.
  #   # Returns
  #       A Keras model instance.
  #   # Raises
  #       ValueError: in case of invalid argument for `weights`,
  #           or invalid input shape.
  #   """
  # global backend, layers, models, keras_utils
  # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
  
  # if not (weights in {'imagenet', NULL} or os.path.exists(weights)):
  #   raise ValueError('The `weights` argument should be either '
  #                    '`NULL` (random initialization), `imagenet` '
  #                    '(pre-training on ImageNet), '
  #                    'or the path to the weights file to be loaded.')
  # 
  # if weights == 'imagenet' and include_top and classes != 1000:
  #   raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
  #                    ' as TRUE, `classes` should be 1000')
  
  # Determine proper input shape
  # input_shape = _obtain_input_shape(input_shape,
  #                                   default_size=default_size,
  #                                   min_size=32,
  #                                   data_format=backend.image_data_format(),
  #                                   require_flatten=include_top,
  #                                   weights=weights)
  
  ## for testing
  # width_coefficient = 1.0
  # depth_coefficient = 1.0
  # default_size = 224
  # dropout_rate = 0.2
  # drop_connect_rate = 0.2
  # depth_divisor = 8
  # activation_fn = "sigmoid"
  # blocks_args = DEFAULT_BLOCKS_ARGS
  # model_name = 'efficientnet-b0'
  # include_top = TRUE
  # weights = 'imagenet'
  # input_tensor = NULL
  # input_shape = c(256, 256, 3)
  # pooling = NULL
  # classes = 1000
  # kernel_size = 3L
  
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
  
  round_filters = function(filters, divisor=depth_divisor){
    # """Round number of filters based on depth multiplier."""
    # filters = 32
    filters = filters*width_coefficient
    # divisor = depth_divisor
    new_filters = max(divisor, floor(floor(filters + divisor / 2) / divisor) * divisor)
    # Make sure that round down does not go down by more than 10%.
    if(new_filters < 0.9 * filters) {
      new_filters = new_filters + divisor
    }
    return(new_filters)
  }
  
  round_repeats = function(repeats){
    # """Round number of repeats based on depth multiplier."""
    return(as.integer(ceiling(depth_coefficient * repeats)))
  }
  
 
  # Build stem
  x = img_input
  x = layer_zero_padding_2d(x,
                            padding = correct_pad(x, 3L),
                            name='stem_conv_pad')
  x = layer_conv_2d(x,
                    round_filters(32),
                    3,
                    strides=2,
                    padding='valid',
                    use_bias=FALSE,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name='stem_conv')
  x = layer_batch_normalization(x, axis=bn_axis, name='stem_bn')
  x = layer_activation(x, activation_fn, name='stem_activation')
  
  # Build blocks
  # from copy import deepcopy
  # blocks_args = deepcopy(blocks_args)
  blocks_args = DEFAULT_BLOCKS_ARGS
  
  b <<- 0
  blocks = blocks_args[[1]]$repeats
  for(ba in 2:length(blocks_args)){
    blocks = blocks + blocks_args[[ba]]$repeats
  }
  
  # float(sum(args['repeats'] for args in blocks_args))
  for(i in 1:length(blocks_args)){
    # assert args['repeats'] > 0
    # print(paste0("i_",i))
    # for testing
    # depth_divisor=8
    # width_coefficient = 1.1
    # depth_coefficient = 1.2
    temp_args = blocks_args[[i]]
    # Update block input and output filters based on depth multiplier.
    temp_args$filters_in = round_filters(blocks_args[[i]]$filters_in)
    # blocks_args[[i]]$filters_in
    temp_args$filters_out = round_filters(blocks_args[[i]]$filters_out)
    
    ###############
    ###############
    for(j in 1:(round_repeats(temp_args$repeats))){
      # print(paste0("j___", j))
      # The first block needs to take care of stride and filter size increase.
      if(j > 1){
        temp_args$strides = 1
        temp_args$filters_in = temp_args$filters_out
      }
      
      # print(str(temp_args))
      
      args = c(
        temp_args[names(temp_args) != "repeats"],
        list(inputs = x,
             activation_fn = "sigmoid",
             drop_rate = drop_connect_rate * b / blocks,
             name = paste0('block', i, letters[j], "_")
             )
      )
      x = do.call(block, args)
      b <<- b+1
    }
  }
  
  # Build top
  x = layer_conv_2d(x,
                    round_filters(1280), 1,
                    padding='same',
                    use_bias=FALSE,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name='top_conv')
  x = layer_batch_normalization(x, axis=bn_axis, name='top_bn')
  x = layer_activation(x, activation_fn, name='top_activation')
  
  if(include_top==TRUE){
    x = layer_global_average_pooling_2d(x, name='avg_pool')
    if(dropout_rate > 0){
      x = layer_dropout(x, dropout_rate, name='top_dropout')
    }
    x = layer_dense(x,
                    classes,
                    activation='softmax',
                    kernel_initializer=DENSE_KERNEL_INITIALIZER,
                    name='probs')
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
  if(!is.null(input_tensor)){
    inputs = keras_utils.get_source_inputs(input_tensor)
  } else {
    inputs = img_input
  }
    
  
  # Create model.
  model = keras_model(inputs, x)#, name=model_name)
  
  # Load weights.
  if(!is.null(weights)){
    if(weights == 'imagenet'){
      if(include_top == TRUE){
        file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
        file_hash = WEIGHTS_HASHES[str_remove(model_name, "efficientnet-")][[1]][1]
      } else {
        file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = WEIGHTS_HASHES[str_remove(model_name, "efficientnet-")][[1]][2]    
      }
      file_name = paste0(model_name, file_suff)
      weights_path = get_file(file_name,
                              paste0(BASE_WEIGHTS_PATH, file_name))
      model %>% load_model_weights_hdf5(filepath = weights_path)    
    } else {
      model %>% load_model_weights_hdf5(filepath = weights_path)
    }  
  }
  
  model
  
}

EfficientNetB0 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation=swish,
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(1.0, 1.0, 224, 0.2,
                      model_name='efficientnet-b0',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                 )
  )
}

# ENB0 = EfficientNetB0(include_top=TRUE,
#                       weights=NULL,#'imagenet',
#                       input_tensor=NULL,
#                       input_shape=c(256,256,3),
#                       pooling=NULL,
#                       classes=1000#,
#                       # **kwargs
#                       )
# summary(ENB0)

EfficientNetB1 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation="relu",
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(1.0, 1.1, 240, 0.2,
                      model_name='efficientnet-b1',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                 )
    )
}
# ENB1 = EfficientNetB1(include_top=FALSE,
#                       weights="imagenet",
#                       input_tensor=NULL,
#                       input_shape=c(256,256,3),
#                       pooling="max",
#                       classes=1000#,
#                       # **kwargs
#                       )
# sink("~/Desktop/application_efficientnet_b1.txt")
# summary(ENB1)
# sink()

EfficientNetB2 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation="relu",
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(1.1, 1.2, 260, 0.3,
                      model_name='efficientnet-b2',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                      )
  )
}
  
EfficientNetB3 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation="relu",
                   classes=1000#,
                   # **kwargs
                   ){
  return(EfficientNet(1.2, 1.4, 300, 0.3,
                      model_name='efficientnet-b3',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                      )
  )
}

EfficientNetB4 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation=swish,
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(1.4, 1.8, 380, 0.4,
                      model_name='efficientnet-b4',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                 )
    )
}

# effmodb4 = EfficientNetB4(include_top=TRUE,
#                    weights='imagenet',
#                    input_tensor=NULL,
#                    input_shape=c(512, 512, 3),
#                    pooling=NULL,
#                    activation=swish,
#                    classes=1000#,
#                    # **kwargs
#                    )

EfficientNetB5 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation="relu",
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(1.6, 2.2, 456, 0.4,
                      model_name='efficientnet-b5',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                 )
    )
}

EfficientNetB6 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation="relu",
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(1.8, 2.6, 528, 0.5,
                      model_name='efficientnet-b6',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes,
                      # **kwargs
                 )
  )
}
# ENB6 = EfficientNetB6(include_top=TRUE,
#                       weights='imagenet',
#                       input_tensor=NULL,
#                       input_shape=c(256,256,3),
#                       pooling=NULL,
#                       classes=1000#,
#                       # **kwargs
#                       )
# sink("~/Desktop/application_efficientnet_b6.txt")
# summary(ENB6)
# sink()

EfficientNetB7 = function(include_top=TRUE,
                   weights='imagenet',
                   input_tensor=NULL,
                   input_shape=NULL,
                   pooling=NULL,
                   activation="relu",
                   classes=1000#,
                   # **kwargs
                   ){
  return(
    EfficientNet(2.0, 3.1, 600, 0.5,
                      model_name='efficientnet-b7',
                      include_top=include_top, weights=weights,
                      input_tensor=input_tensor, input_shape=input_shape,
                      activation_fn = activation,
                      pooling=pooling, classes=classes#,
                      # **kwargs
                      )
    )
}

# ENB7 = EfficientNetB7(include_top=TRUE,
#                       weights='imagenet',
#                       input_tensor=NULL,
#                       input_shape=c(256,256,3),
#                       pooling=NULL,
#                       classes=1000#,
#                       # **kwargs
#                       )
# sink("~/Desktop/application_efficientnet_b7.txt")
# summary(ENB7)
# sink()


