## source
## https://github.com/qubvel/classification_models/blob/a0f006e05485a34ccf871c421279864b0ccd220b/classification_models/models/resnext.py
## https://github.com/qubvel/classification_models/blob/a0f006e05485a34ccf871c421279864b0ccd220b/classification_models/models/_common_blocks.py#L4

backend = NULL
layers = NULL
models = NULL
keras_utils = NULL

WEIGHTS_COLLECTION = list(

    # ResNeXt50
    'resnext50_top' = list(
        'model'= 'resnext50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000.h5',
        'name'= 'resnext50_imagenet_1000.h5',
        'md5'= '7c5c40381efb044a8dea5287ab2c83db'
    ),

    'resnext50' = list(
        'model'= 'resnext50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000_no_top.h5',
        'name'= 'resnext50_imagenet_1000_no_top.h5',
        'md5'= '7ade5c8aac9194af79b1724229bdaa50'
    ),
    
    'resnext50google_top' = list(
        'model'= 'resnext50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//storage.googleapis.com/tensorflow/keras-applications/resnet/',
        'name'= 'resnext50_imagenet_1000.h5',
        'md5'= '67a5b30d522ed92f75a1f16eef299d1a'
    ),
    
    'resnext50google' = list(
        'model'= 'resnext50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//storage.googleapis.com/tensorflow/keras-applications/resnet/',
        'name'= 'resnext50_imagenet_1000_no_top.h5',
        'md5'= '62527c363bdd9ec598bed41947b379fc'
    ),
    
    # 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/'
    #   'resnext50'= c('67a5b30d522ed92f75a1f16eef299d1a',
    #                  '62527c363bdd9ec598bed41947b379fc'),

        # ResNeXt101
    'resnext101_top' = list(
        'model'= 'resnext101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000.h5',
        'name'= 'resnext101_imagenet_1000.h5',
        'md5'= '432536e85ee811568a0851c328182735'
    ),

    'resnext101' = list(
        'model'= 'resnext101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000_no_top.h5',
        'name'= 'resnext101_imagenet_1000_no_top.h5',
        'md5'= '91fe0126320e49f6ee607a0719828c7e'
    ),
    
    'resnext101google_top' = list(
        'model'= 'resnext101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//storage.googleapis.com/tensorflow/keras-applications/resnet/', 
        # 'name'= 'resnext101_imagenet_1000_no_top.h5',
        'md5'= '34fb605428fcc7aa4d62f44404c11509'
    ),
    
    'resnext101google' = list(
        'model'= 'resnext101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//storage.googleapis.com/tensorflow/keras-applications/resnet/', 
        # 'name'= 'resnext101_imagenet_1000_no_top.h5',
        'md5'= '0f678c91647380debd923963594981b3'
    )
    
)


MODELS_PARAMS = list(
    'resnext50' = list(model_name = 'resnext50',
                       repetitions = c(3, 4, 6, 3)),
    'resnext101' = list(model_name = 'resnext101',
                        repetitions =  c(3, 4, 23, 3))
)

# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

handle_block_names = function(stage, block){
    name_base = paste0("stage", stage+1, "_unit", block, "_")
    conv_name = paste0(name_base, 'conv')
    bn_name = paste0(name_base, 'bn')
    relu_name = paste0(name_base, 'relu')
    sc_name = paste0(name_base, 'sc')
    return(list("conv_name" = conv_name,
                "bn_name" = bn_name,
                "relu_name" = relu_name,
                "sc_name" = sc_name))
}

get_conv_params = function(params){
    default_conv_params = list(
        'kernel_initializer'= 'glorot_uniform',
        'use_bias'= FALSE,
        'padding'= 'valid'
    )
    return(default_conv_params)
}

get_bn_params = function(params){
    axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
    default_bn_params = list(
        'axis' = axis,
        'momentum' = 0.99,
        'epsilon' = 2e-5,
        'center' = TRUE,
        'scale' = TRUE
    )
    return(default_bn_params)
}

# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

slice_tensor = function(x, start, stop, axis){
    if(axis == -1){
        return(x = layer_lambda(list(x),
                   function(x){
                     x = x[[1]][,,,start:stop]
                   },
                   output_shape = unlist(dim(x))#,
                #    name = 'slice'
                   ))
    } else if(axis == 1){
        return(x = layer_lambda(list(x),
                   function(x){
                     x = x[[1]][,start:stop,,]
                   },
                   output_shape = unlist(dim(x))#,
                #    name = 'slice'
                   ))
    } 
    # else {
    #     raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))
    # }
}

GroupConv2D = function(input_tensor,
                       filters,
                       kernel_size,
                       strides=c(1, 1),
                       groups=32,
                       kernel_initializer='he_uniform',
                       use_bias=TRUE,
                       activation='linear',
                       padding='valid'
                       ){
    # """
    # Grouped Convolution Layer implemented as a Slice,
    # Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.
    # Args:
    #     filters: Integer, the dimensionality of the output space
    #         (i.e. the number of output filters in the convolution).
    #     kernel_size: An integer or tuple/list of a single integer,
    #         specifying the length of the 1D convolution window.
    #     strides: An integer or tuple/list of a single integer, specifying the stride
    #         length of the convolution.
    #     groups: Integer, number of groups to split input filters to.
    #     kernel_initializer: Regularizer function applied to the kernel weights matrix.
    #     use_bias: Boolean, whether the layer uses a bias vector.
    #     activation: Activation function to use (see activations).
    #         If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    #     padding: one of "valid" or "same" (case-insensitive).
    # Input shape:
    #     4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".
    # Output shape:
    #     4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
    #     rows and cols values might have changed due to padding.
    # """

    # group_conv_params = conv_params

    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    slice_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
    # input_tensor = layer_input(c(512, 512, 3))
    # groups = 32 
    # filters = 256
    # groups
    # filters
    # input_tensor = x
    # x_groupcomnv2_backup = x
    # kernel_size = c(3, 3)
    # kernel_initializer='he_uniform'
    # use_bias=TRUE
    # activation='linear'
    # padding='valid'
    
    # layer = function(input_tensor){
        inp_ch = floor(unlist(rev(k_int_shape(input_tensor))[1]) / groups) #+ 1 # input grouped channels
        out_ch = floor(filters / groups)  # output grouped channels

        # c(32 - 1) * inp_ch + 1
        # groups * inp_ch

        blocks = list()
        for(c in 1:groups){
            # slice_arguments = list('start' = c * inp_ch,
            #                        'stop' = c(c + 1) * inp_ch,
            #                        'axis' = slice_axis)
            x = slice_tensor(input_tensor, start = c(c - 1) * inp_ch + 1, stop = c * inp_ch, slice_axis)
            # x = layer_lambda(input_tensor, slice_tensor, arguments=slice_arguments)
            x = layer_conv_2d(x, 
                              out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)
            blocks[[c]] = x
        }
            

        x = layer_concatenate(blocks, axis=slice_axis)
        return(x)
    # }
    # return(layer)
}

conv_block = function(input_tensor,
                      filters,
                      stage,
                      block,
                      strides=c(2, 2)){
    # """The conv block is the block that has conv layer at shortcut.
    # # Arguments
    #     filters: integer, used for first and second conv layers, third conv layer double this value
    #     strides: tuple of integers, strides for conv (3x3) layer in block
    #     stage: integer, current stage label, used for generating layer names
    #     block: integer, current block label, used for generating layer names
    # # Returns
    #     Output layer for the block.
    # """

    ## for testing
    # filters = 128
    # stage = 1
    # block = 1
    # strides = c(1,1)
    # input_tensor = layer_input(shape = c(512, 512, 3))

    # layer = function(input_tensor){

        # extracting params and names for layers
        bn_params = get_bn_params()
        conv_params = get_conv_params()
        
        temp_name = handle_block_names(stage, block)
        conv_name = temp_name$conv_name
        bn_name = temp_name$bn_name
        relu_name = temp_name$relu_name
        sc_name = temp_name$sc_name

        x = input_tensor

        x = layer_conv_2d(input_tensor, filters, c(1, 1), name=paste0(conv_name, '1'),
            kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
        x = layer_batch_normalization(x, name=paste0(bn_name, '1'), axis = bn_params$axis,
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)
        x = layer_activation(x, 'relu', name=paste0(relu_name, '1'))
        x = layer_zero_padding_2d(x, padding=c(1, 1))

        x = GroupConv2D(x, filters, c(3, 3), strides=strides, 
            kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
        
        x = layer_batch_normalization(x, name=paste0(bn_name, '2'), axis = bn_params$axis,
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)
        x = layer_activation(x, 'relu', name=paste0(relu_name, '2'))

        x = layer_conv_2d(x, filters * 2, c(1, 1), name=paste0(conv_name, '3'), 
            kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
        x = layer_batch_normalization(x, name=paste0(bn_name, '3'), axis = bn_params$axis, 
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)

        shortcut = layer_conv_2d(input_tensor, filters * 2, c(1, 1), name=sc_name, strides=strides, 
            kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
        shortcut = layer_batch_normalization(shortcut, name=paste0(sc_name, '_bn'), axis = bn_params$axis, 
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)
        x = layer_add(list(x, shortcut))

        x = layer_activation(x, 'relu', name=relu_name)
        return(x)
    # }
    
    # return(layer)
}

identity_block = function(input_tensor, filters, stage, block){
    # """The identity block is the block that has no conv layer at shortcut.
    # # Arguments
    #     filters: integer, used for first and second conv layers, third conv layer double this value
    #     stage: integer, current stage label, used for generating layer names
    #     block: integer, current block label, used for generating layer names
    # # Returns
    #     Output layer for the block.
    # """
    ## for testing

    # def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        
        temp_name = handle_block_names(stage, block)
        conv_name = temp_name$conv_name
        bn_name = temp_name$bn_name
        relu_name = temp_name$relu_name
        sc_name = temp_name$sc_name

        x = input_tensor
        
        x = layer_conv_2d(x, filters, c(1, 1), name=paste0(conv_name, '1'), 
            kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
        x = layer_batch_normalization(x, name=paste0(bn_name, '1'), axis = bn_params$axis, 
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)
        x = layer_activation(x, 'relu', name=paste0(relu_name, '1'))

        x = layer_zero_padding_2d(x, padding=c(1, 1))
        x = GroupConv2D(x, filters, c(3, 3), kernel_initializer = conv_params$kernel_initializer, 
            use_bias = conv_params$use_bias, padding = conv_params$padding)
        x = layer_batch_normalization(x, name=paste0(bn_name, '2'), axis = bn_params$axis, 
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)
        x = layer_activation(x, 'relu', name=paste0(relu_name, '2'))

        x = layer_conv_2d(x, filters * 2, c(1, 1), name=paste0(conv_name, '3'), 
            kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
        x = layer_batch_normalization(x, name=paste0(bn_name, '3'), axis = bn_params$axis, 
            epsilon = bn_params$epsilon, momentum = bn_params$momentum)

        x = layer_add(list(x, input_tensor))

        x = layer_activation(x, 'relu', name=relu_name)
        return(x)

    # return layer
}

ResNeXt = function(
        model_name,
        input_tensor=NULL,
        input_shape=NULL,
        include_top=TRUE,
        classes=1000,
        weights='imagenet'
        # **kwargs
        ){
    # """Instantiates the ResNet, SEResNet architecture.
    # Optionally loads weights pre-trained on ImageNet.
    # Note that the data format convention used by the model is
    # the one specified in your Keras config at `~/.keras/keras.json`.
    # Args:
    #     include_top: whether to include the fully-connected
    #         layer at the top of the network.
    #     weights: one of `NULL` (random initialization),
    #           'imagenet' (pre-training on ImageNet),
    #           or the path to the weights file to be loaded.
    #     input_tensor: optional Keras tensor
    #         (i.e. output of `layers.Input()`)
    #         to use as image input for the model.
    #     input_shape: optional shape tuple, only to be specified
    #         if `include_top` is FALSE (otherwise the input shape
    #         has to be `(224, 224, 3)` (with `channels_last` data format)
    #         or `(3, 224, 224)` (with `channels_first` data format).
    #         It should have exactly 3 inputs channels.
    #     classes: optional number of classes to classify images
    #         into, only to be specified if `include_top` is TRUE, and
    #         if no `weights` argument is specified.
    # Returns:
    #     A Keras model instance.
    # Raises:
    #     ValueError: in case of invalid argument for `weights`,
    #         or invalid input shape.
    # """

    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    ## for testing
    # input_tensor = NULL
    # input_shape = c(512, 512, 3)
    # model_name = "resnext101"
    # include_top = TRUE
    # classes = 1000
    # weigths = "imagenet"
    
    # get parameters for model layers
    model_params = MODELS_PARAMS[[model_name]]
    no_scale_bn_params = get_bn_params()
    no_scale_bn_params$scale = FALSE
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 128

    # define input
    if(is.null(input_tensor)){
        input = layer_input(shape=input_shape, name='data')
    } else {
        if(backend()$is_keras_tensor(input_tensor) == FALSE){
            input = layer_input(tensor=input_tensor, shape=input_shape)
        } else { 
            input = input_tensor 
        }
    }
    x = input
    
    # resnext bottom
    x = layer_batch_normalization(x,
                                  name='bn_data',
                                  axis = no_scale_bn_params$axis,
                                  epsilon = no_scale_bn_params$epsilon,
                                  momentum = no_scale_bn_params$momentum,
                                  center = no_scale_bn_params$center,
                                  scale = no_scale_bn_params$scale)
    x = layer_zero_padding_2d(x, padding=c(3, 3))
    x = layer_conv_2d(x, 64, c(7, 7), strides=c(2, 2), name='conv0', 
        kernel_initializer = conv_params$kernel_initializer, use_bias = conv_params$use_bias, padding = conv_params$padding)
    x = layer_batch_normalization(x, name='bn0', axis = bn_params$axis,
        epsilon = bn_params$epsilon, momentum = bn_params$momentum)
    x = layer_activation(x, 'relu', name='relu0')
    x = layer_zero_padding_2d(x, padding=c(1, 1))
    x = layer_max_pooling_2d(x, c(3, 3), strides=c(2, 2), padding='valid', name='pooling0')

    # resnext body
    # init_filters = 128
    # for stage, rep in enumerate(model_params.repetitions):
    #     for block in range(rep):

    #         filters = init_filters * (2 ** stage)

    #         # first block of first stage without strides because we have maxpooling before
    #         if stage == 0 and block == 0:
    #             x = conv_block(filters, stage, block, strides=(1, 1), **kwargs)(x)

    #         elif block == 0:
    #             x = conv_block(filters, stage, block, strides=(2, 2), **kwargs)(x)

    #         else:
    #             x = identity_block(filters, stage, block, **kwargs)(x)

    for(i in 1:(length(model_params$repetitions))){ ## stage 
        # print(paste0("i_",i))
        
        # increase number of filters with each stage
        for(j in 1:(model_params$repetitions[i])){ ## block
            filters = init_filters * (2 ** (i - 1))
            # print(paste0("j_--",j))
            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if(i == 1 & j == 1){
                x = do.call("conv_block", list(input_tensor=x,
                                               filters=filters,
                                               stage = i,
                                               block = j,
                                               strides=1
                                               ))         
            } else if(j == 1){
                x = do.call("conv_block", list(input_tensor=x,
                                               filters=filters,
                                               stage = i,
                                               block = j,
                                               strides=2
                                               ))
            } else {
                x = do.call("identity_block", list(input_tensor=x,
                                                   filters=filters,
                                                   stage = i,
                                                   block = j
                                                   ))
            }
        }
    }

        # resnext top
    if(include_top){
        x = layer_global_average_pooling_2d(x, name = 'pool1')
        x = layer_dense(x, classes, name = 'fc1')
        x = layer_activation(x, 'softmax', name='output')
    }
    
    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    # if(!is.null(input_tensor)){
    #     inputs = keras_utils.get_source_inputs(input_tensor)
    # } else {
        inputs = input
    # }

    # Create model
    model = keras_model(inputs, x)

    if(!is.null(weights)){
        weights_name = ifelse(include_top == TRUE, paste0(model_name, "_top"), paste0(model_name))
        weights_params = WEIGHTS_COLLECTION[[weights_name]]
        weights_path = get_file(weights_params$name,
                                str_replace(weights_params$url, "=", ":"))
        model %>% load_model_weights_hdf5(filepath = weights_path)#,
                                        #   by_name = FALSE,
                                        #   skip_mismatch = FALSE,
                                        #   reshape = FALSE)
    } 

    return(model)

}

# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

ResNeXt50 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(
     ResNeXt(
        model_name='resnext50',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights        
        )
    )
}

ResNeXt101 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(
     ResNeXt(
        model_name='resnext101',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights
        )
    )
}

# resnext101 = ResNeXt101(input_shape=c(512, 512, 3), include_top = FALSE, weights="imagenet")
# sink("~/Desktop/application_resnext101.txt")
# summary(resnext101)
# sink()

# resnext50 = ResNeXt50(input_shape=c(512, 512, 3), include_top = TRUE, weights="imagenet")
# sink("~/Desktop/application_resnext50.txt")
# summary(resnext50)
# sink()


# testmod = ResNeXt101(input_shape=c(512, 512, 3), input_tensor=NULL,
#     weights="imagenet", classes=1000, include_top=TRUE)

# def preprocess_input(x, **kwargs):
#     return(x)

# setattr(ResNeXt50, '__doc__', ResNeXt.__doc__)
# setattr(ResNeXt101, '__doc__', ResNeXt.__doc__)