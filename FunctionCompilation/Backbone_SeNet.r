## source
## https://github.com/qubvel/classification_models/blob/a0f006e05485a34ccf871c421279864b0ccd220b/classification_models/models/senet.py#L197
## https://github.com/qubvel/classification_models/blob/a0f006e05485a34ccf871c421279864b0ccd220b/classification_models/models/_common_blocks.py#L4

require(keras)
require(tidyverse)

backend = NULL
layers = NULL
models = NULL
keras_utils = NULL

WEIGHTS_COLLECTION = list(

    'resnet18_top' = list(
        'model'= 'resnet18',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5',
        'name'= 'resnet18_imagenet_1000.h5',
        'md5'= '64da73012bb70e16c901316c201d9803'
    ),

    'resnet18' = list(
        'model'= 'resnet18',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5',
        'name'= 'resnet18_imagenet_1000_no_top.h5',
        'md5'= '318e3ac0cd98d51e917526c9f62f0b50'
    ),

    # ResNet34
    'resnet34_top' = list(
        'model'= 'resnet34',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name'= 'resnet34_imagenet_1000.h5',
        'md5'= '2ac8277412f65e5d047f255bcbd10383'
    ),

    'resnet34' = list(
        'model'= 'resnet34',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name'= 'resnet34_imagenet_1000_no_top.h5',
        'md5'= '8caaa0ad39d927cb8ba5385bf945d582'
    ),

    # ResNet50
    'resnet50_top' = list(
        'model'= 'resnet50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000.h5',
        'name'= 'resnet50_imagenet_1000.h5',
        'md5'= 'd0feba4fc650e68ac8c19166ee1ba87f'
    ),

    'resnet50' = list(
        'model'= 'resnet50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000_no_top.h5',
        'name'= 'resnet50_imagenet_1000_no_top.h5',
        'md5'= 'db3b217156506944570ac220086f09b6'
    ),

    # list(
    #     'model'= 'resnet50',
    #     'dataset'= 'imagenet11k-places365ch',
    #     'classes'= 11586,
    #     'include_top'= TRUE,
    #     'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586.h5',
    #     'name'= 'resnet50_imagenet11k-places365ch_11586.h5',
    #     'md5'= 'bb8963db145bc9906452b3d9c9917275'
    # ),

    # list(
    #     'model'= 'resnet50',
    #     'dataset'= 'imagenet11k-places365ch',
    #     'classes'= 11586,
    #     'include_top'= FALSE,
    #     'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586_no_top.h5',
    #     'name'= 'resnet50_imagenet11k-places365ch_11586_no_top.h5',
    #     'md5'= 'd8bf4e7ea082d9d43e37644da217324a'
    # ),

    # ResNet101
    'resnet101_top' = list(
        'model'= 'resnet101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000.h5',
        'name'= 'resnet101_imagenet_1000.h5',
        'md5'= '9489ed2d5d0037538134c880167622ad'
    ),

    'resnet101' = list(
        'model'= 'resnet101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000_no_top.h5',
        'name'= 'resnet101_imagenet_1000_no_top.h5',
        'md5'= '1016e7663980d5597a4e224d915c342d'
    ),

    # ResNet152
    'resnet152_top' = list(
        'model'= 'resnet152',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000.h5',
        'name'= 'resnet152_imagenet_1000.h5',
        'md5'= '1efffbcc0708fb0d46a9d096ae14f905'
    ),

    'resnet152' = list(
        'model'= 'resnet152',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000_no_top.h5',
        'name'= 'resnet152_imagenet_1000_no_top.h5',
        'md5'= '5867b94098df4640918941115db93734'
    ),

    # list(
    #     'model'= 'resnet152',
    #     'dataset'= 'imagenet11k',
    #     'classes'= 11221,
    #     'include_top'= TRUE,
    #     'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221.h5',
    #     'name'= 'resnet152_imagenet11k_11221.h5',
    #     'md5'= '24791790f6ef32f274430ce4a2ffee5d'
    # ),

    # list(
    #     'model'= 'resnet152',
    #     'dataset'= 'imagenet11k',
    #     'classes'= 11221,
    #     'include_top'= FALSE,
    #     'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221_no_top.h5',
    #     'name'= 'resnet152_imagenet11k_11221_no_top.h5',
    #     'md5'= '25ab66dec217cb774a27d0f3659cafb3'
    # ),

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

    # SE models
    'seresnet50_top' = list(
        'model'= 'seresnet50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000.h5',
        'name'= 'seresnet50_imagenet_1000.h5',
        'md5'= 'ff0ce1ed5accaad05d113ecef2d29149'
    ),

    'seresnet50' = list(
        'model'= 'seresnet50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000_no_top.h5',
        'name'= 'seresnet50_imagenet_1000_no_top.h5',
        'md5'= '043777781b0d5ca756474d60bf115ef1'
    ),

    'seresnet101_top' = list(
        'model'= 'seresnet101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000.h5',
        'name'= 'seresnet101_imagenet_1000.h5',
        'md5'= '5c31adee48c82a66a32dee3d442f5be8'
    ),

    'seresnet101' = list(
        'model'= 'seresnet101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000_no_top.h5',
        'name'= 'seresnet101_imagenet_1000_no_top.h5',
        'md5'= '1c373b0c196918713da86951d1239007'
    ),

    'seresnet152_top' = list(
        'model'= 'seresnet152',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000.h5',
        'name'= 'seresnet152_imagenet_1000.h5',
        'md5'= '96fc14e3a939d4627b0174a0e80c7371'
    ),

    'seresnet152' = list(
        'model'= 'seresnet152',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000_no_top.h5',
        'name'= 'seresnet152_imagenet_1000_no_top.h5',
        'md5'= 'f58d4c1a511c7445ab9a2c2b83ee4e7b'
    ),

    'seresnext50_top' = list(
        'model'= 'seresnext50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000.h5',
        'name'= 'seresnext50_imagenet_1000.h5',
        'md5'= '5310dcd58ed573aecdab99f8df1121d5'
    ),

    'seresnext50' = list(
        'model'= 'seresnext50',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000_no_top.h5',
        'name'= 'seresnext50_imagenet_1000_no_top.h5',
        'md5'= 'b0f23d2e1cd406d67335fb92d85cc279'
    ),

    'seresnext101_top' = list(
        'model'= 'seresnext101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000.h5',
        'name'= 'seresnext101_imagenet_1000.h5',
        'md5'= 'be5b26b697a0f7f11efaa1bb6272fc84'
    ),

    'seresnext101' = list(
        'model'= 'seresnext101',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000_no_top.h5',
        'name'= 'seresnext101_imagenet_1000_no_top.h5',
        'md5'= 'e48708cbe40071cc3356016c37f6c9c7'
    ),

    'senet154_top' = list(
        'model'= 'senet154',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000.h5',
        'name'= 'senet154_imagenet_1000.h5',
        'md5'= 'c8eac0e1940ea4d8a2e0b2eb0cdf4e75'
    ),

    'senet154' = list(
        'model'= 'senet154',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000_no_top.h5',
        'name'= 'senet154_imagenet_1000_no_top.h5',
        'md5'= 'd854ff2cd7e6a87b05a8124cd283e0f2'
    ),

    'seresnet18_top' = list(
        'model'= 'seresnet18',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000.h5',
        'name'= 'seresnet18_imagenet_1000.h5',
        'md5'= '9a925fd96d050dbf7cc4c54aabfcf749'
    ),

    'seresnet18' = list(
        'model'= 'seresnet18',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000_no_top.h5',
        'name'= 'seresnet18_imagenet_1000_no_top.h5',
        'md5'= 'a46e5cd4114ac946ecdc58741e8d92ea'
    ),

    'seresnet34_top' = list(
        'model'= 'seresnet34',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= TRUE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000.h5',
        'name'= 'seresnet34_imagenet_1000.h5',
        'md5'= '863976b3bd439ff0cc05c91821218a6b'
    ),

    'seresnet34' = list(
        'model'= 'seresnet34',
        'dataset'= 'imagenet',
        'classes'= 1000,
        'include_top'= FALSE,
        'url'= 'https=//github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000_no_top.h5',
        'name'= 'seresnet34_imagenet_1000_no_top.h5',
        'md5'= '3348fd049f1f9ad307c070ff2b6ec4cb'
    )

)

# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

get_bn_params = function(params){
    axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
    default_bn_params = list(
        'axis' = axis,
        'epsilon' = 9.999999747378752e-06
    )
    # ault_bn_params.update(params)
    return(default_bn_params)
}

get_num_channels = function(tensor){
    channels_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
    if(channels_axis == -1){
        return(unlist(rev(k_int_shape(tensor))[1]))
    } else if(channels_axis == 1){
        return(unlist(k_int_shape(tensor))[1])
    }
    # return(k_int_shape(tensor)[channels_axis])
}
    
# -------------------------------------------------------------------------
#   Common blocks
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

    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    slice_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)
    
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

expand_dims = function(x, channels_axis){
    if(channels_axis == -1){
        return(x = layer_lambda(list(x),
                   function(x){
                     x = x[[1]][,NULL,NULL,]
                   },
                   output_shape = unlist(dim(x))#,
                #    name = 'slice'
                   ))
    } else if(channels_axis == 1){
        return(x = layer_lambda(list(x),
                   function(x){
                     x = x[[1]][,,NULL,NULL]
                   },
                   output_shape = unlist(dim(x))#,
                #    name = 'slice'
                   ))
    }# else:
    #     raise ValueError("Slice axis should be in (1, 3), got {}.".format(channels_axis))
}

ChannelSE = function(input_tensor, reduction=16){
    # """
    # Squeeze and Excitation block, reimplementation inspired by
    #     https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
    # Args:
    #     reduction: channels squeeze factor
    # """
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    channels_axis = ifelse(backend()$image_data_format() == "channels_last", -1, 1)

    channels = ifelse(channels_axis == -1,
                        unlist(rev(k_int_shape(input_tensor)))[1],
                        unlist(k_int_shape(input_tensor))[1])
    
    x = input_tensor

    # squeeze and excitation block in PyTorch style with
    x = layer_global_average_pooling_2d(x)
    # x = layer_lambda(x, expand_dims, arguments=list('channels_axis'= channels_axis))
    x = expand_dims(x, channels_axis)
    x = layer_conv_2d(x, floor(channels / reduction), c(1, 1), kernel_initializer='he_uniform')
    x = layer_activation(x, 'relu')
    x = layer_conv_2d(x, channels, c(1, 1), kernel_initializer='he_uniform')
    x = layer_activation(x, 'sigmoid')

    # apply attention
    x = layer_multiply(list(input_tensor, x))

    return(x)

}

# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

SEResNetBottleneck = function(input_tensor, filters, reduction=16, strides=1, groups=NULL){
    bn_params = get_bn_params()


    x = input_tensor
    residual = input_tensor

    # bottleneck
    x = layer_conv_2d(x, floor(filters / 4), c(1, 1), kernel_initializer='he_uniform',
                        strides=strides, use_bias=FALSE)
    x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)
    x = layer_activation(x, 'relu')

    x = layer_zero_padding_2d(x, 1)
    x = layer_conv_2d(x, floor(filters / 4), c(3, 3),
                        kernel_initializer='he_uniform', use_bias=FALSE)
    x = layer_batch_normalization(x, axis = bn_params$axis, epsilon = bn_params$epsilon)
    x = layer_activation(x, 'relu')

    x = layer_conv_2d(x, filters, c(1, 1), kernel_initializer='he_uniform', use_bias=FALSE)
    x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)

    #  if number of filters or spatial dimensions changed
    #  make same manipulations with residual connection
    x_channels = get_num_channels(x)
    r_channels = get_num_channels(residual)

    if(strides != 1 | x_channels != r_channels){   
        residual = layer_conv_2d(x_channels, c(1, 1), strides=strides,
                                    kernel_initializer='he_uniform', use_bias=FALSE)(residual)
        residual = layer_batch_normalization(residual, axis = bn_params$axis, epsilon = bn_params$epsilon)
    }

    # apply attention module
    x = ChannelSE(x, reduction=reduction)

    # add residual connection
    x = layer_add(list(x, residual))

    x = layer_activation(x, 'relu')

    return(x)

}
    
SEResNeXtBottleneck = function(input_tensor, filters, reduction=16, strides=1, groups=32, base_width=4){
    bn_params = get_bn_params()

    x = input_tensor
    residual = input_tensor

    width = floor(floor(filters / 4) * base_width * groups / 64)

    # bottleneck
    x = layer_conv_2d(x, width, c(1, 1), kernel_initializer='he_uniform', use_bias=FALSE)
    x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)
    x = layer_activation(x, 'relu')

    x = layer_zero_padding_2d(x, 1)
    x = GroupConv2D(x, width, c(3, 3), strides=strides, groups=groups,
                    kernel_initializer='he_uniform', use_bias=FALSE)
    x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)
    x = layer_activation(x, 'relu')

    x = layer_conv_2d(x, filters, c(1, 1), kernel_initializer='he_uniform', use_bias=FALSE)
    x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)

    #  if number of filters or spatial dimensions changed
    #  make same manipulations with residual connection
    x_channels = get_num_channels(x)
    r_channels = get_num_channels(residual)

    if(strides != 1 | x_channels != r_channels){
        residual = layer_conv_2d(residual, x_channels, c(1, 1), strides=strides,
                                    kernel_initializer='he_uniform', use_bias=FALSE)
        residual = layer_batch_normalization(residual,  axis = bn_params$axis, epsilon = bn_params$epsilon)
    }
    
    # apply attention module
    x = ChannelSE(x, reduction=reduction)

    # add residual connection
    x = layer_add(list(x, residual))

    x = layer_activation(x, 'relu')

    return(x)

}

SEBottleneck = function(input_tensor, filters, reduction=16, strides=1, groups=64, is_first=FALSE){
    bn_params = get_bn_params()
    # modules_kwargs = ({k: v for k, v in kwargs.items()
    #                    if k in ('backend', 'layers', 'models', 'utils')})

    if(is_first){
        downsample_kernel_size = c(1, 1)
        padding = FALSE
    } else {
        downsample_kernel_size = c(3, 3)
        padding = TRUE
    }
    
    # layer = function(input_tensor){

        x = input_tensor
        residual = input_tensor

        # bottleneck
        x = layer_conv_2d(x, floor(filters / 2), c(1, 1), kernel_initializer='he_uniform', use_bias=FALSE)
        x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)
        x = layer_activation(x, 'relu')

        x = layer_zero_padding_2d(x, 1)
        x = GroupConv2D(x, filters, c(3, 3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=FALSE)
        x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)
        x = layer_activation(x, 'relu')

        x = layer_conv_2d(x, filters, c(1, 1), kernel_initializer='he_uniform', use_bias=FALSE)
        x = layer_batch_normalization(x,  axis = bn_params$axis, epsilon = bn_params$epsilon)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if(strides != 1 | x_channels != r_channels){
            if(padding){
                residual = layer_zero_padding_2d(residual, 1)
            }
            residual = layer_conv_2d(residual, x_channels, downsample_kernel_size, strides=strides,
                                    kernel_initializer='he_uniform', use_bias=FALSE)
            residual = layer_batch_normalization(residual,  axis = bn_params$axis, epsilon = bn_params$epsilon)
            
        }
        
        # apply attention module
        x = ChannelSE(x, reduction=reduction)

        # add residual connection
        x = layer_add(list(x, residual))

        x = layer_activation(x, 'relu')

        return(x)
    # }
    # return(layer)
}
# -------------------------------------------------------------------------
#   SeNet builder
# -------------------------------------------------------------------------

SENet = function(
        model_name,
        input_tensor=NULL,
        input_shape=NULL,
        include_top=TRUE,
        classes=1000,
        weights='imagenet'
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
    # model_name = "seresnext101"
    # input_tensor = layer_input(shape = c(512, 512, 3))
    # include_top = FALSE
    # classes=1000
    # weights='imagenet'

    # model_name='senet154'
    # repetitions=c(3, 8, 36, 3)
    # residual_block="SEBottleneck"
    # groups=64
    # reduction=16
    # init_filters=64
    # input_3x3=TRUE
    # dropout=0.2

    model_params = MODELS_PARAMS[[model_name]]
    residual_block = get(model_params$residual_block)
    init_filters = model_params$init_filters
    bn_params = get_bn_params()
    
    
    # define input
    if(is.null(input_tensor)){
        input = layer_input(shape=input_shape, name='input')
    } else {
        if(backend()$is_keras_tensor(input_tensor) == FALSE){
            input = layer_input(tensor=input_tensor, shape=input_shape)
        } else { 
            input = input_tensor 
        }
    }
    x = input

    if(model_params$input_3x3){

        x = layer_zero_padding_2d(x, 1)
        x = layer_conv_2d(x, init_filters, c(3, 3), strides=2,
                          use_bias=FALSE, kernel_initializer='he_uniform')
        x = layer_batch_normalization(x, axis = bn_params$axis, epsilon = bn_params$epsilon)
        x = layer_activation(x, 'relu')

        x = layer_zero_padding_2d(x, 1)
        x = layer_conv_2d(x, init_filters, c(3, 3), use_bias=FALSE,
                          kernel_initializer='he_uniform')
        x = layer_batch_normalization(x, axis = bn_params$axis, epsilon = bn_params$epsilon)
        x = layer_activation(x, 'relu')

        x = layer_zero_padding_2d(x, 1)
        x = layer_conv_2d(x, init_filters * 2, c(3, 3), use_bias=FALSE,
                          kernel_initializer='he_uniform')
        x = layer_batch_normalization(x, axis = bn_params$axis, epsilon = bn_params$epsilon)
        x = layer_activation(x, 'relu')

    } else {

        x = layer_zero_padding_2d(x, 3)
        x = layer_conv_2d(x, init_filters, c(7, 7), strides=2, use_bias=FALSE,
                          kernel_initializer='he_uniform')
        x = layer_batch_normalization(x, axis = bn_params$axis, epsilon = bn_params$epsilon)
        x = layer_activation(x, 'relu')

    }

    x = layer_zero_padding_2d(x, 1)
    x = layer_max_pooling_2d(x, c(3, 3), strides=2)
    #  backup = x
    # x = backup
    ################################################################################################
    ################################################################################################
    ################################################################################################
    # body of resnet
    filters = model_params$init_filters * 2
    for(i in 1:(length(model_params$repetitions))){
        # print(paste0("i_",i))
        
        # increase number of filters with each stage
        filters = filters * 2

        for(j in 1:(model_params$repetitions[i])){
            # print(paste0("j_--",j))
            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if(i == 1 & j == 1){
                if(model_params$residual_block == "SEBottleneck"){
                    x = do.call(model_params$residual_block, list(input_tensor=x,
                                                                  filters=filters,
                                                                  reduction=model_params$reduction,
                                                                  strides=1,
                                                                  groups=model_params$groups,
                                                                  is_first=TRUE))
                } else {
                    x = do.call(model_params$residual_block, list(input_tensor=x,
                                                                  filters=filters,
                                                                  reduction=model_params$reduction,
                                                                  strides=1,
                                                                  groups=model_params$groups))
                }
                
            } else if(i != 1 & j == 1){
                x = do.call(model_params$residual_block, list(input_tensor=x,
                                                              filters=filters,
                                                              reduction=model_params$reduction,
                                                              strides=2,
                                                              groups=model_params$groups))
            } else {
                x = do.call(model_params$residual_block, list(input_tensor=x,
                                                              filters=filters,
                                                              reduction=model_params$reduction,
                                                              strides=1,
                                                              groups=model_params$groups))
            }
        }
    }
    # x
    ################################################################################################
    ################################################################################################
    ################################################################################################
    if(include_top){
        x = layer_global_average_pooling_2d(x)
        if(!is.null(model_params$dropout)){
            x = layer_dropout(x, model_params$dropout)
        }
        x = layer_dense(x, classes)
        x = layer_activation(x, 'softmax', name='output')
    }
    
    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    # if(!is.null(input_tensor)){
    #     inputs = keras_utils.get_source_inputs(input_tensor)
    # } else {
        inputs = input
    # }
    
    model = keras_model(inputs, x)

    # if(weights){
    #     if(is.character(weights) & os.path.exists(weights)){
    #         model %>% load_weights(weights)
    #     } else {
    #         load_model_weights(model, model_params$model_name,
    #                            weights, classes, include_top)
    #     }
    # }
    # include_top = TRUE
    if(!is.null(weights)){
        weights_name = ifelse(include_top == TRUE, paste0(model_name, "_top"), paste0(model_name))
        weights_params = WEIGHTS_COLLECTION[[weights_name]]
        weights_path = get_file(weights_params$name,
                                str_replace(weights_params$url, "=", ":"))
        model %>% load_model_weights_hdf5(filepath = weights_path)  
    } 
    # # Load weights.
    # if(!is.null(weights)){
    #     # if(weights == 'imagenet'){
    #         if(include_top == TRUE){
    #             file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
    #             file_hash = WEIGHTS_HASHES[str_remove(model_name, "efficientnet-")][[1]][1]
    #         } else {
    #             file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    #             file_hash = WEIGHTS_HASHES[str_remove(model_name, "efficientnet-")][[1]][2]    
    #         }
    #         file_name = paste0(model_name, file_suff)
    #         weights_path = get_file(file_name,
    #                                 paste0(BASE_WEIGHTS_PATH, file_name))
              
    #     # } else {
    #         # model %>% load_model_weights_hdf5(filepath = weights_path)
    #     # }  
    # }

    return(model)

}

# -------------------------------------------------------------------------
#   SE Residual Models
# -------------------------------------------------------------------------
ModelParams = list(
    'ModelParams',
    list('model_name', 'repetitions', 'residual_block', 'groups',
         'reduction', 'init_filters', 'input_3x3', 'dropout')
)

MODELS_PARAMS = list(
    'seresnet50'= list(
        model_name='seresnet50', repetitions=c(3, 4, 6, 3), residual_block="SEResNetBottleneck",
        groups=1, reduction=16, init_filters=64, input_3x3=FALSE, dropout=NULL
    ),

    'seresnet101'= list(
        model_name='seresnet101', repetitions=c(3, 4, 23, 3), residual_block="SEResNetBottleneck",
        groups=1, reduction=16, init_filters=64, input_3x3=FALSE, dropout=NULL
    ),

    'seresnet152'= list(
        model_name='seresnet152', repetitions=c(3, 8, 36, 3), residual_block="SEResNetBottleneck",
        groups=1, reduction=16, init_filters=64, input_3x3=FALSE, dropout=NULL
    ),

    'seresnext50'= list(
        model_name='seresnext50', repetitions=c(3, 4, 6, 3), residual_block="SEResNeXtBottleneck",
        groups=32, reduction=16, init_filters=64, input_3x3=FALSE, dropout=NULL
    ),

    'seresnext101'= list(
        model_name='seresnext101', repetitions=c(3, 4, 23, 3), residual_block="SEResNeXtBottleneck",
        groups=32, reduction=16, init_filters=64, input_3x3=FALSE, dropout=NULL
    ),

    'senet154'= list(
        model_name='senet154', repetitions=c(3, 8, 36, 3), residual_block="SEBottleneck",
        groups=64, reduction=16, init_filters=64, input_3x3=TRUE, dropout=0.2
    )
)


 SEResNet50 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(SENet(
        model_name = 'seresnet50',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights)        
    )
}

 SEResNet101 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(SENet(
        model_name = 'seresnet101',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights)
    )
}

 SEResNet152 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(SENet(
        model_name = 'seresnet152',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights)
    )
}

 SEResNeXt50 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(SENet(
        model_name = 'seresnext50',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights)
        )
}

SEResNeXt101 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(SENet(
        model_name = 'seresnext101',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights)        
    )
}

SENet154 = function(input_shape=NULL, input_tensor=NULL, weights=NULL, classes=1000, include_top=TRUE){
    return(SENet(
        model_name = 'senet154',
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights)        
    )
}

# setest = SEResNeXt101(input_shape =  c(512L, 512L, 3L), weights = "imagenet")
# sink(file = "~/Desktop/summary_seresnext101.txt")
# summary(setest)
# sink()




