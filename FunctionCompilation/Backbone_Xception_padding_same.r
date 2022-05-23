# library(keras)
# library(tidyverse)

Xception = function(include_top=TRUE, weights='imagenet',
             input_tensor=NULL, input_shape=NULL,
             pooling=NULL,
             classes=1000){

    ##for testing
    # include_top=TRUE
    # weights='imagenet'
    # input_tensor=NULL
    # input_shape=c(512, 512, 3)
    # pooling=NULL
    # classes=1000
    # TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
    TF_WEIGHTS_PATH = paste0(RelPath, "/Weights/xception_weights_tf_dim_ordering_tf_kernels.h5")
    # TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
    TF_WEIGHTS_PATH_NO_TOP = paste0(RelPath, "/Weights/xception_weights_tf_dim_ordering_tf_kernels_no_top.h5")

    # Determine proper input shape
    img_input = layer_input(shape = input_shape)

    x =layer_conv_2d(img_input, 32, c(3, 3), strides=c(2, 2), use_bias=FALSE, name='block1_conv1', padding="same")
    x = layer_batch_normalization(x, name='block1_conv1_bn')
    x = layer_activation(x, 'relu', name='block1_conv1_act')
    x =layer_conv_2d(x, 64, c(3, 3), use_bias=FALSE, name='block1_conv2', padding='same')
    x = layer_batch_normalization(x, name='block1_conv2_bn')
    x = layer_activation(x, 'relu', name='block1_conv2_act')

    residual = layer_conv_2d(x, 128, c(1, 1), strides=c(2, 2), padding='same', use_bias=FALSE)
    residual = layer_batch_normalization(residual)

    x = layer_separable_conv_2d(x, 128, c(3, 3), padding='same', use_bias=FALSE, name='block2_sepconv1')
    x = layer_batch_normalization(x, name='block2_sepconv1_bn')
    x = layer_activation(x, 'relu', name='block2_sepconv2_act')
    x = layer_separable_conv_2d(x, 128, c(3, 3), padding='same', use_bias=FALSE, name='block2_sepconv2')
    x = layer_batch_normalization(x, name='block2_sepconv2_bn')

    x = layer_max_pooling_2d(x, c(3, 3), strides=c(2, 2), padding='same', name='block2_pool')
    x = layer_add(list(x, residual))

    residual = layer_conv_2d(x, 256,c(1, 1), strides=c(2, 2), padding='same', use_bias=FALSE)
    residual = layer_batch_normalization(residual)

    x = layer_activation(x, 'relu', name='block3_sepconv1_act')
    x = layer_separable_conv_2d(x, 256, c(3, 3), padding='same', use_bias=FALSE, name='block3_sepconv1')
    x = layer_batch_normalization(x, name='block3_sepconv1_bn')
    x = layer_activation(x, 'relu', name='block3_sepconv2_act')
    x = layer_separable_conv_2d(x, 256, c(3, 3), padding='same', use_bias=FALSE, name='block3_sepconv2')
    x = layer_batch_normalization(x, name='block3_sepconv2_bn')

    x = layer_max_pooling_2d(x, c(3, 3), strides=c(2, 2), padding='same', name='block3_pool')
    x = layer_add(list(x, residual))

    residual = layer_conv_2d(x, 728, c(1, 1), strides=c(2, 2), padding='same', use_bias=FALSE)
    residual = layer_batch_normalization(residual)

    x = layer_activation(x, 'relu', name='block4_sepconv1_act')
    x = layer_separable_conv_2d(x, 728, c(3, 3), padding='same', use_bias=FALSE, name='block4_sepconv1')
    x = layer_batch_normalization(x, name='block4_sepconv1_bn')
    x = layer_activation(x, 'relu', name='block4_sepconv2_act')
    x = layer_separable_conv_2d(x, 728, c(3, 3), padding='same', use_bias=FALSE, name='block4_sepconv2')
    x = layer_batch_normalization(x, name='block4_sepconv2_bn')

    x = layer_max_pooling_2d(x, c(3, 3), strides=c(2, 2), padding='same', name='block4_pool')
    x = layer_add(list(x, residual))

    for(i in 1:8){
        residual = x
        prefix = paste0('block', (i + 5))

        x = layer_activation(x, 'relu', name=paste0(prefix, '_sepconv1_act'))
        x = layer_separable_conv_2d(x, 728, c(3, 3), padding='same', use_bias=FALSE, name=paste0(prefix, '_sepconv1'))
        x = layer_batch_normalization(x, name=paste0(prefix, '_sepconv1_bn'))
        x = layer_activation(x, 'relu', name=paste0(prefix, '_sepconv2_act'))
        x = layer_separable_conv_2d(x, 728, c(3, 3), padding='same', use_bias=FALSE, name=paste0(prefix, '_sepconv2'))
        x = layer_batch_normalization(x, name=paste0(prefix, '_sepconv2_bn'))
        x = layer_activation(x, 'relu', name=paste0(prefix, '_sepconv3_act'))
        x = layer_separable_conv_2d(x, 728, c(3, 3), padding='same', use_bias=FALSE, name=paste0(prefix, '_sepconv3'))
        x = layer_batch_normalization(x, name=paste0(prefix, '_sepconv3_bn'))

        x = layer_add(list(x, residual))
    }
    
    residual = layer_conv_2d(x, 1024,c(1, 1), strides=c(2, 2), padding='same', use_bias=FALSE)
    residual = layer_batch_normalization(residual)

    x = layer_activation(x, 'relu', name='block14_sepconv1_act')
    x = layer_separable_conv_2d(x, 728, c(3, 3), padding='same', use_bias=FALSE, name='block14_sepconv1')
    x = layer_batch_normalization(x, name='block14_sepconv1_bn')
    x = layer_activation(x, 'relu', name='block14_sepconv2_act')
    x = layer_separable_conv_2d(x, 1024, c(3, 3), padding='same', use_bias=FALSE, name='block14_sepconv2')
    x = layer_batch_normalization(x, name='block14_sepconv2_bn')
    
    x = layer_max_pooling_2d(x, c(3, 3), strides=c(2, 2), padding='same', name='block14_pool')
    x = layer_add(list(x, residual))

    x = layer_separable_conv_2d(x, 1536, c(3, 3), padding='same', use_bias=FALSE, name='block15_sepconv1')
    x = layer_batch_normalization(x, name='block15_sepconv1_bn')
    x = layer_activation(x, 'relu', name='block15_sepconv1_act')

    x = layer_separable_conv_2d(x, 2048, c(3, 3), padding='same', use_bias=FALSE, name='block15_sepconv2')
    x = layer_batch_normalization(x, name='block15_sepconv2_bn')
    x = layer_activation(x, 'relu', name='block15_sepconv2_act')

    if(include_top){
        x = layer_global_average_pooling_2d(x, name='avg_pool')
        x = layer_dense(x, classes, activation='softmax', name='predictions')
    } else {
        if(pooling == 'avg'){
            x = layer_global_average_pooling_2d(x)
        } else if(pooling == 'max'){
            x = layer_global_max_pooling_2d(x)
        }
    }
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if(!is.null(input_tensor)){
        inputs = get_source_inputs(input_tensor)
    } else {
        inputs = img_input
    }
    
    # Create model.
    model = keras_model(inputs=inputs, outputs=x, name='xception')

    # load weights
    if(weights == 'imagenet'){
        if(include_top == TRUE){ 
            if(str_detect(TF_WEIGHTS_PATH, "home") == FALSE){
                weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
            } else {
               weights_path = TF_WEIGHTS_PATH
            }
        } else {
            if(str_detect(TF_WEIGHTS_PATH_NO_TOP, "home") == FALSE){
                weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash='b0042744bf5b25fce3cb969f33bebb97')
            } else {
               weights_path = TF_WEIGHTS_PATH_NO_TOP
            }
        }
        model %>% load_model_weights_hdf5(weights_path)
    } else if(!is.null(weights)){
        model %>% load_model_weights_hdf5(weights)
    }

    return(model)
}

# sink("~/xception.txt")
 Xception(input_shape = c(512, 512, 3))
# sink()
