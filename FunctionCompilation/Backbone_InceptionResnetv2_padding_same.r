# based on original keras code

conv2d_bn = function(x,
                     filters,
                     kernel_size,
                     strides=c(1,1),
                     padding='same',
                     activation='relu',
                     use_bias=F,
                     name=NULL) {
  x = layer_conv_2d(x,
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=use_bias,
                    name=name)
  if(use_bias == F){
    bn_axis = ifelse(k_image_data_format() == 'channels_first', 1, -1)
    bn_name = ifelse(is.null(name), "", paste0(name, "_bn"))
    x = layer_batch_normalization(x, axis=bn_axis, scale=F, name=bn_name)
    if(!is.null(activation)){
      ac_name = ifelse(is.null(name), "", paste0(name, "_ac"))
      x = layer_activation(x, activation, name=ac_name)
      x
    } else {
      x
    }
  } else {
    x
  }
}

residual_block = function(blockInput, num_filters=16, activation = "relu"){
  x = layer_activation(blockInput, activation)
  x = layer_batch_normalization(x)
  blockInput = layer_batch_normalization(blockInput)
  x = conv2d_bn(x,num_filters, c(3L,3L), strides=c(1,1), padding='same', activation='relu',
                use_bias=F, name=NULL) 
  x = conv2d_bn(x,num_filters, c(3L,3L), strides=c(1,1), padding='same', activation=NULL,
                use_bias=F, name=NULL) 
  # x = convolution_block(x, num_filters, c(3L,3L), use_bias = T)
  # x = convolution_block(x, num_filters, c(3L,3L), use_bias = F)
  x = layer_add(list(x, blockInput))
  x
}

## inception_resnet_block
inception_resnet_block = function(x,
                                  scale,
                                  block_type,
                                  block_idx,
                                  activation='relu'){
  if(block_type == 'block35'){
    branch_0 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(branch_1, 32, 3)
    branch_2 = conv2d_bn(x, 32, 1)
    branch_2 = conv2d_bn(branch_2, 48, 3)
    branch_2 = conv2d_bn(branch_2, 64, 3)
    branches = list(branch_0, branch_1, branch_2)
  } else if(block_type == 'block17'){
    branch_0 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(x, 128, 1)
    branch_1 = conv2d_bn(branch_1, 160, list(1, 7))
    branch_1 = conv2d_bn(branch_1, 192, list(7, 1))
    branches = list(branch_0, branch_1)
  } else if(block_type == 'block8'){
    branch_0 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(branch_1, 224, list(1, 3))
    branch_1 = conv2d_bn(branch_1, 256, list(3, 1))
    branches = list(branch_0, branch_1)
  } else {
    stop(cat('Unknown Inception-ResNet block type. ',
             'Expects \"block35\", \"block17\" or \"block8\", ',
             'but got: ', as.character(block_type)))
  } 
  # block_type = "block35"
  block_name = paste0(block_type, '_', as.character(block_idx))
  channel_axis = ifelse(backend()$image_data_format() == 'channels_first', 1, -1)
  mixed = layer_concatenate(branches, axis=channel_axis, name=paste0(block_name, '_mixed'))
  up = conv2d_bn(mixed,
                 k_int_shape(x)[[4]],
                 1,
                 activation=NULL,
                 use_bias=T,
                 name=paste0(block_name, '_conv'))
  # scale = 0.17
  x = layer_lambda(list(x, up),
                   function(x){
                     x = x[[1]] + x[[2]]*scale
                   },
                   output_shape = unlist(dim(x)),
                   name = block_name)
  
  if(!is.null(activation)){
    x = layer_activation(x, activation, name=paste0(block_name, '_ac'))
    x
  } else {
    x
  }
}
  
InceptionResNetV2Same = function(include_top=TRUE,
                                 weights='imagenet',
                                 input_tensor=NULL,
                                 input_shape=NULL,
                                 pooling=NULL,
                                 classes,
                                 load_irv2_weigths=TRUE){
  # include_top=F
  # input_shape = c(256, 256, 3)
  img_input = layer_input(shape = input_shape)
  
  # Stem block: 35 x 35 x 192
  x = conv2d_bn(img_input, 32, 3, strides=c(2,2), padding='same')
  x = conv2d_bn(x, 32, 3, padding='same')
  x = conv2d_bn(x, 64, 3)
  conv1 = x
  x = layer_max_pooling_2d(x, 3, strides=c(2,2), padding='same')
  x = conv2d_bn(x, 80, 1, padding='same')
  x = conv2d_bn(x, 192, 3, padding='same')
  conv2 = x
  x = layer_max_pooling_2d(x, 3, strides=c(2,2), padding='same')
  
  # Mixed 5b (Inception-A block): 35 x 35 x 320
  branch_0 = conv2d_bn(x, 96, 1)
  branch_1 = conv2d_bn(x, 48, 1)
  branch_1 = conv2d_bn(branch_1, 64, 5)
  branch_2 = conv2d_bn(x, 64, 1)
  branch_2 = conv2d_bn(branch_2, 96, 3)
  branch_2 = conv2d_bn(branch_2, 96, 3)
  branch_pool = layer_average_pooling_2d(x, 3, strides=c(1,1), padding='same')
  branch_pool = conv2d_bn(branch_pool, 64, 1)
  branches = list(branch_0, branch_1, branch_2, branch_pool)
  channel_axis = ifelse(backend()$image_data_format() == 'channels_first', 1, -1)
  x = layer_concatenate(branches, axis=channel_axis, name='mixed_5b')
  
  # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
  for(block_idx in 1:10){
    x = inception_resnet_block(x,
                               scale=0.17,
                               block_type='block35',
                               block_idx=block_idx)
  }
  conv3 = x
  
  # Mixed 6a (Reduction-A block): 17 x 17 x 1088
  branch_0 = conv2d_bn(x, 384, 3, strides=c(2,2), padding='same')
  branch_1 = conv2d_bn(x, 256, 1)
  branch_1 = conv2d_bn(branch_1, 256, 3)
  branch_1 = conv2d_bn(branch_1, 384, 3, strides=c(2,2), padding='same')
  branch_pool = layer_max_pooling_2d(x, 3, strides=c(2.2), padding='same')
  branches = list(branch_0, branch_1, branch_pool)
  x = layer_concatenate(branches, axis=channel_axis, name='mixed_6a')
  
  # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
  for(block_idx in 1:20){
    x = inception_resnet_block(x,
                               scale=0.1,
                               block_type='block17',
                               block_idx=block_idx)
  }
  conv4 = x
  
  # Mixed 7a (Reduction-B block): 8 x 8 x 2080
  branch_0 = conv2d_bn(x, 256, 1)
  branch_0 = conv2d_bn(branch_0, 384, 3, strides=c(2,2), padding='same')
  branch_1 = conv2d_bn(x, 256, 1)
  branch_1 = conv2d_bn(branch_1, 288, 3, strides=c(2,2), padding='same')
  branch_2 = conv2d_bn(x, 256, 1)
  branch_2 = conv2d_bn(branch_2, 288, 3)
  branch_2 = conv2d_bn(branch_2, 320, 3, strides=c(2,2), padding='same')
  branch_pool = layer_max_pooling_2d(x, 3, strides=c(2,2), padding='same')
  branches = list(branch_0, branch_1, branch_2, branch_pool)
  x = layer_concatenate(branches, axis=channel_axis, name='mixed_7a')
  
  # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
  for(block_idx in 1:9){
    x = inception_resnet_block(x,
                               scale=0.2,
                               block_type='block8',
                               block_idx=block_idx)
  }
    
  x = inception_resnet_block(x,
                             scale=1.0,
                             activation=NULL,
                             block_type='block8',
                             block_idx=10)
  
  # Final convolution block: 8 x 8 x 1536
  x = conv2d_bn(x, 1536, 1, name='conv_7b')
  conv5 = x
  
  # classes = 3
  # pooling = NULL
  # include_top = F
  if(include_top == T){
    # Classification block
    x = layer_global_average_pooling_2d(x, name='avg_pool')
    x = layer_dense(x, classes, activation='softmax', name='predictions')
  } else {
    if(!is.null(pooling)){
      if(pooling == 'avg'){
        x = layer_global_average_pooling_2d(x)
      } else if(pooling == 'max'){
        x = layer_global_max_pooling_2d(x)
      } else {
        x
      }
    }
    x
  }
    
  # Create model
  model = keras_model(img_input, list(conv1, conv2, conv3, conv4, conv5))#, name='inception_resnet_v2')
  
  if(load_irv2_weigths == TRUE){
    app_irv2 = application_inception_resnet_v2(input_shape=input_shape, weights='imagenet', include_top=F)
    app_irv2_weight = get_weights(app_irv2)
    model %>% set_weights(app_irv2_weight)
    model
  } else {
    model
  }
  
  
}

# IRv2 = InceptionResNetV2Same(input_shape = c(512, 512, 3), 
#                             include_top = F,
#                             weights = "imagenet")
