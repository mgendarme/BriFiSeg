## plain unet and unet with pretrained encoder

# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
# source(paste0(RelPath, "/Scripts/FunctionCompilation/Backbone_Zoo.r"))
# library(keras)
# library(tidyverse)

## building blocks
double_conv_layer <- function(object,
                       filters,
                       kernel_size = c(3, 3),
                       padding = "same",
                       kernel_initializer = "he_normal",
                       dropout = 0.0,
                       activation = "relu"){
  
  x = layer_conv_2d(object, filters = filters, kernel_size = kernel_size, padding = padding)
  x = layer_batch_normalization(x)
  if(activation != "leaky_relu"){
    x = layer_activation(x, activation)
  } else if(activation == "leaky_relu"){
    x = layer_activation_leaky_relu(x)
  }
  x = layer_spatial_dropout_2d(x, rate = dropout)
  x = layer_conv_2d(x, filters = filters, kernel_size = kernel_size, padding = padding)
  x = layer_batch_normalization(x) 
  if(activation != "leaky_relu"){
    x = layer_activation(x, activation)
  } else if(activation == "leaky_relu"){
    x = layer_activation_leaky_relu(x)
  }
  return(x)
}
## residual block missing for decoder (UNet++ like)

## for testing
# HEIGHT = 512  
# WIDTH = 512      
# CHANNELS = 3L    
# input_shape = c(WIDTH, HEIGHT, CHANNELS)
# # nlevels = 4
# backbone = "inception_resnet_v2" #NULL
# upsample='upsampling'
# output_activation='sigmoid'
# output_channels=3
# nlevels = NULL
# dec_filters = c(16, 32, 64, 128, 256, 512)
# dropout = c(.5, .5, .5, .5, .5, .5)

build_unet = function(input_shape,
                backbone = NULL,
                nlevels = 7,
                upsample='upsampling', #c("upsampling", "transpose")
                output_activation='sigmoid',
                output_channels=3,
                dec_filters = c(32, 64, 128, 256, 512, 512, 512, 512),
                dropout = c(.0, .0, .0, .0, .0, .0, .0, .0) # nlevels + 1
                ){

  # input_shape = c(512L, 512L, 3L)
  # backbone = "seresnext101"
  # nlevels = 5L
  # upsample='upsampling'
  # output_activation='softmax'
  # output_channels=1L
  # dec_filters = c(32, 64, 128, 256, 512, 512, 512, 512)
  # dropout = rep(0, nlevels)
    
  ## Loop over contracting layers
  clayers <- clayers_pooled <- list()
  
  if(is.null(backbone)){
    ## inputs
    clayers_pooled[[1]] = layer_input(shape = input_shape)
      
    ## down
    for(i in 2:(nlevels+1)) {
      
      clayers[[i]] = double_conv_layer(clayers_pooled[[i - 1]],
                                 filters = dec_filters[i - 1],
                                 dropout = dropout[i - 1])
      
      clayers_pooled[[i]] = layer_max_pooling_2d(clayers[[i]],
                                                  pool_size = c(2, 2),
                                                  strides = c(2, 2))

    }

    unet_input = clayers_pooled[[1]]

    ## Loop over expanding layers
    elayers <- list()
    
    ## center
    elayers[[nlevels + 1]] <- double_conv_layer(clayers_pooled[[nlevels + 1]],
                                        filters = dec_filters[nlevels + 1],
                                        dropout = dropout[nlevels + 1])

  } else {
    
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
        backbone_model = tf$keras$applications$densenet$DenseNet121(input_shape = input_shape, include_top = FALSE)
      }
      if(str_detect(backbone, "169") == TRUE){
        backbone_model = tf$keras$applications$densenet$DenseNet169(input_shape = input_shape, include_top = FALSE)
      }
      if(str_detect(backbone, "201") == TRUE){
        backbone_model = tf$keras$applications$densenet$DenseNet201(input_shape = input_shape, include_top = FALSE)
      }
    } else if(str_detect(backbone_name, "nasnet") == FALSE & str_detect(backbone_name, "densenet") == FALSE){
      backbone_model = do.call(backbone_name, list(input_shape = input_shape,
                                                   include_top = ifelse(backbone == "inception_resnet_v2", FALSE, TRUE),
                                                   weights = "imagenet")) ## finalize this
    }
    
    unet_input = backbone_model$input

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
                   "Constructing --UNet-- using --", ifelse(is.null(backbone), "no backbone", backbone),
                   "-- as encoder with --", nlevels, "--levels",
                   "\n########################################################################"))
    
    ## buld input layer
    clayers[[2]] = backbone_model$input # layer_input(shape = shape) #%>%
      # conv2d_bn(kernel_size = 3, filters = dec_filters[1])#, name = paste0("dec_",i,"_1"))
    
    ## extract layers
    
    for(i in 1:(nlevels-1)){
      if(backbone_name != "InceptionResNetV2Same"){
        if(is.numeric(backbone_layers[[i]]) == TRUE){
          clayers[[i+2]] = get_layer(backbone_model, index = backbone_layers[[i]])$output
        } else {
          clayers[[i+2]] = get_layer(backbone_model, name = backbone_layers[[i]])$output
        } 
      } else {
        clayers[[i+2]] = backbone_model$output[[i]]
      } 
      if(i == 1 & str_detect(backbone_name, "nasnet") == TRUE){
        clayers[[i+2]] = layer_zero_padding_2d(clayers[[i+2]], padding = list(list(1,0),list(1,0)))
        int_shape = k_int_shape(clayers[[i+2]])
        clayers[[i+2]] = conv_bn(clayers[[i+2]], rev(int_shape)[[1]], kernel_size = 3L, stride = c(1,1), padding = "same", name = "zeropad")
        clayers[[i+2]] = conv_relu(clayers[[i+2]], rev(int_shape)[[1]], kernel_size = 3L, stride = c(1,1), padding = "same", name = "zeropad")
      }
    }

    # ## extract layers
    # for(i in 1:(nlevels-1)){
    #   if(backbone_name != "InceptionResNetV2Same"){
    #     clayers[[i+2]] = get_layer(backbone_model, backbone_layers[[i]])$output
    #   } else {
    #     clayers[[i+2]] = backbone_model$output[[i]]
    #   }
    # }

    ## Loop over expanding layers
    elayers <- list()
  
    ### center ########################################################################
    if(backbone != "inception_resnet_v2"){
      if(is.numeric(backbone_layers[[i]]) == TRUE){
        elayers[[nlevels+1]] = get_layer(backbone_model, index = backbone_layers[[nlevels]])$output
      } else {
        elayers[[nlevels+1]] = get_layer(backbone_model, name = backbone_layers[[nlevels]])$output
      }
    } else {
      elayers[[nlevels+1]] = backbone_model$output[[nlevels]]
    }
    
    ## Add a conv step ?

  }
  
  ## up ###############################################################################
  for(i in nlevels:1) {

    concat_axis = ifelse(k_image_data_format() == "channels_last", -1, 1)

    if(upsample == "upsampling"){
      elayers[[i]] = layer_upsampling_2d(elayers[[i+1]],
                                         size = c(2, 2))
    } else {
      elayers[[i]] = layer_conv_2d_transpose(elayers[[i]],
                                            filters = dec_filters[i],
                                            kernel_size = c(2, 2),
                                            strides = c(2, 2),
                                            padding = "same")
    }
        
    elayers[[i]] = layer_concatenate(list(elayers[[i]], clayers[[i+1]]), axis = concat_axis)
    elayers[[i]] = double_conv_layer(elayers[[i]], kernel_size = 3, filters = dec_filters[i])#, name = paste0("dec_",i,"_1"))
    elayers[[i]] = layer_spatial_dropout_2d(elayers[[i]], rate = dropout[i], name = paste0("dropout_",i))
    elayers[[i]] = double_conv_layer(elayers[[i]], kernel_size = 3, filters = dec_filters[i])#, name = paste0("dec_",i,"_2"))
    # if(i %in% 2:4){
    #   # filters_i = k_int_shape(elayers[[i]])[4]
    #   conv1x1_i_act = layer_conv_2d(elayers[[i]], kernel_size = 1, filters = 1, name = paste0("conv1x1_", i))
    #   conv1x1_i_act = layer_activation(conv1x1_i_act, activation = "softmax") # if doesn't work replace with sigmoid
    #   elayers[[i]] = conv1x1_i_act #layer_add(list(elayers[[i]], conv1x1_i_act))
    # }
  }
   
  ## Output layer
  outputs = layer_conv_2d(elayers[[1]],
                          filters = output_channels,
                          kernel_size = c(3, 3),
                          strides = c(1, 1),
                          padding = "same",
                          use_bias = TRUE,
                          # kernel_initializer='glorot_uniform', ## used in segmentation tools from qubvel
                          # activation = output_activation,
                          name = paste0("final_conv")) 
                          
  if(output_activation == "softmax"){
    outputs = layer_activation_softmax(outputs, name = "output_ac_softmax")
    # outputs = tf$keras$activations$softmax(outputs)
  } else {
    outputs = layer_activation(outputs, output_activation, name = paste0("output_ac_", output_activation))
  }
  
  model = keras_model(inputs = unet_input, outputs = outputs)
  return(model)
}

# testunet = build_unet(input_shape = c(512L, 512L, 3L),
#      backbone = NULL,
#      nlevels = NULL,
#      upsample = 'upsampling', #c("upsampling", "transpose")
#      output_activation = 'softmax',
#      output_channels = 3,
#      dec_filters = c(32, 64, 128, 256, 512, 512, 512, 512),
#      dropout = rep(0, 8) # nlevels + 1
#      )

# sink("~/Desktop/summary_unet_seresnext101.txt")
# summary(testunet)
# sink()