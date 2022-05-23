Save_dir = "/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgBF512_1Class--12"
Sub_dir = "--Arch_fpn--Enc_seresnext101--Drop_5--Class_1--LR_5e-05--PT_8"
Save_dir = paste0(Save_dir, "/", Sub_dir)
mod_name = paste0("unet_model_", Sub_dir, ".hdf5")
model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                  backbone = "seresnext101",
                  nlevels = NULL,
                  output_activation = ACTIVATION,
                  output_channels = 1,
                  decoder_skip = FALSE,
                  dropout = 0.5
                  )
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
weight_name = paste0("unet_model_weights_", Sub_dir, ".hdf5")
model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/", weight_name))
# model = load_model_hdf5(filepath = paste0(Save_dir,"/unet_model_",Current_i, "_", loop_id,".hdf5"), compile = F)
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
# weight_name = "unet_model_weights_--Arch_fpn--Enc_resnet101--Drop_5--Class_1--LR_5e-05--PT_8.hdf5"
# # model %>% load_model_weights_hdf5(filepath = paste0(Save_dir,"/unet_model_weights_",Current_i, "_", loop_id, ".hdf5"))
# model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/", weight_name))
# # ## dev set
X_test = list2tensor(test_input$X)
X_test = X_test[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_test) = c(dim(X_test), 1)}
Y_test = list2tensor(test_input$Y)
Y_test = Y_test[,1:HEIGHT, 1:WIDTH,]

pred_batch_size = 8
# # Y_hat <- predict(model, x = X_train)
Y_hat_test = predict(model, 
                    X_test,
                    batch_size = pred_batch_size)
Nimg = 1
display(abind(Y_hat_test[Nimg,,,1],
              Y_test[Nimg,,,1],
              along = 1))

LayersOfInterestEncoder = rev(list(2721, 2473, 585, 255, 5)) 
LayersOfInterestDecoder = c("conv2d_5904", # 16*16
                            "conv2d_5909", # 32 * 32
                            "activation_837", # 64*64
                            "activation_840", # 128*128
                            "activation_841", # 256*256
                            "activation_844"
                            )

ListLayerEncoder = list()
for(i in 1:(length(LayersOfInterestEncoder))){
    if(is.numeric(LayersOfInterestEncoder[[i]]) == TRUE){
        ListLayerEncoder[[i]] = get_layer(model, index = LayersOfInterestEncoder[[i]])$output
    } else {
        ListLayerEncoder[[i]] = get_layer(model, name = LayersOfInterestEncoder[[i]])$output
    }
}

ListLayerDecoder = list()
for(i in 1:(length(LayersOfInterestDecoder))){
    if(is.numeric(LayersOfInterestDecoder[[i]]) == TRUE){
        ListLayerDecoder[[i]] = get_layer(model, index = LayersOfInterestDecoder[[i]])$output
    } else {
        ListLayerDecoder[[i]] = get_layer(model, name = LayersOfInterestDecoder[[i]])$output
    }
}

encoder_model = keras_model(inputs = model$input, outputs = ListLayerEncoder)
encoder_activations = predict(encoder_model, 
                      X_test,
                      batch_size = pred_batch_size)
str(encoder_activations)
for(t in 1:length(ListLayerEncoder)){
    # tempIm = normalize(activations[[t]][1,,,1])
    tempIm = resize(normalize(encoder_activations[[t]][1,,,12]), 512)
    writeImage(tempIm, files = paste0("/home/gendarme/Desktop/Img_presentation/filters_encoder_", t,".png"))
}

decoder_model = keras_model(inputs = model$input, outputs = ListLayerDecoder)
decoder_activations = predict(decoder_model, 
                              X_test,
                              batch_size = pred_batch_size)
for(t in 1:length(ListLayerDecoder)){
    # tempIm = normalize(activations[[t]][1,,,1])
    tempIm = resize(normalize(decoder_activations[[t]][1,,,62]), 512)
    writeImage(tempIm, files = paste0("/home/gendarme/Desktop/Img_presentation/filters_decoder_", t,".png"))
}