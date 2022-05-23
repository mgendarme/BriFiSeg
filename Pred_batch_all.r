library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts

# set seed for reproducibility
set.seed(11)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')

# Load custom functions
list_scripts = list("Params_batch.r", "PreprocessingAndTransformation.r",
                    "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus.r",
                    "Backbone_Zoo.r",
                    "Loss.r",
                    "CustomGenerator_CropFix.r",
                    "Inspection.r",
                    "Postprocessing.r") %>%
    map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
    source(list_scripts[[l]])
}

## Z-stack == 4
# PredPath = paste0(RelPath, "/BF_Data/BATCH_ALL/Prediction/ImgBF512_1Class--5/")
# Sub_dir = "--Arch_fpn--Enc_seresnext101--Drop_5--Class_1--LR_1e-04--PT_8--RedFct_8"

## Z-stack == c(3, 4, 5)
PredPath = paste0(RelPath, "/BF_Data/BATCH_ALL/Prediction/ImgBF512_1Class--6/")
Sub_dir = "--Arch_fpn--Enc_seresnext101--Drop_5--Class_1--LR_1e-04--PT_8--RedFct_8"

ModelPath = paste0(PredPath, Sub_dir)
Save_dir = paste0(PredPath, "SingleCellLine")
dir.create(Save_dir)

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
model %>% load_model_weights_hdf5(filepath = paste0(ModelPath, "/", weight_name))

for(i in c(
    # "A549",   # DONE
    "HELA",   # DONE
    "MCF7",   # DONE
    "RPE1"    # DONE
    # "BATCH_THP1"  # PROBLEMATIC
    )){
    CELL = i
    Current_i = CELL
    print(message(paste0("\n##################\n",
                    "Current samples: ", CELL,
                    "\n####################")))
    ## load and pre-process data
    source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_batch.r"))

    loop_id = paste0(CELL)

    message(paste0("\n######################################################\n",
                    loop_id,
                    "\n######################################################"))

    # build loop dir, plot dir and image dir
    Save_loop = paste0(Save_dir, "/", loop_id) 
    dir.create(Save_loop)
    Save_plot = paste0(Save_loop, "/Plot")
    dir.create(Save_plot)
    Save_image = paste0(Save_loop, "/Image")
    dir.create(Save_image)

    ### Generate single instances and perform measurements #####################################################
    ## generate plots for IOU and DICE
    source(paste0(RelPath, "/Scripts/FunctionCompilation/IOU_DICE_plot.r"))

    ## generate sample images
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images.r"))

    ## generate instance segmentation
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Postprocessing_1c.r"))

    ## generate plots for single IOU comparison ## can take quite some time
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/IOU_single.r"))

}
#               }
#           }
#       }
#   }
# }

# display(Y_test[1,,,2])
# display(Y_hat_test[1,,,2])
# get_layer(model, "output_ac")$output
# writeImage(normalize(Y_hat_test[1,,,1]), files = "/home/gendarme/Desktop/Img_presentation/prob_map_a549.png")

## 
# Save_dir = "/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgBF512_1Class--12"
# Save_dir = "/home/gendarme/Documents/UNet/BF_Data/THP1/Prediction/ImgBF512_1Class--1"
# # Sub_dir = "--Arch_fpn--Enc_seresnext101--Drop_5--Class_1--LR_5e-05--PT_8"
# Sub_dir = "--Arch_fpn--Enc_seresnext101--Drop_5--Class_1--LR_5e-05--PT_8"
# Save_dir = paste0(Save_dir, "/", Sub_dir)
# Save_plot = paste0(Save_dir, "/Plot")
# # dir.create(Save_plot)
# Save_image = paste0(Save_dir, "/Image")
# # dir.create(Save_image)
# mod_name = paste0("unet_model_", Sub_dir, ".hdf5")
# model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
#                   backbone = "seresnext101",
#                   nlevels = NULL,
#                   output_activation = ACTIVATION,
#                   output_channels = 1,
#                   decoder_skip = FALSE,
#                   dropout = 0.5
#                   )
# get_layer(model, "output_ac")$output
# # model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
# weight_name = paste0("unet_model_weights_", Sub_dir, ".hdf5")
# model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/", weight_name))
# # # model = load_model_hdf5(filepath = paste0(Save_dir,"/unet_model_",Current_i, "_", loop_id,".hdf5"), compile = F)
# # # model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
# # # weight_name = "unet_model_weights_--Arch_fpn--Enc_resnet101--Drop_5--Class_1--LR_5e-05--PT_8.hdf5"
# # # # model %>% load_model_weights_hdf5(filepath = paste0(Save_dir,"/unet_model_weights_",Current_i, "_", loop_id, ".hdf5"))
# # # model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/", weight_name))
# # # # ## dev set
# X_test = list2tensor(test_input$X)
# X_test = X_test[,1:HEIGHT, 1:WIDTH,]
# if(CHANNELS == 1){dim(X_test) = c(dim(X_test), 1)}
# Y_test = list2tensor(test_input$Y)
# Y_test = Y_test[,1:HEIGHT, 1:WIDTH,]
# if(length(dim(Y_test)) == 3){dim(Y_test) = c(dim(Y_test), 1)}
# str(Y_test)
# pred_batch_size = 8
# Y_hat_test = predict(model, 
#                     X_test,
#                     batch_size = pred_batch_size)
# Nimg = 1
# display(abind(Y_hat_test[Nimg,,,2],
#               Y_test[Nimg,,,2],
#               along = 1))

# ## generate instance segmentation
# source(paste0(RelPath, "/Scripts/FunctionCompilation/Postprocessing_1c.r"))
# writeImage(Y_hat_test[1,,,1], files = "/home/gendarme/Desktop/Img_presentation/prob_map_thp1.png")
# LayersOfInterestEncoder = rev(list(2721, 2473, 585, 255, 5)) 
# LayersOfInterestDecoder = c("conv2d_5904", # 16*16
#                             "conv2d_5909", # 32 * 32
#                             "activation_837", # 64*64
#                             "activation_840", # 128*128
#                             "activation_841", # 256*256
#                             "activation_844"
#                             )

# ListLayerEncoder = list()
# for(i in 1:(length(LayersOfInterestEncoder))){
#     if(is.numeric(LayersOfInterestEncoder[[i]]) == TRUE){
#         ListLayerEncoder[[i]] = get_layer(model, index = LayersOfInterestEncoder[[i]])$output
#     } else {
#         ListLayerEncoder[[i]] = get_layer(model, name = LayersOfInterestEncoder[[i]])$output
#     }
# }

# ListLayerDecoder = list()
# for(i in 1:(length(LayersOfInterestDecoder))){
#     if(is.numeric(LayersOfInterestDecoder[[i]]) == TRUE){
#         ListLayerDecoder[[i]] = get_layer(model, index = LayersOfInterestDecoder[[i]])$output
#     } else {
#         ListLayerDecoder[[i]] = get_layer(model, name = LayersOfInterestDecoder[[i]])$output
#     }
# }

# encoder_model = keras_model(inputs = model$input, outputs = ListLayerEncoder)
# encoder_activations = predict(encoder_model, 
#                       X_test,
#                       batch_size = pred_batch_size)
# str(encoder_activations)
# for(t in 1:length(ListLayerEncoder)){
#     # tempIm = normalize(activations[[t]][1,,,1])
#     tempIm = resize(normalize(encoder_activations[[t]][1,,,12]), 512)
#     writeImage(tempIm, files = paste0("/home/gendarme/Desktop/Img_presentation/filters_encoder_", t,".png"))
# }

# decoder_model = keras_model(inputs = model$input, outputs = ListLayerDecoder)
# decoder_activations = predict(decoder_model, 
#                               X_test,
#                               batch_size = pred_batch_size)
# for(t in 1:length(ListLayerDecoder)){
#     # tempIm = normalize(activations[[t]][1,,,1])
#     tempIm = resize(normalize(decoder_activations[[t]][1,,,62]), 512)
#     writeImage(tempIm, files = paste0("/home/gendarme/Desktop/Img_presentation/filters_decoder_", t,".png"))
# }


# # bw_label = Y_val[Nimg,,]
# # bw_pred = Y_hat_val[Nimg,,,1]
# # color_label = colorLabels(bwlabel(Y_val[Nimg,,]))
# # hist(X_val[Nimg],,,1])
# # bright_img = combine_col(image_1 = EBImage::normalize(X_val[Nimg,,,1], inputRange = c(-10, 50)),
# # bright_img = combine_col(image_1 = EBImage::normalize(X_dapi_test[Nimg,,], inputRange = c(0.05, 0.12)),
# hist(X_test[Nimg,,,1])
# bright_img = combine_col(image_1 = EBImage::normalize(X_test[Nimg,,,1], inputRange = c(-100, 100)),
#                          color_1 = "grey",
#                          dimension = c(dim(X_val[Nimg,,,1])))
# bright_img = UnSharpMasking(bright_img)
# display(bright_img)
# # display(normalize(X_dapi_test[1,,]))
# # hist(X_dapi_test[Nimg,,])
# # dapi_img = combine_col(image_1 = EBImage::normalize(X_dapi_val[Nimg,,], inputRange = c(0.005, 0.07)),
# # dapi_img = combine_col(image_1 = EBImage::normalize(X_test[Nimg,,,1], inputRange = c(-100, -0)),
# hist(X_dapi_test[Nimg,,])
# dapi_img = combine_col(image_1 = EBImage::normalize(X_dapi_test[Nimg,,], inputRange = c(0.08, 0.15)),
#                        color_1 = "grey",
#                        dimension = c(dim(X_dapi_val[Nimg,,])))
# dapi_img = UnSharpMasking(dapi_img)
# display(dapi_img)
# # display(normalize(X_test[Nimg,,,1]))
# # hist(X_test[Nimg,,,1])
# # dapi_blue_img = rgbImage(blue = dapi_img * 2)
# # display(dapi_blue_img)
# # display(abind(
# #             #  bw_label,
# #               color_label,
# #               bright_img, along = 1))
# # viridis_pred = img_to_viridis(Y_hat_val[Nimg,,,1])
# # writeImage(bright_img, files = "/home/gendarme/Desktop/Img_presentation/bright_img.png")
# # writeImage(bright_img, files = "/home/gendarme/Desktop/Img_presentation/bright_img_test.png")
# # writeImage(bright_img, files = "/home/gendarme/Desktop/Img_presentation/bright_img_test_a549.png")
# # writeImage(dapi_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_img.png")
# # writeImage(dapi_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_img_test.png")
# # writeImage(dapi_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_img_test_a549.png")
# # writeImage(dapi_blue_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_blue_img.png")
# # writeImage(color_label, files = "/home/gendarme/Desktop/Img_presentation/color_label.png")
# # writeImage(bw_label, files = "/home/gendarme/Desktop/Img_presentation/bw_label.png")
# # writeImage(viridis_pred, files = "/home/gendarme/Desktop/Img_presentation/viridis_pred.png")
# # writeImage(bw_pred, files = "/home/gendarme/Desktop/Img_presentation/bw_pred.png")



