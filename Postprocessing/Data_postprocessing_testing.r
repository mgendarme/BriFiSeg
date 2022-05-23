library(keras)
use_implementation("keras")
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(magick)
library(rstudioapi)

# ## Settings
# # general parameters
# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
# # needs current directory to load the correct set of parameters
# CurDir = str_split(rstudioapi::getActiveDocumentContext()$path, "Data_", simplify = F)[[1]][1]

# # Load hyper parameters
# source(paste0(CurDir, "/Params.r"))
# # Load custom functions
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_PreprocessingAndTransformation.r"))
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_DataAugmentation_BFonly_Cropfix.r"))
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_Loss.r"))
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_LossFactory_Optimizers.r"))
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_Inspection.r"))
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_Postprocessing.r"))

# # set seed for reproducibility
# set.seed(11)

# # Parameters for saving files
# Current_i = as.character(paste0("Train_", MODEL,
#                                 "_", CLASS, "Class",
#                                 "_", IMAGE_SRC
#                                 ))
# Save_dir = paste0(Unet_dir,"/","Prediction/", Current_i, "--",1)

# if(dir.exists(Save_dir) == T){
#   list_dir = list.dirs(paste0(Unet_dir,"/","Prediction"), full.names = T, recursive = F)
#   my_dir = sub("--.*", "", Save_dir)
#   my_dirs = which(grepl(my_dir, list_dir))
#   Save_dir = paste0(sub("--.*", "", Save_dir), "--", (length(my_dirs)))
# } 

# ## Prepare training images and metadata
# all_sample_out <- map_df(list.files(TRAIN_PATH,
#                                     pattern = paste0(TypeOfObjects, "_noBorder_shapedisc_thickness5.csv"),
#                                     full.names = TRUE,
#                                     recursive = T),
#                          read_csv, col_types = cols())#,
# # "_noBorder_shapedisc_thickness5.csv" #"_noBorderDiamond11.csv"

# train_data <- all_sample_out %>%
#   mutate(ImageFile = file.path(paste0(Unet_dir, "/Image/", ImageId)),
#          ImageShape =  list(c(512, 512)),  #map(ImageFile, .f = function(file) dim(readImage(file))[1:2]),
#          Plate = 1,#as.numeric(str_sub(ImageId, 16, 18)),
#          ID = sapply(str_split(sapply(str_split(ImageId, "Well"), "[", 2), "_"), "[", 1),
#          Position = as.numeric(sapply(str_split(sapply(str_split(ImageId, "_"), "[", 3), "_Channel"), "[", 1)),
#          Channel = sapply(str_split(sapply(str_split(ImageId, "Channel"), "[", 2), "_"), "[", 1),
#          Crop = map(str_split(Crop, " "), as.numeric)
#   ) %>% select(-c("Channel"))
# # train_data$Crop

# image_data <- tibble(ImageId_BF = list.files(paste0(Unet_dir, "/Image"))) %>% 
#   mutate(ImageFile_BF = file.path(paste0(Unet_dir, "/Image/", ImageId_BF)),
#          Plate = 1,#as.numeric(str_sub(ImageId, 16, 18)),
#          ID = sapply(str_split(sapply(str_split(ImageId_BF, "Well"), "[", 2), "_"), "[", 1),
#          Position = as.numeric(sapply(str_split(sapply(str_split(ImageId_BF, "_"), "[", 3), "_Channel"), "[", 1)),
#          Channel = sapply(str_split(sapply(str_split(ImageId_BF, "Channel"), "[", 2), "_"), "[", 1), 
#          ZStack = as.numeric(sapply(str_split(sapply(str_split(ImageId_BF, "ZStack"), "[", 2), "_"), "[", 1)))

# if(IMAGE_SRC == "BF"){
#   image_data = filter(image_data, ZStack == 0 & Channel == "DIA") #%>% select(-c("Channel"))  
# } else {
#   image_data = filter(image_data, Channel == "DAPI") #%>% select(-c("Channel"))    
# }
  
# # display(readImage(image_data$ImageFile_BF[1]))

# # Filter A1 & B1 (very high cell density)
# WELLS = c("B02", "B03", "B04", "B10")#,
#           # "B05", "B06", "B09", "B11",
#           # "C02", "C03", "C04", "C06",
#           # ### more data to double the training set
#           # 'B07', 'B08', 'B13', 'B14',
#           # 'B16', 'B21', 'B22', 'B23',
#           # 'C07', 'C08', 'C09', 'C10')

# train_data <- train_data %>%
#   filter(ID %in% WELLS) %>%
#   left_join(image_data, by = c("Plate", "ID", "Position")) %>%
#   glimpse()
# unique(train_data$ID)

# # generate input data:
# input <- train_data %>% #sample_n(train_data, nrow(train_data)) %>%
#   mutate(Y = map(MaskEncodedPixels, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
#          B = map(BorderEncodedPixels, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
#          I = map(InterfaceEncodedPixels, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
#          # X = map2(ImageFile_BF, Crop, ~ preprocess_image_TL(.x, shape = c(WIDTH, HEIGHT), cropping = HEIGHT, .y)),
#          X = map2(ImageFile_BF, Crop, ~ preprocess_image_TL(.x, shape = c(1000, 1000), cropping = .y)),
#          Img_Src = IMAGE_SRC,
#         #  X = map_if(X, Img_Src == "BF", UnSharpMasking, UnSharpMasking_Power),
#          X = map(X, normalize),
#          X = map_if(X, Img_Src == "BF", filter2, sharp),
#         #  X = map_if(X, Img_Src == "DIA", filter2, outline),
#          X = map_if(X, Img_Src == "FLUO", normalize),
#          Mod = MODEL,
#          X = map_if(X, Mod != "unet", transform_gray_to_rgb_rep),
#          B = map(B, ~ ifelse( .x == 1, 2, .x)),
#          I = map(I, ~ ifelse( .x == 1, 3, .x)),
#          Y = map2(Y, B, ~ .x + .y),
#          Y = map2(Y, I, ~ .x + .y),
#          Y = map(Y, ~ to_categorical( .x, 4))) %>%
#   select(X, Y) %>%
#   mutate(Activation = ACTIVATION,
#          Y = map_if(Y, Activation == "softmax" & CLASS == 3, softmax_transf_channel, 3),
#          Y = map_if(Y, Activation == "softmax" & CLASS == 2, softmax_transf_channel, 2),
#          # Y = map(Y, resize_masks, shape = SHAPE_MASK),
#          Y = map(Y, sum_channels, 3, 3, 4),
#          Y = map(Y, diff_channels, 2, 2, 4),
#          Y = map(Y, ~ ifelse( .x > 0, 1, 0)),
#          Class = CLASS,
#          DIM3 = 1,
#          Y = map_if(Y, Class == 1, ~ select_channels(.x, 2, 2)),
#          Y = map_if(Y, Class == 1, ~ add_dim(.x, 1)),
#          Y = map_if(Y, Class == 2, ~ keep_channels(.x, 2, ifelse(SND_CLASS == "border", 3, 4))),
#          Y = map_if(Y, Class == 2, ~ select_channels(.x , 1, 2)),
#          Y = map_if(Y, Class == 3, ~ select_channels(.x , 2, 4))
#   ) %>% 
#   select(X, Y)

# display(input$X[[1]])

# ## Randomise samples and split into train and test sets:
# TRAIN_INDEX <- sample(1:nrow(input), as.integer(round(nrow(input) * (1 - VALIDATION_SPLIT), 0)), replace = F)

# sampling_generator <- function(data, train_index = TRAIN_INDEX) {
#   train_input <<- data[train_index,]
#   val_input <<- data[-train_index,]
# }
# sampling_generator(input)

# ## Load model
# model <- load_model_hdf5(filepath = paste0(Save_dir, "/unet_model_",Current_i,"_",".hdf5"), compile = F) # need to specify the custom metric to load
# source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_UnetInceptionResnetv2_Model.r"))
# model = inception_resnet_v2_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
#                                   output_channels = CLASS,
#                                   output_activation = ACTIVATION)
# model %>% load_model_weights_hdf5(model, filepath = paste0(str_replace(Save_dir, "/Image", ""),
#    "/unet_model_weights_",Current_i,"_",".hdf5"))
# summary(model)

n_samples = 20

X_train <- list2tensor(train_input$X)
Y_train <- list2tensor(train_input$Y)

X_val <- list2tensor(val_input$X)
Y_val <- list2tensor(val_input$Y)

# X = X[,1:HEIGHT, 1:WIDTH,]
# if(CHANNELS == 1){dim(X) = c(dim(X), 1)}
X_train = X_train[1:n_samples, 1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_train) = c(dim(X_train), 1)}
X_val = X_val[1:n_samples, 1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_val) = c(dim(X_val), 1)}

# Y = Y[,1:HEIGHT, 1:WIDTH,]
Y_train = Y_train[1:n_samples, 1:HEIGHT, 1:WIDTH,]
Y_val = Y_val[1:n_samples, 1:HEIGHT, 1:WIDTH,]

for (i in c("X_train", "Y_train")) {
  print(paste0("Str: ", i, " ", as.character(str(get(i)))))
}

# test = predict(model, x = X_train)
# display(test[1,,,3], all = T)
# display(normalize(X_train[1,,,]), all = T)
## Evaluate the fitted model
## Predict and evaluate on training images:
Y_hat <- predict(model, x = X_train)
Y_hat_val <- predict(model, x = X_val)

#######################################################################################################################
# Explore the model and its prediction #################################################################################
#######################################################################################################################
Save_dir = paste0(Save_dir, "/Image")
dir.create(Save_dir)

# saving some representative images with transformation
for(i in 1:10){
  # Save_dir = paste0("home/gendarme/", str_replace(Save_dir, "~", "")) # if relative path, because of a bug in writeImage (tiff package)
  # IR = c(0.513, 0.520)
  # i = 1
  IR = range(X_train[i,,,])
  # hist(X[1,,,])
  merge_Y <- make_merge(Y_train, image_number = i)
  # display(merge_Y)
  writeImage(merge_Y, files = paste0(Save_dir, "/", "merge_Y_",i,".tiff"), quality = 100, type = "tiff")
  
  merge_Yhat <- make_merge(Y_hat, image_number = i)
  # display(merge_Yhat)
  writeImage(merge_Yhat, files = paste0(Save_dir, "/", "merge_Yhat_",i,".tif"),
             quality = 100, type = "tiff")
  
  merge_Y_mask <- normalize(toRGB(X_train[i,,,1])) 
  # if(MODEL != "unet"){
  #   merge_Y_mask = merge_Y_mask[,,1,]
  # }
  # display(merge_Y_mask)
  merge_Y_mask <- merge_Y_mask + rgbImage(red = Y_train[i,,,2])
  #  display(merge_Y_mask)
  #make_merge_image_mask(X = X, Y = Y, image_number = i, InputRange = IR)
  # display(normalize(X[i,,,], inputRange = c(0.007, 0.010)))
  # range(X[i,,,])
  # str(X)
  if(MODEL != "unet"){
    col_merge_nuc_pred <- combine_col(image_1 = EBImage::normalize(X_train[i,,,1], inputRange = IR),
                                      image_2 = Y_hat[i,,,1],
                                      color_1 = "grey",
                                      color_2 = "blue",
                                      dimension = c(dim(X_train[i,,,1])[1:2])
                                      )
  } else {
    col_merge_nuc_pred <- combine_col(image_1 = EBImage::normalize(X_train[i,,,], inputRange = IR),
                                      image_2 = Y_hat[i,,,1],
                                      color_1 = "grey",
                                      color_2 = "blue",
                                      dimension = c(dim(X_train[i,,,])[1:2]))
    
  }
  # display(col_merge_nuc_pred)
  writeImage(col_merge_nuc_pred,
             files = paste0(Save_dir, "/", "col_merge_nuc_gt_",i,".tif"),
             quality = 100, type = "tiff")
  
  if(MODEL != "unet"){
    col_merge_nuc_gt <- combine_col(image_1 = normalize(X_train[i,,,1], inputRange = IR),
                                    image_2 = Y_train[i,,,1],
                                    color_1 = "grey",
                                    color_2 = "blue",
                                    dimension = c(dim(X_train[i,,,1])[1:2]))
  } else {
    col_merge_nuc_gt <- combine_col(image_1 = normalize(X_train[i,,,], inputRange = IR),
                                    image_2 = Y_train[i,,,1],
                                    color_1 = "grey",
                                    color_2 = "blue",
                                    dimension = c(dim(X_train[i,,,])[1:2]))
    
  }
  # display(col_merge_nuc_gt)
  writeImage(col_merge_nuc_gt,
             files = paste0(Save_dir, "/", "col_merge_nuc_gt_",i,".tif"),
             quality = 100, type = "tiff")
  
  if(MODEL != "unet"){
    col_merge_border_pred <- combine_col(image_1 = normalize(X_train[i,,,1], inputRange = IR),
                                       image_2 = Y_hat[i,,,2],
                                       image_3 = Y_hat[i,,,3],
                                       color_1 = "grey",
                                       color_2 = "green",
                                       color_3 = "red",
                                       dimension = c(dim(X_train[i,,,1])[1:2]))
  } else {
    col_merge_border_pred <- combine_col(image_1 = normalize(Xv[i,,,], inputRange = IR),
                                         image_2 = Y_hat[i,,,2],
                                         image_3 = Y_hat[i,,,3],
                                         color_1 = "grey",
                                         color_2 = "green",
                                         color_3 = "red",
                                         dimension = c(dim(X_train[i,,,])[1:2]))
  }
   # display(col_merge_border_pred)
  writeImage(col_merge_border_pred,
             files = paste0(Save_dir, "/", "col_merge_border_pred_",i,".tif"),
             quality = 100, type = "tiff")
  if(MODEL != "unet"){
    col_merge_border_gt <- combine_col(image_1 = normalize(X_train[i,,,1], inputRange = IR),
                                     image_2 = Y_train[i,,,2],
                                     image_3 = Y_train[i,,,3],
                                     color_1 = "grey",
                                     color_2 = "green",
                                     color_3 = "red",
                                     dimension = c(dim(X_train[i,,,1])[1:2]))
  } else {
    col_merge_border_gt <- combine_col(image_1 = normalize(X_train[i,,,], inputRange = IR),
                                       image_2 = Y_train[i,,,2],
                                       image_3 = Y_train[i,,,3],
                                       color_1 = "grey",
                                       color_2 = "green",
                                       color_3 = "red",
                                       dimension = c(dim(X_train[i,,,])[1:2]))
  }
  # display(col_merge_border_gt)
  writeImage(col_merge_border_gt,
             files = paste0(Save_dir, "/", "col_merge_border_gt_",i,".tif"),
             quality = 100, type = "tiff")
  
  # col_merge_mask <- paintObjects(Y[i,,,], col_merge_nuc, col = "red")
  # 
  # writeImage(col_merge_mask,
  #            files = paste0(Save_dir, "/", "col_merge_mask",i,".tif"),
  #            quality = 100, type = "tiff")
  # 
  # merge_Yhat_mask <- make_merge_image_mask(X = normalize(X), Y = Y_hat, image_number = i, InputRange = IR)
  # display(merge_Yhat_mask)
  # writeImage(abind(merge_Y_mask, merge_Yhat_mask, along = 1),
  #            files = paste0(Save_dir, "/", "merge_Y_Yhat_mask",i,".tif"),
  #            quality = 100, type = "tiff")
  # 
  merge_Y_Yhat <- abind(make_merge(Y_train, image_number = i),
                        make_merge(Y_hat, image_number = i),
                        along = 1)
    # display(merge_Y_Yhat)
  writeImage(merge_Y_Yhat, files = paste0(Save_dir, "/", "merge_Y_Yhat_",i,".tif"),
             quality = 100, type = "tiff")
  
  if(MODEL != "unet"){
    montage_XY_XYhat <- abind(make_montage(array(X_train[,,,1], dim = c(dim(X_train[,,,1]), 1)), Y_train, image_number = i, InputRange = NULL),
                              make_montage(array(X_train[,,,1], dim = c(dim(X_train[,,,1]), 1)), Y_hat, image_number = i, InputRange = NULL),
                              along = 2)
    # display(montage_XY_XYhat)
  } else {
    montage_XY_XYhat <- abind(make_montage(X_train, Y_train, image_number = i, InputRange = NULL),
                              make_montage(X_train, Y_hat, image_number = i, InputRange = NULL),
                              along = 2)
  }
  # display(montage_XY_XYhat)
  writeImage(montage_XY_XYhat, files = paste0(Save_dir, "/", "montage_X_Y_XYhat_",i,".tif"),
             quality = 100, type = "tiff")
  
  pred_gt_3col = abind(paintObjects(Y_train[i,,,1], rgbImage(blue = Y_train[i,,,1],
                                                       green = Y_train[i,,,2],
                                                       red = Y_train[i,,,3])),
                       paintObjects((Y_hat[i,,,1] - Y_hat[i,,,2] - Y_hat[i,,,3]) > .5, rgbImage(blue = Y_hat[i,,,1],
                                                                                                red = Y_hat[i,,,3],
                                                                                                green = Y_hat[i,,,2]), col = "red"),
                       along = 1)
  writeImage(pred_gt_3col, files = paste0(Save_dir, "/", "pred_gt_3col_",i,".tif"),
             quality = 100, type = "tiff")
    
  pred_gt_3col_oneimg = paintObjects(Y_train[i,,,2],
                                   rgbImage(blue = Y_hat[i,,,1], red = Y_hat[i,,,3], green = Y_hat[i,,,2]),
                                   col = "red")
   # display(pred_gt_3col_1img)
  writeImage(pred_gt_3col_oneimg, files = paste0(Save_dir, "/", "pred_gt_3col_oneimg_",i,".tif"),
             quality = 100, type = "tiff")

}

# Post-processing step
for(i in 1:10){
  if(MODEL != "unet"){
    simple_postprocessing(array(X_train[,,,1], dim = c(dim(X_train[,,,1]), 1)), Y_train, Y_hat, thresh = 0.3, size = 100)
  } else {
    simple_postprocessing(X_train, Y_train, Y_hat, thresh = 0.3, size = 100)
  }
}

# modify shape with pretrained encoder
if(MODEL != "unet"){
  X_val = array(X_val[,,,1], dim = c(dim(X_val[,,,1]), 1))
}

# Some overlays
for(i in 1:10){
  # Save_dir = paste0("home/gendarme/", str_replace(Save_dir, "~", "")) # if relative path, because of a bug in writeImage (tiff package)
  # IR = c(0.513, 0.520)
  IR = range(X_val[i,,,])
  # hist(X_val[1,,,])
  merge_Y <- make_merge(Y_val, image_number = i)
  # display(merge_Y)
  writeImage(merge_Y, files = paste0(Save_dir, "/", "merge_Y_val_",i,".tiff"), quality = 100, type = "tiff")
  
  merge_Yhat <- make_merge(Y_hat_val, image_number = i)
  # display(merge_Yhat)
  writeImage(merge_Yhat, files = paste0(Save_dir, "/", "merge_Yhat_",i,".tif"),
             quality = 100, type = "tiff")
  
  merge_Y_mask <- normalize(toRGB(X_val[i,,,])) 
  merge_Y_mask <- merge_Y_mask + rgbImage(red = Y_val[i,,,2])
  #make_merge_image_mask(X = X, Y = Y, image_number = i, InputRange = IR)
  # display(merge_Y_mask)
  # display(normalize(X[i,,,], inputRange = c(0.007, 0.010)))
  # range(X[i,,,])
  # str(X)
  col_merge_nuc_pred <- combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                                    image_2 = Y_hat_val[i,,,1],
                                    color_1 = "grey",
                                    color_2 = "blue",
                                    dimension = c(dim(X_val[i,,,])[1:2]))
  # display(col_merge_nuc_pred)
  writeImage(col_merge_nuc_pred,
             files = paste0(Save_dir, "/", "col_merge_nuc_gt_",i,".tif"),
             quality = 100, type = "tiff")
  col_merge_nuc_gt <- combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                                  image_2 = Y_val[i,,,1],
                                  color_1 = "grey",
                                  color_2 = "blue",
                                  dimension = c(dim(X_val[i,,,])[1:2]))
   # display(col_merge_nuc_gt)
  writeImage(col_merge_nuc_gt,
             files = paste0(Save_dir, "/", "col_merge_nuc_gt_",i,".tif"),
             quality = 100, type = "tiff")
  
  if(CLASS == 3){
    col_merge_border_pred <- combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                                         image_2 = Y_hat_val[i,,,2],
                                         image_3 = Y_hat_val[i,,,3],
                                         color_1 = "grey",
                                         color_2 = "green",
                                         color_3 = "red",
                                         dimension = c(dim(X_val[i,,,])[1:2]))
  } else if(CLASS == 2){
    col_merge_border_pred <- combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                                         image_2 = Y_hat_val[i,,,2],
                                         # image_3 = Y_hat_val[i,,,3],
                                         color_1 = "grey",
                                         color_2 = "green",
                                         # color_3 = "red",
                                         dimension = c(dim(X_val[i,,,])[1:2]))
  }
  # display(col_merge_border_pred)
  writeImage(col_merge_border_pred,
             files = paste0(Save_dir, "/", "col_merge_border_pred_",i,".tif"),
             quality = 100, type = "tiff")
  if(CLASS == 3){
    col_merge_border_gt <- combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                                      image_2 = Y_val[i,,,2],
                                      image_3 = Y_val[i,,,3],
                                      color_1 = "grey",
                                      color_2 = "green",
                                      color_3 = "red",
                                      dimension = c(dim(X_val[i,,,])[1:2]))
  } else if(CLASS == 2){
    col_merge_border_gt <- combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                                       image_2 = Y_val[i,,,2],
                                       # image_3 = Y_val[i,,,3],
                                       color_1 = "grey",
                                       color_2 = "green",
                                       # color_3 = "red",
                                       dimension = c(dim(X_val[i,,,])[1:2]))
  }
   # display(col_merge_border_gt)
  writeImage(col_merge_border_gt,
             files = paste0(Save_dir, "/", "col_merge_border_gt_",i,".tif"),
             quality = 100, type = "tiff")
  
  # col_merge_mask <- paintObjects(Y[i,,,], col_merge_nuc, col = "red")
  # 
  # writeImage(col_merge_mask,
  #            files = paste0(Save_dir, "/", "col_merge_mask",i,".tif"),
  #            quality = 100, type = "tiff")
  # 
  # merge_Yhat_mask <- make_merge_image_mask(X = normalize(X), Y = Y_hat_val, image_number = i, InputRange = IR)
  # display(merge_Yhat_mask)
  # writeImage(abind(merge_Y_mask, merge_Yhat_mask, along = 1),
  #            files = paste0(Save_dir, "/", "merge_Y_Yhat_mask",i,".tif"),
  #            quality = 100, type = "tiff")
  # 
  merge_Y_Yhat <- abind(make_merge(Y_val, image_number = i),
                        make_merge(Y_hat_val, image_number = i),
                        along = 1)
   # display(merge_Y_Yhat)
  writeImage(merge_Y_Yhat, files = paste0(Save_dir, "/", "merge_Y_val_Yhat_",i,".tif"),
             quality = 100, type = "tiff")
  
  montage_X_valY_X_valYhat <- abind(make_montage(X_val, Y_val, image_number = i, InputRange = NULL),
                                    make_montage(X_val, Y_hat_val, image_number = i, InputRange = NULL),
                                    along = 2)
  # display(montage_X_valY_X_valYhat)
  writeImage(montage_X_valY_X_valYhat, files = paste0(Save_dir, "/", "montage_X_val_Y_X_valYhat_",i,".tif"),
             quality = 100, type = "tiff")

  img_gt = combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                         image_2 = Y_val[i,,,2],
                         image_3 = Y_val[i,,,3],
                         color_1 = "grey",
                         color_2 = "green",
                         color_3 = "red",
                         dimension = c(dim(X_val[i,,,])[1:2]))
  writeImage(img_gt, files = paste0(Save_dir, "/", "img_gt_",i,".tif"),
               quality = 100, type = "tiff")

  img_pred = combine_col(image_1 = normalize(X_val[i,,,], inputRange = IR),
                         image_2 = Y_hat_val[i,,,2],
                         image_3 = Y_hat_val[i,,,3],
                         color_1 = "grey",
                         color_2 = "green",
                         color_3 = "red",
                         dimension = c(dim(X_val[i,,,])[1:2]))
  writeImage(img_pred, files = paste0(Save_dir, "/", "img_pred_",i,".tif"),
               quality = 100, type = "tiff")

  img_gt_pred = abind(img_gt, img_pred, along = 1)
  writeImage(img_gt_pred, files = paste0(Save_dir, "/", "img_gt_pred_",i,".tif"),
               quality = 100, type = "tiff")
  # display(img_gt_pred)

}

# Intersection over Union and Mean Precision
# These functions require labeled images as input!

##############
### IOU #2 ###
##############

iou_thresh <- c(seq(0.01, 0.99, 0.01))

iou <- function(y_true, y_pred){
   # y_true = Y_train[,,,1]
   # y_pred = Y_hat[,,,1]
  
  intersection <- sum((y_true * y_pred)>0)
  union <- sum((y_true + y_pred)>0)
  
  if(union == 0){
    return(union)
  }
  
  return(intersection/union)
}

iou_metric <- function(y_true, y_pred){
  if(!is.list(y_true)){
    num_imgs <- dim(y_true)[1]
  }
  
  scores <- array(0, num_imgs)
  #scores <- c()
  
  for(i in 1:num_imgs){
    
    y_true_i = array(y_true[i,,], dim = dim(y_true)[2]*dim(y_true)[3])
    y_pred_i = array(y_pred[i,,], dim = dim(y_pred)[2]*dim(y_pred)[3])
    
    if(sum(y_true[i,,]) == 0 & sum(y_pred[i,,]) == 0){
      scores[i] = 1
    } else {
      scores[i] = mean(iou_thresh <= iou(y_true[i,,], y_pred[i,,]))
    }
  }
  
  return(scores)
}

Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= .5, Y_hat[,,,1], 0) -
  ifelse(Y_hat[,,,2] >= .5, Y_hat[,,,2], 0) -
  ifelse(Y_hat[,,,3] >= .5, Y_hat[,,,3], 0)
Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)

Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= .5, Y_hat_val[,,,1], 0) -
  ifelse(Y_hat_val[,,,2] >= .5, Y_hat_val[,,,2], 0) -
  ifelse(Y_hat_val[,,,3] >= .5, Y_hat_val[,,,3], 0)
Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)

iou_train = iou_metric(Y_train[,,,1], Y_hat[,,,1])
iou_train_post_proc = iou_metric(Y_train[,,,1], Y_hat_train_ab_p5)
iou_val = c(iou_metric(Y_val[,,,1], Y_hat_val[,,,1]), 
            rep(NA, (dim(Y_train)[1] - dim(Y_val)[1])))
iou_val_post_proc = c(iou_metric(Y_val[,,,1], Y_hat_val_ab_p5),
                      rep(NA, (dim(Y_train)[1] - dim(Y_val)[1])))

# ious = tibble(Dataset = c(rep(as.integer(1), dim(Y_train)[1]*2)),
#                           # rep("Validation", dim(Y_train)[1]*2)),
#               IOU = rnd#c(iou_train, iou_train_post_proc)#,# iou_val, iou_val_post_proc),
#               # Method = c(rep("Raw", dim(Y_train)[1]*2))#,
#                          # rep("Post-processed", dim(Y_train)[1]*2))
#               )
# 
# a = ggplot(ious, aes(x = Dataset, y = IOU)) %>% #, fill = Method)) %>% 
#   boxplot()

## val
png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_Boxplot_train.png"), width = 800, height = 1200, res = 300)
boxplot(iou_train, 
        main = c(paste0("Jaccard index \nmedian = ", round(median(iou_train, na.rm = T), 2))),
        xlab = "Train post-proc",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_Boxplot_train_postproc.png"), width = 800, height = 1200, res = 300)
boxplot(iou_train_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", round(median(iou_train_post_proc, na.rm = T), 2))),
        xlab = "Train",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

## val
png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_Boxplot_val.png"), width = 800, height = 1200, res = 300)
boxplot(iou_val, 
        main = c(paste0("Jaccard index \nmedian = ", round(median(iou_val, na.rm = T), 2))),
        xlab = "Validation",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_Boxplot_val_postproc.png"), width = 800, height = 1200, res = 300)
boxplot(iou_val_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", round(median(iou_val_post_proc, na.rm = T), 2))),
        xlab = "Validation post-proc.",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

## for some DAPI overlay
## Prepare training images and metadata

train_dapi_data <- all_sample_out %>%
  mutate(ImageFile = file.path(paste0(Unet_dir, "/Image/", ImageId)),
         ImageShape =  list(c(512, 512)),  #map(ImageFile, .f = function(file) dim(readImage(file))[1:2]),
         Plate = 1,#as.numeric(str_sub(ImageId, 16, 18)),
         ID = sapply(str_split(sapply(str_split(ImageId, "Well"), "[", 2), "_"), "[", 1),
         Position = as.numeric(sapply(str_split(sapply(str_split(ImageId, "_"), "[", 3), "_Channel"), "[", 1)),
         Channel = sapply(str_split(sapply(str_split(ImageId, "Channel"), "[", 2), "_"), "[", 1),
         Crop = map(str_split(Crop, " "), as.numeric)
  ) %>% select(-c("Channel"))
# train_data$Crop

image_dapi_data <- tibble(ImageId = list.files(paste0(Unet_dir, "/Image"))) %>% 
  mutate(ImageFile = file.path(paste0(Unet_dir, "/Image/", ImageId)),
         Plate = 1,#as.numeric(str_sub(ImageId, 16, 18)),
         ID = sapply(str_split(sapply(str_split(ImageId, "Well"), "[", 2), "_"), "[", 1),
         Position = as.numeric(sapply(str_split(sapply(str_split(ImageId, "_"), "[", 3), "_Channel"), "[", 1)),
         Channel = sapply(str_split(sapply(str_split(ImageId, "Channel"), "[", 2), "_"), "[", 1), 
         ZStack = as.numeric(sapply(str_split(sapply(str_split(ImageId, "ZStack"), "[", 2), "_"), "[", 1))) %>% 
  filter(Channel == "DAPI") %>% 
  select(-c("Channel"))

# Filter A1 & B1 (very high cell density)
train_dapi_data <- train_dapi_data %>% 
  filter(ID %in% WELLS) %>%
  left_join(image_data, by = c("Plate", "ID", "Position")) %>% 
  glimpse()
unique(train_data$ID)

# generate input data:
input_dapi <- train_dapi_data %>% #sample_n(train_data, nrow(train_data)) %>%
  mutate(X = map2(ImageFile, Crop, ~ preprocess_image_TL(.x, shape = c(1000, 1000), cropping = .y))) %>% 
  select(X)
# str(input, list.len = 2)

display(abind(normalize(input_dapi$X[[1]][,,1])[1:HEIGHT, 1:WIDTH],
              normalize(input$X[[1]][,,1])[1:HEIGHT, 1:WIDTH],
              input$Y[[1]][,,2][1:HEIGHT, 1:WIDTH],
              along = 1))

## Randomise samples and split into train and test sets:
sampling_generator <- function(data, train_index = TRAIN_INDEX) {
  train_dapi_input <<- data[train_index,]
  val_dapi_input <<- data[-train_index,]
}
sampling_generator(input_dapi)

X_dapi_train <- list2tensor(train_dapi_input$X)
X_dapi_val <- list2tensor(val_dapi_input$X)

X_dapi_train = X_dapi_train[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_dapi_train) = c(dim(X_dapi_train), 1)}
X_dapi_val = X_dapi_val[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_dapi_val) = c(dim(X_dapi_val), 1)}

simple_postprocessing(array(X_train[,,,1], dim = c(dim(X_train[,,,1]), 1)), Y_train, Y_hat, thresh = 0.3, size = 100)
prob_pp = function(y_hat, ind, thresh, size, wts = T){
  # y_hat = Y_hat[1:10,,,]
  # tresh = .5
  # size = 100
  # ind = 1
  # wts = T
  y_hat[ind,,,1] <- y_hat[ind,,,1] > thresh
  y_hat[ind,,,2] <- y_hat[ind,,,2] > thresh
  y_hat[ind,,,3] <- y_hat[ind,,,3] > thresh
  objectpred <- y_hat[ind,,,1] - y_hat[ind,,,2] - y_hat[ind,,,3]
  objectpred <- ifelse(objectpred < 0, 0, objectpred)
  objectpred_mask <- bwlabel(objectpred)
  objectpred_op <- opening(objectpred_mask, kern = makeBrush(size, shape = "disc"))
  if(wts == T){
    wt = watershed(distmap(objectpred_op), tolerance = 1, ext = 1)
    wt
  } else {
    objectpred_op
  }
}

Y_hat_pp = array(0, dim = dim(Y_hat)[1:3])
for (i in 1:10){#:dim(Y_hat)[1]) {
  Y_hat_pp[i,,] = prob_pp(Y_hat, i, .5, 25)
}  

Y_hat_val_pp = array(0, dim = dim(Y_hat)[1:3])
for (i in 1:10){#dim(Y_hat_val)[1]) {
  Y_hat_val_pp[i,,] = prob_pp(Y_hat_val, i, .5, 25)
}

for(i in 1:10){
  dapi_gt_predpp =  paintObjects(Y_train[i,,,1],
                                 paintObjects(Y_hat_pp[i,,],
                                              combine_col(image_1 = normalize(X_dapi_train[i,,]),
                                                          color_1 = "grey",
                                                          dimension = c(dim(X_dapi_train[i,,])[1:2])),
                                              col = "red", thick = T),
                                 col = "green", thick = T)
  writeImage(dapi_gt_predpp, files = paste0(Save_dir, "/", "dapi_gt_predpp_",i,".tiff"), quality = 100, type = "tiff")
  
  BF_gt_predpp =  paintObjects(Y_train[i,,,1],
                               paintObjects(Y_hat_pp[i,,],
                                            combine_col(image_1 = normalize(X_train[i,,,1]),
                                                        color_1 = "grey",
                                                        dimension = c(dim(X_val[i,,,])[1:2])),
                                            col = "red", thick = T),
                               col = "green", thick = T)
  writeImage(BF_gt_predpp, files = paste0(Save_dir, "/", "BF_gt_predpp_",i,".tiff"), quality = 100, type = "tiff")
  
  BF_DAPI_gt_predpp = abind(dapi_gt_predpp, BF_gt_predpp, along = 1)
  writeImage(BF_DAPI_gt_predpp, files = paste0(Save_dir, "/", "BF_DAPI_gt_predpp_",i,".tiff"), quality = 100, type = "tiff")
}

## compute iou after postprocessing
# prob_pp4iou = function(y_hat, ind, thresh, size, wts = T){
#   # y_hat = Y_hat[1:10,,,]
#   # tresh = .5
#   # size = 100
#   # ind = 1
#   # wts = T
#   y_hat[ind,,,1] <- y_hat[ind,,,1] > thresh
#   y_hat[ind,,,2] <- y_hat[ind,,,2] > thresh
#   y_hat[ind,,,3] <- y_hat[ind,,,3] > thresh
#   objectpred <- y_hat[ind,,,1] - y_hat[ind,,,2] - y_hat[ind,,,3]
#   objectpred <- ifelse(objectpred < 0, 0, objectpred)
#   objectpred_mask <- bwlabel(objectpred)
#   objectpred_op <- opening(objectpred_mask, kern = makeBrush(size, shape = "disc"))
#   objectpred <- ifelse(objectpred_op = 0, 0, objectpred_op)
# }

# Y_hat_pp4iou = array(0, dim = dim(Y_hat)[1:3])
# for (i in 1:dim(Y_hat)[1]) {
#   Y_hat_pp4iou[i,,] = prob_pp(Y_hat, i, .5, 25)
# } 
# Y_hat_val_pp4iou = array(0, dim = dim(Y_hat_val)[1:3])
# for (i in 1:dim(Y_hat_val)[1]) {
#   Y_hat_val_pp4iou[i,,] = prob_pp(Y_hat_val, i, .5, 25)
# } 

# iou_train_pp = iou_metric(Y_train[,,,1], Y_hat_pp4iou[,,])
# iou_val_pp = iou_metric(Y_val[,,,1], Y_hat_val_pp4iou[,,])

# png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_Boxplot_train_pp.png"), width = 800, height = 1200, res = 300)
# boxplot(iou_train_pp, 
#         main = c(paste0("Jaccard index \nmedian = ", round(median(iou_train_pp, na.rm = T), 2))),
#         xlab = "Train post-proc",
#         ylab = "Intersection over union per image",
#         ylim = c(0, 1))
# dev.off()
# # hist(iou_train_pp)
# png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_Boxplot_val_pp.png"), width = 800, height = 1200, res = 300)
# boxplot(iou_val_pp, 
#         main = c(paste0("Jaccard index \nmedian = ", round(median(iou_val_pp, na.rm = T), 2))),
#         xlab = "Train post-proc",
#         ylab = "Intersection over union per image",
#         ylim = c(0, 1))
# dev.off()
# # hist(iou_train_pp)

