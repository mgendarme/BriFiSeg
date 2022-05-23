library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)
library(tensorflow)

# set to run tensorflow on cpu if GPU is busy
tf$config$set_visible_devices(list(), 'GPU')

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
TASK = 2
source(paste0(RelPath, "/Scripts/FunctionCompilation/PreprocessingAndTransformation.r"))

model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--8/fpn--seresnext101--Epochs_200--Minibatches_100--FOLD_1")
# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--5/fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_1")
# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_1Class--36/fpn--seresnext101--FOLD_1")
image_folder = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/imagesTr")
label_folder = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/labelsTr")
image_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/imagesTs")
label_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/labelsTs")

arraynum = function(image){
  return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

# Load custom functions
list_scripts = list("Params_nnUNet_comparaison_cca.r", "PreprocessingAndTransformation.r", "Load_data_from_disk.r",
                    "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus.r",
                    "Loss.r",
                    "Backbone_Zoo.r",
                    "Inspection.r",
                    "Postprocessing.r") %>%
  map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
  source(list_scripts[[l]])
}

Save_dir = paste0(model_folder, "/NewPlot")
dir.create(Save_dir)

weight_name = paste0(
  # "best_",
  "model_weights.hdf5")

# build the model
enc = ENCODER 
redfct = 8L
arc = ARCHITECTURE
dropout = DROPOUT
source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
model = build_fpn(input_shape = c(HEIGHT, WIDTH, 3),
                  backbone = "seresnext101",
                  nlevels = NULL,
                  output_activation = ACTIVATION,
                  output_channels = 3,
                  decoder_skip = FALSE,
                  dropout = DROPOUT
)
# if model saved
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
model %>% load_model_weights_hdf5(filepath = paste0(model_folder, "/", weight_name))

# get index of validation data
json_param = jsonlite::read_json(paste0(model_folder, "/detail_param_fold_1.json"), simplifyVector = TRUE)
train_index = as.numeric(json_param$Train_index)
val_index = as.numeric(json_param$Val_index)

# load data
# image_data = tibble()
# image_data_test = tibble()

image_data = load_data(image_folder, label_folder)
image_data_test = load_data(image_folder_test, label_folder_test)

# # for(i in c(1)){
# image_data <- tibble(Image_Path = list.files(paste0(image_folder), recursive = T),
#                           Label_Path = list.files(paste0(label_folder), recursive = T)) %>%
#   filter(str_detect(Image_Path, "nii")) %>%
#   mutate(Image_Path = paste0(image_folder, "/", Image_Path),
#          Label_Path = paste0(label_folder, "/", Label_Path),
#          X = map(Image_Path, ~ readNIfTI(fname = .x)),
#          X = map(X, ~ arraynum(.x)),
#          X = map(X, ~ add_dim(.x, 1)),
#          X = map(X, transform_gray_to_rgb_rep),
#          X = map(X, EBImage::normalize, separate = T, ft = c(0, 255)),
#          X = map(X, imagenet_preprocess_input, mode = "torch"),
#          Y = map(Label_Path, ~ readNIfTI(fname = .x)),
#          Y = map(Y, ~ arraynum(.x)),
#          Y = map(Y, to_categorical),
#          Class = CLASS,
#          Y = map_if(Y, Class == 1, ~ add_dim(.x, 1)),
#          ID = as.numeric(sapply(str_split(sapply(str_split(Image_Path, "A549_"), "[", 2), "_0000"), "[", 1))
#          # Y_hat = map(Y_hat, flip),
#          # Y_hat = map(Y_hat, ~ add_dim(.x, 1))
#   )

# image_data_test = tibble(Image_Path = list.files(paste0(image_folder_test), recursive = T),
#                          Label_Path = list.files(paste0(label_folder_test), recursive = T)) %>%
#   filter(str_detect(Image_Path, "nii")) %>%
#   mutate(Image_Path = paste0(image_folder_test, "/", Image_Path),
#          Label_Path = paste0(label_folder_test, "/", Label_Path),
#          X = map(Image_Path, ~ readNIfTI(fname = .x)),
#          X = map(X, ~ arraynum(.x)),
#          X = map(X, ~ add_dim(.x, 1)),
#          X = map(X, transform_gray_to_rgb_rep),
#          X = map(X, EBImage::normalize, separate = T, ft = c(0, 255)),
#          X = map(X, imagenet_preprocess_input, mode = "torch"),
#          Y = map(Label_Path, ~ readNIfTI(fname = .x)),
#          Y = map(Y, ~ arraynum(.x)),
#          Y = map(Y, to_categorical),
#          Class = CLASS,
#          Y = map_if(Y, Class == 1, ~ add_dim(.x, 1))
#          # Y_hat = map(Y_hat, flip),
#          # Y_hat = map(Y_hat, ~ add_dim(.x, 1))
#   )

X_train = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(X) %>% pull)
Y_train = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(Y) %>% pull)
Y_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(Y) %>% pull)
X_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% val_index) %>% select(X) %>% pull)
Y_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% val_index) %>% select(Y) %>% pull)
str(Y_val)
X_test = list2tensor(image_data_test$X)
Y_test = list2tensor(image_data_test$Y)

pred_batch_size = 1
Y_hat = predict(model,
                X_train,
                batch_size = pred_batch_size)
Y_hat_val = predict(model,
                    X_val,
                    batch_size = pred_batch_size)
Y_hat_test = predict(model,
                     X_test,
                     batch_size = pred_batch_size)
str(Y_hat_val)
display(Y_test[4,,,2] + Y_test[4,,,3])

## display one representative image
display(abind(Y_val[1,,,2], normalize(X_val[1,,,1]), along = 1))
display(abind(
  paintObjects(Y_val[1,,,2], rgbImage(green = Y_hat_val[1,,,2]) > 0.4, col = 'red', thick = T),
  paintObjects(Y_val[1,,,3], rgbImage(green = Y_hat_val[1,,,3]) > 0.4, col = 'red', thick = T),
  along = 1),
  bg = "black")
str(Y_hat_val)

source(paste0(RelPath, "/Scripts/FunctionCompilation/connected_component_analysis.r"))

Y_hat_val_label = apply(Y_hat_val, 1, convert_to_instance_seg, simplify = "array")
Y_hat_val_label = simplify2array(Y_hat_val_label)
Y_hat_val_label = aperm(Y_hat_val_label, c(3, 1, 2))
Y_hat_val_label = add_dim(Y_hat_val_label, dim3 = 1)
str(Y_hat_val_label)

Y_val_label = apply(Y_val, 1, convert_to_instance_seg, simplify = "array")
Y_val_label = simplify2array(Y_val_label)
Y_val_label = aperm(Y_val_label, c(3, 1, 2))
Y_val_label = add_dim(Y_val_label, dim3 = 1)

display(
  colorLabels(
    mul2one(Y_hat_val_label, 5, 1)
  ),
  bg = "black"
)
display(colorLabels(Y_val[4,,,2]))
writeImage(colorLabels(
    mul2one(Y_hat_val_label, 5, 1)
  ), files = paste0("/home/gendarme/Desktop/mask_cca_montage_5.tiff"))

im2load = 1
writeImage(colorLabels(bwlabel(Y_hat_val[im2load,,,2] > 0.5)), files = paste0("/home/gendarme/Desktop/mask_cca_hat_val_center_", im2load,".tiff"))
writeImage(colorLabels(bwlabel(Y_hat_val[im2load,,,3] > 0.5)), files = paste0("/home/gendarme/Desktop/mask_cca_hat_val_border_", im2load,".tiff"))
writeImage(colorLabels(Y_hat_val_label[im2load,,,1]), files = paste0("/home/gendarme/Desktop/mask_cca_hat_val_", im2load,".tiff"))
writeImage(colorLabels(Y_val_label[im2load,,,1]), files = paste0("/home/gendarme/Desktop/mask_cca_gt_val_", im2load,".tiff"))

montage = abind(
  colorLabels(bwlabel(Y_hat_val[im2load,,,2] > 0.5)),
  colorLabels(bwlabel(Y_hat_val[im2load,,,3] > 0.5)),
  colorLabels(Y_hat_val_label[im2load,,,1]),
  colorLabels(Y_val_label[im2load,,,1]),
  along = 1
)
display(montage)
writeImage(montage, files = paste0("/home/gendarme/Desktop/mask_cca_montage_", im2load,".tiff"))

for(i in 1:48){
  writeImage(t(Y_test[i,,,2]) + t(Y_test[i,,,3]), files = paste0("/home/gendarme/Desktop/Y_test_", i,".tiff"))
}

for(i in 1:45){
  writeImage(t(Y_val[i,,,2]) + t(Y_val[i,,,3]), files = paste0("/home/gendarme/Desktop/Y_val_", i,".tiff"))
}

for(i in 1:(dim(Y_train)[1])){
  writeImage(t(Y_train[i,,,2]) + t(Y_train[i,,,3]), files = paste0("/home/gendarme/Desktop/Y_train_", i,".tiff"))
}

for(i in 1:3){
  writeImage(img_to_viridis(Y_val[4,,,i]), files = paste0("/home/gendarme/Desktop/mask_cca_", i,".tiff"))
}

for(i in 1:3){
  writeImage(colorLabels(bwlabel(Y_val[4,,,i])), files = paste0("/home/gendarme/Desktop/mask_cca_color_label_", i,".tiff"))
}

for(i in 1:3){
  writeImage(Y_val[4,,,i], files = paste0("/home/gendarme/Desktop/mask_cca_bw_", i,".tiff"))
}

for(i in 1:3){
  writeImage(Y_hat_val[1,,,i], files = paste0("/home/gendarme/Desktop/mask_cca_bw_1_", i,".tiff"))
}

dice_train_test = array(0, c(nrow(image_data_test), 2))
for(i in 1:nrow(image_data_test)){
  temp = array(0, nrow(image_data))
  for(j in 1:nrow(image_data)){
    temp[j] = dice_coef(image_data_test$Y[[i]][,,2], image_data$Y[[j]][,,2]) %>% as.numeric
  }
  temp = c(max(temp), which(temp == max(temp)))
  dice_train_test[i,] = temp
}
dice_train_test

display(abind(
  image_data_test$Y[[1]][,,2], image_data$Y[[1]][,,2], along = 1
))
