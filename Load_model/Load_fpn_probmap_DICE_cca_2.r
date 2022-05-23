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

# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--8/fpn--seresnext101--Epochs_200--Minibatches_100--FOLD_1")
# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--5/fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_2")
model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--11/fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_2")
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
list_scripts = list("Params_nnUNet_comparaison.r", "PreprocessingAndTransformation.r", "Load_data_from_disk.r",
                    "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus.r",
                    "Loss.r",
                    "Backbone_Zoo.r",
                    "Inspection.r",
                    "Postprocessing.r") 
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

Save_dir = paste0(model_folder)
Save_plot_instance = paste0(Save_dir, "/Plot_instance")
dir.create(Save_plot_instance, showWarnings = F)
Save_image_instance = paste0(Save_dir, "/Image_instance")
dir.create(Save_image_instance, showWarnings = F)


weight_name = paste0(
  "best_",
  "model_weights.hdf5")

# build the model
enc = ENCODER 
redfct = 8L
arc = ARCHITECTURE
dropout = DROPOUT
source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
# model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
#                   backbone = ENCODER,
#                   nlevels = NULL,
#                   output_activation = ACTIVATION,
#                   output_channels = CLASS,
#                   decoder_skip = FALSE,
#                   dropout = DROPOUT
# )
# if model saved
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
model %>% load_model_weights_hdf5(filepath = paste0(model_folder, "/", weight_name))

# get index of validation data
json_param = jsonlite::read_json(paste0(model_folder, "/detail_param_fold_2.json"), simplifyVector = TRUE)
train_index = as.numeric(json_param$Train_index)
val_index = as.numeric(json_param$Val_index)

# load data
image_data = load_data(image_folder, label_folder)
image_data_test = load_data(image_folder_test, label_folder_test)

Y_train = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(Y) %>% pull)
X_train = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(X) %>% pull)
Y_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(Y) %>% pull)
X_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% val_index) %>% select(X) %>% pull)
Y_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% val_index) %>% select(Y) %>% pull)
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
#  str(Y_hat_val)

## generate plots for single dice (boxplot dice instances, lin regression)
## generate rep. images for instance generation
source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single.r"))

# ## display one representative image
# display(abind(Y_val[1,,,2], normalize(X_val[1,,,1]), along = 1))
# display(abind(
#   paintObjects(Y_val[1,,,2], rgbImage(green = Y_hat_val[1,,,2]) > 0.4, col = 'red', thick = T),
#   paintObjects(Y_val[1,,,3], rgbImage(green = Y_hat_val[1,,,3]) > 0.4, col = 'red', thick = T),
#   along = 1),
#   bg = "black")
# str(Y_hat_val)



# str(Y_hat_val_label)
# display(
#   colorLabels(
#     mul2one(Y_hat_val_label, 5, 1)
#   ),
#   bg = "black"
# )

# dice_train = array(0, c(nrow(image_data), 2))
# for(i in 1:nrow(image_data)){
#   temp = array(0, nrow(image_data))
#   for(j in 1:nrow(image_data)){
#     temp[j] = dice_coef(image_data_test$Y[[i]][,,2], image_data$Y[[j]][,,2]) %>% as.numeric
#   }
#   temp = c(max(temp), which(temp == max(temp)))
#   dice_train_test[i,] = temp
# }

# display(
#   abind(
#   colorLabels(Y_val_label[1,,,]),
#   colorLabels(Y_hat_val_label[1,,,]),
#   along = 1))
# display(
#   abind(
#   colorLabels(Y_test_label[30,,,]),
#   colorLabels(Y_hat_test_label[30,,,]),
#   along = 1))
# for(i in 1:dim(Y_hat_test)[1]){
#   print(paste0(
#     "img #: ", i,
#     " || gt: ", max(Y_test_label[i,,,]),
#     "|| pred: ", max(Y_hat_test_label[i,,,])
#   ))
# }


# img2write = abind(colorLabels(bwlabel(Y_hat_val[1,,,2] > 0.5)),
#                   colorLabels(bwlabel((Y_hat_val[1,,,2] + Y_hat_val[1,,,3]) > 0.5)),
#                   colorLabels(Y_hat_val_label[1,,,1]),
#                   colorLabels(convert_to_instance_seg(Y_val[1,,,])), #+ Y_val[1,,,3])),
#                   along = 1
#                   )
# display(img2write)
# writeImage(img2write, files = paste0("/home/gendarme/Desktop/mask_cca_bef_aft_cca", "",".tiff"))

# for(i in 1:3){
#   writeImage(img_to_viridis(Y_val[4,,,i]), files = paste0("/home/gendarme/Desktop/mask_cca_", i,".tiff"))
# }

# dice_train_test = array(0, c(nrow(image_data_test), 2))
# for(i in 1:nrow(image_data_test)){
#   temp = array(0, nrow(image_data))
#   for(j in 1:nrow(image_data)){
#     temp[j] = dice_coef(image_data_test$Y[[i]][,,2], image_data$Y[[j]][,,2]) %>% as.numeric
#   }
#   temp = c(max(temp), which(temp == max(temp)))
#   dice_train_test[i,] = temp
# }
# dice_train_test

# display(abind(
#   image_data_test$Y[[1]][,,2], image_data$Y[[1]][,,2], along = 1
# ))
