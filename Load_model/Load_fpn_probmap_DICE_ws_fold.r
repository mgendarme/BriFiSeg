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
TASK = 1
source(paste0(RelPath, "/Scripts/FunctionCompilation/PreprocessingAndTransformation.r"))

# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--8/fpn--seresnext101--Epochs_200--Minibatches_100--FOLD_1")
# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--5/fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_2")

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

exp_folder = "/BF_Data/A549/Prediction/ImgBF512_1Class--60/fpn--seresnext101--FOLD_"
CLASS = 1

for(i in 1:5){
  print(paste0("fold ", i))

  model_folder = paste0(RelPath, exp_folder, i)
  
  Save_dir = paste0(model_folder)
  Save_plot_instance = paste0(Save_dir, "/Plot_instance")
  dir.create(Save_plot_instance, showWarnings = F)
  Save_image_instance = paste0(Save_dir, "/Image_instance")
  dir.create(Save_image_instance, showWarnings = F)
  
  # generate weight model name
  weight_name = paste0(
    "best_",
    "model_weights.hdf5"
    )
  
  # build the model
  enc = ENCODER 
  redfct = 8L
  arc = ARCHITECTURE
  dropout = DROPOUT
  source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
  model %>% load_model_weights_hdf5(filepath = paste0(model_folder, "/", weight_name))
  
  # get index of validation data
  # json_param = jsonlite::read_json(paste0(model_folder, "/detail_param_fold", i,".json"), simplifyVector = TRUE)
  json_param = jsonlite::read_json(paste0(model_folder, "/data_index_fold_", i,".json"), simplifyVector = TRUE)
  train_index = as.numeric(json_param$Train_index)
  val_index = as.numeric(json_param$Val_index)
  
  # load data
  print(paste0("load data fold ", i))
  image_data = load_data(image_folder, label_folder)
  # image_data_test = load_data(image_folder_test, label_folder_test)
  
  # Pred
  print(paste0("run prediction data fold ", i))
  # Y_train = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(Y) %>% pull)
  # X_train = list2tensor(image_data %>% filter(as.numeric(image_id) %in% train_index) %>% select(X) %>% pull)
  X_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% val_index) %>% select(X) %>% pull)
  Y_val = list2tensor(image_data %>% filter(as.numeric(image_id) %in% val_index) %>% select(Y) %>% pull)
  # X_test = list2tensor(image_data_test$X)
  # Y_test = list2tensor(image_data_test$Y)
  
  pred_batch_size = 1
  # Y_hat = predict(model,
  #                 X_train,
  #                 batch_size = pred_batch_size)
  Y_hat_val = predict(model,
                      X_val,
                      batch_size = pred_batch_size)
  # Y_hat_test = predict(model,
  #                      X_test,
  #                      batch_size = pred_batch_size)
  
  ## generate plots for single dice (boxplot dice instances, lin regression)
  ## generate rep. images for instance generation
  print(paste0("run single dice fold ", i))
  source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single_ws_nnunet.r"))

}

