library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(glue)
library(EBImage)
options(EBImage.display = "raster")
options(EBImage.bg = "black")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)
library(jsonlite)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = '/home/gendarme/Documents/UNet'
Unet_dir = paste0(RelPath, "/BF_Data")

# Load custom functions
list_scripts = list("PreprocessingAndTransformation.r",
                    "Load_data_from_disk.r",
                    "Model_FPN.r",
                    "Model_UNET_new.r", 
                    "Model_PSPNET.r", 
                    "Model_Deeplabv3plus_v3.r",
                    "Backbone_Zoo.r",
                    "Loss.r",
                    "CustomGenerator_CropFix.r",
                    "Inspection.r",
                    "Postprocessing.r")
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

for(cell in c("A549", "HELA", "MCF7", "RPE1"
)){
    # cell="HELA"

    print(cell)
    
    if(cell == "HELA"){
        ds = paste0("Task010_", cell)
    } else if(cell == "MCF7"){
        ds = paste0("Task011_", cell)
    } else if(cell == "RPE1"){
        ds = paste0("Task015_", cell)
    } else if(cell == "A549"){
        ds = paste0("Task001_", cell)
    }

    image_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/", ds, "/imagesTs")
    label_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/", ds, "/labelsTs")
    
    # Parameters for saving files
    Current_i = as.character(paste0("ImgBF512_1Class"))
    Save_dir = paste0(Unet_dir,"/",cell,"/","Prediction/", Current_i, "--",ifelse(cell == "A549", 100, 1))
    Save_plot_semantic = paste0(Save_dir, "/Plot_semantic")
    dir.create(Save_plot_semantic, showWarnings = F)
    source(paste0(Save_dir, "/Params_nnUNet_comparaison.r"))

    # generate weight model name
    weight_name = paste0("best_", "model_weights.hdf5")
    
    # build the models
    enc = ENCODER 
    redfct = 8L
    arc = ARCHITECTURE
    dropout = DROPOUT
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
    model1 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 1, "/", weight_name))
    model2 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 2, "/", weight_name))
    model3 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 3, "/", weight_name))
    model4 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 4, "/", weight_name))
    model5 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 5, "/", weight_name))

    test_input = load_data(image_folder_test, label_folder_test)

    ### Generate single instances and perform measurements #####################################################
    ## generate plots for IOU and DICE
    source(paste0(RelPath, "/Scripts/Inspection/Old/IOU_DICE_plot_2_ensemble.r"))

    ## generate sample images
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images_no_dapi.r"))
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images_no_dapi_2.r"))

}
