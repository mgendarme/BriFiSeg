library(keras)
use_implementation("tensorflow")
# tensorflow::tf_gpu_configured()
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
TASK = 17

# Load custom functions
list_scripts = list("PreprocessingAndTransformation.r",
                    "Load_data_from_disk.r",
                    "Loss.r") 
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
Save_dir = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/WS")
dir.create(Save_dir, showWarnings = F)
pred_folder = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/imagesTsRWA_predicted_ensemble")
label_folder = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK+1, "_RWA_A549_TEST/labelsTsRWA")

arraynum = function(image){
    return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

# set saving directories for plots and images
# Save_dir = paste0(pred_folder, i)
Save_data = paste0(Save_dir, "/data2")
dir.create(Save_data, showWarnings = F)
Save_plot_instance = paste0(Save_dir, "/Plot_instance_ws")
dir.create(Save_plot_instance, showWarnings = F)
Save_image_instance = paste0(Save_dir, "/Image_instance_ws")
dir.create(Save_image_instance, showWarnings = F)

list_files = list.files(paste0(pred_folder, ""), pattern="nii.gz", recursive = T)
instance_tb = tibble

# library(doMC)
# registerDoMC(8)

# foreach(i = 1:8, .packages = c("EBImage", "tidyverse", "keras")) %dopar% {#
for(i in 1:length(list_files)){
    
    # load data
    image_data <- tibble(Ref = list_files[i]) %>%
        filter(str_detect(Ref, "nii")) %>%
        mutate(Prob_Path = paste0(pred_folder,
                                  "/", Ref),
                Label_Path = paste0(label_folder, "/", Ref),
                ID = as.numeric(sapply(str_split(sapply(str_split(Ref, "A549_"), "[", 2), ".nii"), "[", 1)),
                Well = sapply(str_split(sapply(str_split(Ref, "_"), "[", 1), ".nii"), "[", 1), 
                Y = map(Label_Path, ~ readNIfTI(fname = .x)),
                Y_hat = map(Prob_Path, ~ readNIfTI(fname = .x)),
                Y_hat = map(Y_hat, ~ arraynum(.x)),
                Y_hat = map(Y_hat, flip),
                Y_hat = map(Y_hat, ~ add_dim(.x, 1)),
                Y = map(Y, to_categorical),
                Y_hat = map(Y_hat, to_categorical)
                )
    ID = unique(image_data$ID)
    Well = unique(image_data$Well)

    # transform to tensor
    Y_val = list2tensor(image_data$Y)
    Y_hat_val = list2tensor(image_data$Y_hat)
    # str(Y_val)

    # img = (abind(
    #     colorLabels(Y_val[1,,,2]),
    #     colorLabels(Y_hat_val[1,,,2]),
    #     along=1
    # ))
    # writeImage(img, "~/Desktop/img.png")

    # print(paste0("---------- Dice instances validation ----------"))
    source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single_ws_nnunet_rwa2.r"))

}
