library(keras)
use_implementation("tensorflow")
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

# Load custom functions
list_scripts = list("PreprocessingAndTransformation.r",
                    "Load_data_from_disk.r",
                    "Loss.r") 
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

arraynum = function(image){
    return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

nnunetpath = str_replace(RelPath, "UNet", "nnUNet")

Tasks = c(#"001",    # A549  1 class unet plain          ws
          "002",    # A549  2 class unet plain          cca
          #"006",    # A549  1 class unet seresnext101   ws
          "019")    # A549  2 class unet seresnext101   cca

for(TASK in Tasks){

    message(paste0("Run Task", TASK))

    Save_dir = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Instance_comparison")
    dir.create(Save_dir, showWarnings = F)

    if(TASK %in% c("006", "019")){
        pred_folder = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task", TASK,
                             "_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/imagesTs_predicted_ensemble")
        label_folder = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task", TASK, "_A549/labelsTs")
    } else if(TASK %in% c("001", "002")){
        pred_folder = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task", TASK,
                             "_A549/nnUNetTrainerV2_200ep__nnUNetPlansv2.1/imagesTs_predicted_ensemble")
        label_folder = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task", TASK, "_A549/labelsTs")
    }

    # set saving directories for plots and images
    Save_task = paste0(Save_dir, "/Task", TASK)
    dir.create(Save_task, showWarnings = F)
    Save_data = paste0(Save_task, "/data")
    dir.create(Save_data, showWarnings = F)
    Save_plot_instance = paste0(Save_task, "/Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
    Save_image_instance = paste0(Save_task, "/Image_instance")
    dir.create(Save_image_instance, showWarnings = F)

    list_files = list.files(paste0(pred_folder, ""), pattern="nii.gz", recursive = T)
    instance_tb = tibble

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
        if(TASK %in% c("001", "006")){
            source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single_ws_nnunet_rwa2.r"))
        } else if(TASK %in% c("002", "019")){
            source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single_cca_nnunet_rwa2.r"))
        }
    }
}
