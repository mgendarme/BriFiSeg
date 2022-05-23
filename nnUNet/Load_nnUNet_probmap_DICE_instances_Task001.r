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
TASK = 1

# Load custom functions
list_scripts = list("PreprocessingAndTransformation.r", "Load_data_from_disk.r",
                    "Loss.r") 
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
pred_folder = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task00", TASK, "_A549/nnUNetTrainerV2_200ep__nnUNetPlansv2.1/fold_")
label_folder = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/labelsTr")

arraynum = function(image){
    return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

for(i in 0:4){
    print(paste0("fold ", i))

    # load data
    image_data <- tibble(Ref = list.files(paste0(pred_folder, i, "/validation_raw",
                                                      # "_postprocessed",
                                                      ""), recursive = T)) %>%
        filter(str_detect(Ref, "nii")) %>%
        mutate(Prob_Path = paste0(pred_folder, i, "/validation_raw",
                                  # "_postprocessed",
                                  "/", Ref),
                Label_Path = paste0(label_folder, "/", Ref),
                ID = as.numeric(sapply(str_split(sapply(str_split(Ref, "A549_"), "[", 2), ".nii"), "[", 1)),
                Y = map(Label_Path, ~ readNIfTI(fname = .x)),
                Y_hat = map(Prob_Path, ~ readNIfTI(fname = .x)),
                Y_hat = map(Y_hat, ~ arraynum(.x)),
                Y_hat = map(Y_hat, flip),
                Y_hat = map(Y_hat, ~ add_dim(.x, 1)),
                Y = map(Y, to_categorical),
                Y_hat = map(Y_hat, to_categorical)
                )

    # transform to tensor
    Y_val = list2tensor(image_data$Y)
    Y_hat_val = list2tensor(image_data$Y_hat)
        
    # display one representative image
    display(paintObjects(Y_val[2,,,2], rgbImage(green = Y_hat_val[2,,,2]), col = 'red', thick = T))
  
    # set saving directories for plots and images
    Save_dir = paste0(pred_folder, i)
    Save_plot_instance = paste0(Save_dir, "/Plot_instance_ws")
    dir.create(Save_plot_instance, showWarnings = F)
    Save_image_instance = paste0(Save_dir, "/Image_instance_ws")
    dir.create(Save_image_instance, showWarnings = F)

    ## generate plots for single dice (boxplot dice instances, lin regression)
    ## generate rep. images for instance generation
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single_nnunet.r"))
    source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single_ws_nnunet.r"))

}
