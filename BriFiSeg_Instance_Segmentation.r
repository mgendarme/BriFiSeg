library(keras)
use_implementation("tensorflow")
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)
library(tensorflow)

# set to run tensorflow on cpu if GPU is busy
# tf$config$set_visible_devices(list(), 'GPU')

## Settings
# for linux full path might be necessary due to image export from EBImage that can't deal with relative path
# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/BriFi', '~/Documents/BriFi')
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/GitHub/BriFiSeg', '~/Documents/GitHub/BriFiSeg')

# Load custom functions
list_scripts = list("PreprocessingAndTransformation.r",
                    "Inspection.r",
                    "Load_data_from_disk.r",
                    "Loss.r")
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

## set directiory to find predictions and to export instances to
dataset = "Task001_A549"
run = 5
fold = NULL
Save_dir = ifelse(grepl("Windows", sessionInfo()$running), '~/BF_Data', '~/Documents/BF_Data')
file.copy(from = paste0(paste0(Save_dir, "/", dataset, "/Prediction/Run--", run),"/BriFiSeg_Instance_Segmentation.r"),
          to = paste0(paste0(Save_dir, "/", dataset, "/Prediction/Run--", run),"/BriFiSeg_Instance_Segmentation.r"))

## load parameters used for training 
## (useful to extract back metainformation, e.g. number of semantic classes)
Save_dir = paste0(Save_dir, "/", dataset, "/Prediction/Run--", run)
source(paste0(Save_dir, "/Params.r"))
folds = as.numeric(sub(".*FOLD_", "", list.dirs(paste0(Save_dir), full.names = T, recursive = F)))
folds = folds[!is.na(folds)]

# image and label folder for test set
image_folder = paste0(ifelse(grepl("Windows", sessionInfo()$running), '~/BF_Data/', '~/Documents/BF_Data/'), dataset, "/imagesTs")
label_folder = paste0(ifelse(grepl("Windows", sessionInfo()$running), '~/BF_Data/', '~/Documents/BF_Data/'), dataset, "/labelsTs")

Ens_Img_dir = paste0(Save_dir, "/Ensemble_Pred_Test")
dir.create(Ens_Img_dir, showWarnings = FALSE)

# for ensembling prediction of 5-fold cross validation
if (length(folds) == 5) {
   for(i in 1:length(list.files(image_folder, full.names = T, recursive = F))){
        Y_hat_test1 = readNIfTI(fname = paste0(Save_dir, "/FOLD_", 1, "/predict_test", "/", CELL, "_", make_correct_label(i)))
        Y_hat_test2 = readNIfTI(fname = paste0(Save_dir, "/FOLD_", 2, "/predict_test", "/", CELL, "_", make_correct_label(i)))
        Y_hat_test3 = readNIfTI(fname = paste0(Save_dir, "/FOLD_", 3, "/predict_test", "/", CELL, "_", make_correct_label(i)))
        Y_hat_test4 = readNIfTI(fname = paste0(Save_dir, "/FOLD_", 4, "/predict_test", "/", CELL, "_", make_correct_label(i)))
        Y_hat_test5 = readNIfTI(fname = paste0(Save_dir, "/FOLD_", 5, "/predict_test", "/", CELL, "_", make_correct_label(i)))     
                
        #  class_1 = background
        Y_hat_test_class_1 = abind(Y_hat_test1[,,1], Y_hat_test2[,,1], Y_hat_test3[,,1], Y_hat_test4[,,1], Y_hat_test5[,,1], along = 3)
        Y_hat_test_class_1 = apply(Y_hat_test_class_1, c(1, 2), mean)
       
        # class_2 = nuclei unless trained on two non-background semantic classes than class_2 = center of nuclei
        Y_hat_test_class_2 = abind(Y_hat_test1[,,2], Y_hat_test2[,,2], Y_hat_test3[,,2], Y_hat_test4[,,2], Y_hat_test5[,,2], along = 3)
        Y_hat_test_class_2 = apply(Y_hat_test_class_2, c(1, 2), mean)
        
        if (CLASS == 1) {
           Y_hat_test = abind(Y_hat_test_class_1, Y_hat_test_class_2, along = 3)
        } else if(CLASS == 2){
            # class_3 = not existent unless trained on two non-background semantic classes than class_3 = border of nuclei
            Y_hat_test_class_3 = abind(Y_hat_test1[,,3], Y_hat_test2[,,3], Y_hat_test3[,,3], Y_hat_test4[,,3], Y_hat_test5[,,3], along = 3)
            Y_hat_test_class_3 = apply(Y_hat_test_class_3, c(1, 2), mean)
            Y_hat_test = abind(Y_hat_test_class_1, Y_hat_test_class_2, Y_hat_test_class_3, along = 3)
        }

        temp_im_name = paste0(Ens_Img_dir, "/", CELL, "_", make_correct_label(i))

        writeNIfTI(
            nim = Y_hat_test,
            filename = temp_im_name,
            onefile = TRUE,
            gzipped = TRUE,
            verbose = FALSE,
            warn = -1,
            compression = 1
        )

        print(paste0("test sample #:" , i, " || Sample ID: ", i))
   }
    pred_folder = Ens_Img_dir
    # set saving directories for plots and images
    Save_data = paste0(Save_dir, "/data")
    dir.create(Save_data, showWarnings = F)
    Save_plot_instance = paste0(Save_dir, "/Ensemble_Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
    Save_image_instance = paste0(Save_dir, "/Ensemble_Image_instance")
    dir.create(Save_image_instance, showWarnings = F)

# for indiviudal folds of 5-fold cross validation
} else {
    # pick a fold or modify to iterate over folds if of interest
    picked_fold = 5
    pred_folder = paste0(Save_dir, "/FOLD_", picked_fold, "/predict_test") 
    # set saving directories for plots and images
    Save_data = paste0(Save_dir, "/FOLD_", picked_fold, "/data")
    dir.create(Save_data, showWarnings = F)
    Save_plot_instance = paste0(Save_dir, "/FOLD_", picked_fold, "/Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
    Save_image_instance = paste0(Save_dir, "/FOLD_", picked_fold, "/Image_instance")
    dir.create(Save_image_instance, showWarnings = F)
}

list_files = list.files(paste0(pred_folder, ""), pattern="nii.gz", recursive = T)
instance_tb = tibble

for(i in 1:length(list_files)){

    # load data
    image_data <- tibble(Ref = list_files[i]) %>%
        filter(str_detect(Ref, "nii")) %>%
        mutate(Image_Path = paste0(image_folder, "/", str_replace(Ref, ".nii.gz", "_0000.nii.gz")), 
                Prob_Path = paste0(pred_folder, "/", Ref),
                Label_Path = paste0(label_folder, "/", Ref),
                ID = as.numeric(sapply(str_split(sapply(str_split(Ref, "A549_"), "[", 2), ".nii"), "[", 1)),
                Well = sapply(str_split(sapply(str_split(Ref, "_"), "[", 1), ".nii"), "[", 1), 
                X = map(Image_Path, ~ readNIfTI(fname = .x)),
                Y = map(Label_Path, ~ readNIfTI(fname = .x)),
                Y_hat = map(Prob_Path, ~ readNIfTI(fname = .x)),
                # Y_hat = map(Y_hat, ~ arraynum(.x)),
                # Y_hat = map(Y_hat, flip),
                # Y_hat = map(Y_hat, ~ add_dim(.x, 1)),
                Y = map(Y, to_categorical)#,
                # Y_hat = map(Y_hat, to_categorical)
                )
    
    ID = unique(image_data$ID)
    Well = unique(image_data$Well)

    # transform to tensor
    X_val = list2tensor(image_data$X)
    Y_val = list2tensor(image_data$Y)
    Y_hat_val = list2tensor(image_data$Y_hat)

    # derives instances from probability maps using watershed-based
    # instance segmentation when trained on one class (entire nuclei)
    # or uses connected component analysis when trained on two classes 
    # namely center and borders of objects
    if(CLASS == 1){
        source(paste0(RelPath, "/Inspection/Dice_instance_ws.r"))
    } else if(CLASS == 2){
        source(paste0(RelPath, "/Inspection/Dice_instance_cca.r.r"))
    }
   
}

# generate plots for dice per instance, instance count as well as
# extra, missed, under- and over-split objects
source(paste0(RelPath, "/Inspection/Instance_analysis.r"))
