library(keras)
use_implementation("tensorflow")
# tensorflow::tf_gpu_configured()
library(tidyverse)
library(glue)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)

# set seed for reproducibility
set.seed(11)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')

# Load custom functions
list_scripts = list("Params_nnUNet_comparaison.r", "PreprocessingAndTransformation.r",
                    # "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus_v3.r",
                    # "Backbone_Zoo.r",
                    # "Loss2.r",
                    # "CustomGenerator_CropFix_3.r",
                    # "Inspection.r",
                    "Postprocessing.r") %>%
    map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
    source(list_scripts[[l]])
}

Save_dir = paste0(Unet_dir,"/DataAugTest")
dir.create(Save_dir, showWarnings = F)


## load and pre-process data
image_folder = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task001_A549/imagesTr")
label_folder = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task001_A549/labelsTr")

image_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task001_A549/imagesTs")
label_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task001_A549/labelsTs")

arraynum = function(image){
    return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

input <- tibble(Image_Path = list.files(paste0(image_folder), recursive = T),
                Label_Path = list.files(paste0(label_folder), recursive = T)) %>%
    filter(str_detect(Image_Path, "nii")) %>%
    mutate(# generate full path
            Image_Path = paste0(image_folder, "/", Image_Path),
            Label_Path = paste0(label_folder, "/", Label_Path),
            
            # load images
            X = map(Image_Path, ~ readNIfTI(fname = .x)),
            X = map(X, ~ arraynum(.x)),
            X = map(X, ~ add_dim(.x, 1)),
            X = map(X, transform_gray_to_rgb_rep),
            X = map(X, EBImage::normalize, separate = T, ft = c(0, 255)),
            X = map(X, imagenet_preprocess_input, mode = "torch"),
            
            # load labels
            Y = map(Label_Path, ~ readNIfTI(fname = .x)),
            Y = map(Y, ~ arraynum(.x)),
            Y = map(Y, ~ add_dim(.x, 1))
            ) %>% 
    select(X, Y)

test_input <- tibble(Image_Path = list.files(paste0(image_folder_test), recursive = T),
                        Label_Path = list.files(paste0(label_folder_test), recursive = T)) %>%
    filter(str_detect(Image_Path, "nii")) %>%
    mutate(# generate full path
            Image_Path = paste0(image_folder_test, "/", Image_Path),
            Label_Path = paste0(label_folder_test, "/", Label_Path),
            
            # load images
            X = map(Image_Path, ~ readNIfTI(fname = .x)),
            X = map(X, ~ arraynum(.x)),
            X = map(X, ~ add_dim(.x, 1)),
            X = map(X, transform_gray_to_rgb_rep),
            X = map(X, EBImage::normalize, separate = T, ft = c(0, 255)),
            X = map(X, imagenet_preprocess_input, mode = "torch"),
            
            # load labels
            Y = map(Label_Path, ~ readNIfTI(fname = .x)),
            Y = map(Y, ~ arraynum(.x)),
            Y = map(Y, ~ add_dim(.x, 1))
            ) %>% 
    select(X, Y)

## Randomise samples and split into train and test sets:
TRAIN_INDEX <- sample(1:nrow(input),
                        as.integer(round(nrow(input) * (1 - 0.2), 0)),
                        replace = F)
VAL_TEST_INDEX <- c(1:nrow(input))[!c(1:nrow(input) %in% TRAIN_INDEX)]

sampling_generator <- function(data,
                                train_index = TRAIN_INDEX,
                                val_index = VAL_TEST_INDEX) {
    train_input <<- data[train_index,]
    val_input <<- data[val_index,]
}
sampling_generator(input)

str(train_input,  list.len = 2)
str(val_input,  list.len = 2)
str(test_input,  list.len = 2)

train_input_test = train_input %>%
        mutate(last_channel = CLASS,
            Y = map2(Y, last_channel, ~ select_channels(.x, 1, .y)),
            Y = map_if(Y, last_channel == 1, ~ add_dim(.x, 1))) %>%
        select(-c(last_channel))
val_input_test = val_input %>%
        mutate(last_channel = CLASS,
            Y = map2(Y, last_channel, ~ select_channels(.x, 1, .y)),
            Y = map_if(Y, last_channel == 1, ~ add_dim(.x, 1))) %>%
        select(-c(last_channel))

xtest = train_input_test$X[[3]]
ytest = train_input_test$Y[[3]]

display(abind(
    EBImage::normalize(xtest[,,1]),
    EBImage::normalize(ytest[,,1]),
    along = 1))

source(paste0(RelPath, "/Scripts/FunctionCompilation/CustomGenerator_CropFix_4.r"))

train_generator = custom_generator(data = train_input_test,
                            shuffle = TRUE,
                            scale = SCALE,
                            intensity_operation = TRUE,
                            batch_size = BATCH_SIZE)

xtest1 = contrast_aug(xtest, 1.5)
xtest2 = gamma_correction(xtest1, 1.5)
xtest3 = brightness_aug(xtest2, 1.3)
xtest4 = gaussian_noise(xtest3, 0.1)

panel_xtest = normalize(abind(
    xtest,
    xtest1,
    xtest2,
    xtest3,
    xtest4,
    along = 1
))
display(panel_xtest)
writeImage(panel_xtest, paste0(Save_dir, "/panel_multi_aug.tiff"), type = "tiff")


