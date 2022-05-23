library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)

# set seed for reproducibility
set.seed(11)
TASK = "019"
## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet/')
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")

# imagenet settings
imagenet_settings = list(
    'input_space' = 'RGB',
    'input_range' = c(0, 1),
    'mean' = c(0.485, 0.456, 0.406),
    'std' = c(0.229, 0.224, 0.225)
)

# https://github.com/qubvel/segmentation_models.pytorch/blob/4f94380815f831605f4641b7193df2eccd5652a3/segmentation_models_pytorch/encoders/_preprocessing.py#L4
preprocess_input = function(x, settings){
    
    mean = settings$mean
    std = settings$std
    input_space = settings$input_space
    input_range = settings$input_range
    
    if(input_space == "BGR"){
        x = x[,,c(2, 3, 1)]
    }

    if(!is.null(input_range)){
        if(max(x) > 1 & input_range[2] == 1){
            # x = x / 255.0
            x = EBImage::normalize(x)
        }
    }
    
    if(!is.null(mean)){
        x[,,1] = x[,,1] - mean[1]
        x[,,2] = x[,,2] - mean[2]
        x[,,3] = x[,,3] - mean[3]
    }
    
    
    if(!is.null(std)){
        x[,,1] = x[,,1] / std[1]
        x[,,2] = x[,,2] / std[2]
        x[,,3] = x[,,3] / std[3]
    }
    
    return(x)

}

# Load custom functions
list_scripts = list("Params_convert4nnunet.r", "PreprocessingAndTransformation.r"#,
                    # "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus_v3.r",
                    # "Backbone_Zoo.r",
                    # "Loss.r",
                    # "CustomGenerator_CropFix_2.r",
                    # "Inspection.r",
                    # "Postprocessing.r"
                    )
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

# set path for nnunet training folder
task_dir = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task", TASK, "_A549")
dir.create(task_dir, showWarnings = FALSE)
Train_Img_dir = paste0(task_dir, "/imagesTr")
Train_Lbl_dir = paste0(task_dir, "/labelsTr")
Test_Img_dir = paste0(task_dir, "/imagesTs")
Test_Lbl_dir = paste0(task_dir, "/labelsTs")
for(i in c(Train_Img_dir, Train_Lbl_dir, Test_Img_dir, Test_Lbl_dir)){
    dir.create(i)
}

# load data
## original
# source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_nnunet.r"))
## with border and center for connected component analysis
source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_nnunet_border_center.r"))
str(input, list.len = 2)

input_crop = input %>%
        mutate(X = map(X, ~ crop_mut(.x, cropping = 512)),
            X = map(X, transform_gray_to_rgb_rep),
            X = map(X, EBImage::normalize, separate = T),
            X = map(X, ~ preprocess_input(.x, settings = imagenet_settings)),
            R = map(X, ~ select_channels(.x, 1, 1)),
            G = map(X, ~ select_channels(.x, 2, 2)),
            B = map(X, ~ select_channels(.x, 3, 3)),
            # X = map(X, ~ add_dim(.x, 1)),
            R = map(R, ~ add_dim(.x, 1)),
            G = map(G, ~ add_dim(.x, 1)),
            B = map(B, ~ add_dim(.x, 1)),
            Y = map(Y, ~ crop_mut(.x, cropping = 512)),
            # Y = map(Y, ~ select_channels(.x, 1, 1)),
            Y = map(Y, ~ add_dim(.x, 1))
            ) %>% 
        select(X, R, G, B, Y, ID)
            # Y = map(Y, ~ add_dim_rgb_nifti(.x, 1)))
    # glimpse(input_crop)
str(input_crop, list.len = 5)


input$X[[1]][1:3, 1:3, 1]
input_crop$X[[1]][1:3, 1:3, 1:3]
range(input_crop$Y[[1]])

# Convert to nifti
input_nifti = input_crop %>%
    mutate(X = as.nifti(X),
           Y = as.nifti(Y))
str(input_nifti, list.len = 2)

# Export
## S4 method for signature 'nifti'
make_correct_label = function(label){
    if(str_length(label) == 1){
        return(paste0("00", label))
    } else if (str_length(label) == 2){
       return(paste0("0", label))
    } else if(str_length(label) == 3){
       return(label)
    }
}

## Randomise samples and split into train and test sets:
TRAIN_INDEX <- sample(1:nrow(input),
                      as.integer(round(nrow(input) * (1 - VALIDATION_SPLIT), 0)),
                      replace = F)
NOT_TRAIN_INDEX <- c(1:nrow(input))[!c(1:nrow(input) %in% TRAIN_INDEX)]
VAL_TEST_INDEX <- sample(NOT_TRAIN_INDEX,
                         as.integer(round(length(NOT_TRAIN_INDEX) * 0.5, 0)),
                         replace = F)

sampling_generator <- function(data, train_index = TRAIN_INDEX, val_index = VAL_TEST_INDEX) {
    train_input <<- data[train_index,]
    val_input <<- data[val_index,]
    test_input <<- data[-c(val_index, train_index),]
}
sampling_generator(input_nifti)

# train_input = rbind(train_input, val_input)

# str(train_input)
# TRAIN_INDEX

# write the iamges and labels to disk
for(i in 1:nrow(train_input)){
        for (j in 0:2) {
            temp_im_name = paste0(Train_Img_dir, "/", CELL, "_", make_correct_label(i), "_000", j)
                        
            if(j == 0){
                temp_im = train_input$R[[i]]
            } else if(j == 1){
                temp_im = train_input$G[[i]]
            } else if(j == 2){
                temp_im = train_input$B[[i]]
            }
            
            writeNIfTI(
                nim = as.nifti(temp_im),
                filename = temp_im_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )
            
        }

        temp_lbl_name = paste0(Train_Lbl_dir, "/", CELL, "_", make_correct_label(i))
        writeNIfTI(
                nim = as.nifti(train_input$Y[[i]]),
                filename = temp_lbl_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )

        print(paste0("train sample #:" , i))
}
str(train_input$Y[[i]])
str(train_input$R[[i]])

# for(i in 1:nrow(train_input)){
#     temp_im_name = paste0(Train_Img_dir, "/", CELL, "_", make_correct_label(i), "_0000")
#     temp_lbl_name = paste0(Train_Lbl_dir, "/", CELL, "_", make_correct_label(i))

#     writeNIfTI(
#         nim = as.nifti(train_input$X[[i]]),
#         filename = temp_im_name,
#         onefile = TRUE,
#         gzipped = TRUE,
#         verbose = FALSE,
#         warn = -1,
#         compression = 1 # replcae with 0
#     )
#     writeNIfTI(
#         nim = as.nifti(train_input$Y[[i]]),
#         filename = temp_lbl_name,
#         onefile = TRUE,
#         gzipped = TRUE,
#         verbose = FALSE,
#         warn = -1,
#         compression = 1 # replace with 0
#     )
# }
for(i in 1:nrow(test_input)){
        for (j in 0:2) {
            temp_im_name = paste0(Test_Img_dir, "/", CELL, "_", make_correct_label(i), "_000", j)
                        
            if(j == 0){
                temp_im = test_input$R[[i]]
            } else if(j == 1){
                temp_im = test_input$G[[i]]
            } else if(j == 2){
                temp_im = test_input$B[[i]]
            }
            
            writeNIfTI(
                nim = as.nifti(temp_im),
                filename = temp_im_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )
            
        }

        temp_lbl_name = paste0(Test_Lbl_dir, "/", CELL, "_", make_correct_label(i))
        writeNIfTI(
                nim = as.nifti(test_input$Y[[i]]),
                filename = temp_lbl_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )

        print(paste0("test sample #:" , i))
}
# for(i in 1:nrow(test_input)){
#     temp_im_name = paste0(Test_Img_dir, "/", CELL, "_", make_correct_label(i), "_0000")
#     temp_lbl_name = paste0(Test_Lbl_dir, "/", CELL, "_", make_correct_label(i))
    
#     writeNIfTI(
#         nim = test_input$X[[i]],
#         filename = temp_im_name,
#         onefile = TRUE,
#         gzipped = TRUE,
#         verbose = FALSE,
#         warn = -1,
#         compression = 1
#     )
#     writeNIfTI(
#         nim = test_input$Y[[i]],
#         filename = temp_lbl_name,
#         onefile = TRUE,
#         gzipped = TRUE,
#         verbose = FALSE,
#         warn = -1,
#         compression = 1
#     )
# }

# write json to summarize the data

json_train = list()
for(i in 1:nrow(train_input)){
    temp_im_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
    temp_lbl_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
    train = I(list(
        "image" = paste0("./imageTr/", temp_im_name),   
        "label" = paste0("./labelTr/", temp_lbl_name)
    ))
    if(i == 1){
        json_train[[i]] = train
    } else {
        json_train[[i]] = train
    }
}
# json_train

json_test = list()
for(i in 1:nrow(test_input)){
    temp_im_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
    temp_lbl_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
    test = I(list(
        "image" = paste0("./imageTs/", temp_im_name),   
        "label" = paste0("./labelTs/", temp_lbl_name)
    ))
    if(i == 1){
        json_test[[i]] = test
    } else {
        json_test[[i]] = test
    }
}
# json_test

json_list = list(
    name = CELL, 
    description = "Nuclear brightfield segmentation",
    reference = "Gendarme M, BioMed X Institute",
    licence = "NA",
    release = "1.0 11/09/2021",
    tensorImageSize = "4D",
    modality = I(list( 
        `0` = "R",
        `1` = "G",
        `2` = "B"
    )),
    labels = I(list(
        `0` = "background",   
        `1` = "nucleus_center",
        `2` = "nucleus_border"
    )),
    numTraining = nrow(train_input),
    numTest = 0, #nrow(test_input),
    training = json_train,
    test = I(list())#json_test
)
# json_list = list(Cell = CELL,
#                  Train_index = train_input$image_id,
#                  Val_index = val_input$image_id)
json_list = jsonlite::toJSON(json_list, auto_unbox = TRUE, pretty = TRUE)

write_json_data_index = function(data, filename){
    sink(filename)
    print(data)
    sink()
}

write_json_data_index(json_list, paste0(task_dir, "/dataset", ".json"))
message(paste0("Save json dataset"))
# str(train_input, list.len = 2)
# idx = 8
# display(abind(normalize(train_input$X[[idx]][1:512, 1:512, ]),
#                 train_input$Y[[idx]][1:512, 1:512, ] == 1,
#                 train_input$Y[[idx]][1:512, 1:512, ] == 2,
#                 along = 1
#                 # color_1 = "#bebebe85",
#                 # color_2 = "#ff000013",
#                 # color_3 = "#00ff0017",
#                 # dimension = dim(normalize(input$X[[1]][1:256, 1:256, 1]))
#                 ))

###############################################################################################
##################### run in terminal for preprocessing wihtout norm ##########################
# nnUNet_plan_and_preprocess -t 019 -pl3d None -pl2d ExperimentPlanner2D_v21_RGB_nonorm
###############################################################################################