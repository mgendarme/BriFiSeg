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

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")

# Load custom functions
list_scripts = list("Params_convert4nnunet.r", "PreprocessingAndTransformation.r"#,
                    # "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus_v3.r",
                    # "Backbone_Zoo.r",
                    # "Loss.r",
                    # "CustomGenerator_CropFix_2.r",
                    # "Inspection.r",
                    # "Postprocessing.r"
                    ) %>%
    map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
    source(list_scripts[[l]])
}

# set path for nnunet training folder
task_dir = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task001_A549")
dir.create(task_dir, showWarnings = FALSE)
Train_Img_dir = paste0(task_dir, "/imagesTr")
Train_Lbl_dir = paste0(task_dir, "/labelsTr")
Test_Img_dir = paste0(task_dir, "/imagesTs")
Test_Lbl_dir = paste0(task_dir, "/labelsTs")
for(i in c(Train_Img_dir, Train_Lbl_dir, Test_Img_dir, Test_Lbl_dir)){
    dir.create(i)
}

# load data
source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_nnunet.r"))
str(input, list.len = 2)

input_crop = input %>%
    mutate(X = map(X, ~ crop_mut(.x, cropping = 512)),
           X = map(X, ~ add_dim(.x, 1)),
           Y = map(Y, ~ crop_mut(.x, cropping = 512)),
           Y = map(Y, ~ add_dim(.x, 1)))
str(input_crop, list.len = 2)

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

for(i in 1:nrow(train_input)){
    temp_im_name = paste0(Train_Img_dir, "/", CELL, "_", make_correct_label(i), "_0000")
    temp_lbl_name = paste0(Train_Lbl_dir, "/", CELL, "_", make_correct_label(i))

    writeNIfTI(
        nim = as.nifti(train_input$X[[i]]),
        filename = temp_im_name,
        onefile = TRUE,
        gzipped = TRUE,
        verbose = FALSE,
        warn = -1,
        compression = 1 # replcae with 0
    )
    writeNIfTI(
        nim = as.nifti(train_input$Y[[i]]),
        filename = temp_lbl_name,
        onefile = TRUE,
        gzipped = TRUE,
        verbose = FALSE,
        warn = -1,
        compression = 1 # replace with 0
    )
}

for(i in 1:nrow(test_input)){
    temp_im_name = paste0(Test_Img_dir, "/", CELL, "_", make_correct_label(i), "_0000")
    temp_lbl_name = paste0(Test_Lbl_dir, "/", CELL, "_", make_correct_label(i))
    
    writeNIfTI(
        nim = test_input$X[[i]],
        filename = temp_im_name,
        onefile = TRUE,
        gzipped = TRUE,
        verbose = FALSE,
        warn = -1,
        compression = 1
    )
    writeNIfTI(
        nim = test_input$Y[[i]],
        filename = temp_lbl_name,
        onefile = TRUE,
        gzipped = TRUE,
        verbose = FALSE,
        warn = -1,
        compression = 1
    )
}

# write json to summarize the data

json_train = list()
for(i in 1:nrow(train_input)){
    temp_im_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
    temp_lbl_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
    train = I(list(
        "image" = paste0("./imageTs/", temp_im_name),   
        "label" = paste0("./labelTs/", temp_lbl_name)
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
        `0` = "brightfield"
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

# dim(input_nifti$X[[1]])
# image(input_nifti$Y[[1]])#, z = 2, plot.type = "single")
# display(flip(input_crop$Y[[1]]))
 
# writeNIfTI(
#     nim = train_input$X[[1]],
#     filename = "~/Desktop/Compression9",
#     onefile = TRUE,
#     gzipped = TRUE,
#     verbose = FALSE,
#     warn = -1,
#     compression = 1
# )
 
# writeNIfTI(
#     nim = train_input$X[[1]],
#     filename = "~/Desktop/Compression0",
#     onefile = TRUE,
#     gzipped = TRUE,
#     verbose = FALSE,
#     warn = -1,
#     compression = 1
# )

# compr0 = readNIfTI("~/Desktop/Compression0")
# compr9 = readNIfTI("~/Desktop/Compression9")
# image(compr0, z = 1, plot.type = 0)
# image(compr9, z = 3, plot.type = 0)
# display(flip(normalize(train_input$X[[1]][,,1])), bg = "black")

# json_train = ''
# for(i in 1:nrow(train_input)){
#     temp_im_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
#     temp_lbl_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
#     json_train_temp = paste0('{"image":"./imagesTr/', temp_im_name,'"', 
#                             ',',
#                             '"label":"./labelsTr/', temp_lbl_name, '"}')
#     if(i == 1){
#         json_train = json_train_temp
#     } else {
#         json_train = paste0(json_train, ',', json_train_temp)    
#     }
# }
# cat(json_train)


# json_test = ''
# for(i in 1:nrow(val_input)){
#     temp_im_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
#     temp_lbl_name = paste0(CELL, "_", make_correct_label(i), ".nii.gz")
#     json_val_temp = paste0('{"image":"./imagesTs/', temp_im_name,'"', 
#                              ',',
#                              '"label":"./labelsTs/', temp_lbl_name, '"}')
#     if(i == 1){
#         json_test = json_val_temp
#     } else {
#         json_test = paste0(json_test, ',', json_val_temp)    
#     }
# }
# cat(json_test)

# json = 
# paste0(
# '{
#  "name": "A549", 
#  "description": "Nuclear brightfield segmentation",
#  "reference": "Gendarme M, BioMed X Institute",
#  "licence":"NA",
#  "release":"1.0 11/09/2021",
#  "tensorImageSize": "4D",
#  "modality": { 
#    "0": "brightfield"
#  }, 
#  "labels": { 
#    "0": "background", 
#    "1": "nucleus"
#  }, 
#  # "numTraining": ', nrow(train_input), ', 
#  # "numTest": ', nrow(val_input), ',
#  # "training": [', json_train, '],
#  # "test": [', json_test, ']
#  }'
# )
# 
# json = list(
#     name = c("A549"), 
#     description = "Nuclear brightfield segmentation",
#     reference = "Gendarme M, BioMed X Institute",
#     licence = "NA",
#     release = "1.0 11/09/2021",
#     tensorImageSize = "4D",
#     modality = I(list( 
#         `0` = "brightfield"
#     )),
#     labels = I(list(
#         `0` = "background",   
#         `1` = "nucleus"
#     )),
#     numTraining = nrow(train_input),
#     numTest = nrow(val_input),
#     training = json_train,
#     test = json_test
# )
#     
# jsonlite::write_json(jsonlite::toJSON(json),"~/Desktop/tst.json")
# 
#     
# library(jsonlite)
# 
# ID=c(100,110,200)
# Title=c("aa","bb","cc")
# more=I(list(Interesting=c("yes","no","yes"),new=c("no","yes","yes"),original=c("yes","yes","no")))
# 
# a=list(ID=ID,Title=Title,more=more)
# a=jsonlite::toJSON(a)
# write(a,"~/Desktop/temp.JSON")
# 
# ID=c(100,110,200)
# Title=c("aa","bb","cc")
# 
# df <- data.frame(ID, Title)
# more=data.frame(Interesting=c("yes","no","yes"),new=c("no","yes","yes"),original=c("yes","yes","no"))
# df$more <- more
# 
# jsonlite::toJSON(df)
# jsonlite::write_json(df, "~/Desktop/df_json.json")
# jsonlite::read_json(path = "/home/gendarme/Desktop/dataset.json")
# 
# test = rjson::fromJSON(file = "/home/gendarme/Desktop/dataset.json")


