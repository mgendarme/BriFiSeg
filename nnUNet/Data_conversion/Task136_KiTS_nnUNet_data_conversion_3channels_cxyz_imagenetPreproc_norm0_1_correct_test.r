library(tidyverse)
library(EBImage)
library(oro.nifti)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = '/home/gendarme/Documents/UNet'
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
TASK = "Task136_KiTS2021_2D_RGB"
orig_file_path = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data", "/Task135_KiTS2021")

# imagenet settings for seresnext101
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
            # x = x / 255.0 # if 8-bit 
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

make_correct_label = function(label){
    if(str_length(label) == 1){
        return(paste0("00", label))
    } else if (str_length(label) == 2){
       return(paste0("0", label))
    } else if(str_length(label) == 3){
       return(label)
    }
}

write_json_data_index = function(data, filename){
    sink(filename)
    print(data)
    sink()
}

# set path for nnunet training folder
task_dir = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/", TASK) # dir.create(task_dir, showWarnings = FALSE)
Train_Img_dir = paste0(task_dir, "/imagesTr")
Train_Lbl_dir = paste0(task_dir, "/labelsTr")
Test_Img_dir = paste0(task_dir, "/imagesTs")
Test_Lbl_dir = paste0(task_dir, "/labelsTs")
for(i in c(task_dir, Train_Img_dir, Train_Lbl_dir, Test_Img_dir, Test_Lbl_dir)){
    dir.create(i)
}

train_input_img = list.files(paste0(orig_file_path, "/imagesTr"), pattern = "nii.gz")
train_input_lbl = list.files(paste0(orig_file_path, "/labelsTr"), pattern = "nii.gz")

# read modify and write the images and labels to disk
for(i in 0:2){#(length(train_input_img)-1)){
    # i=0
    temp_img = readNIfTI(fname = paste0(orig_file_path, "/imagesTr/", train_input_img[i+1]), reorient = F)
    temp_lbl = readNIfTI(fname = paste0(orig_file_path, "/labelsTr/", train_input_lbl[i+1]), reorient = F)

    for(s in 1:dim(temp_img)[3]) {
        # s=1
        slice_img = EBImage::normalize(temp_img[,,s])
        dim(slice_img) = c(dim(slice_img), 1)
        slice_img = abind(slice_img, slice_img, slice_img, along = 3)
        slice_img = preprocess_input(slice_img, imagenet_settings)

        slice_lbl = temp_lbl[,,s]
        
        for(j in 0:2){
            # j=0
            temp_im_name = paste0(Train_Img_dir, "/case_", paste0(make_correct_label(i), make_correct_label(s)), "_000", j)
            temp_lbl_name = paste0(Train_Lbl_dir, "/case_", paste0(make_correct_label(i), make_correct_label(s)))
            
            if(j == 0){
                iter_img = slice_img[,,1]
            } else if(j == 1){
                iter_img = slice_img[,,2]
            } else if(j == 2){
                iter_img = slice_img[,,3]
            }
            
            dim(iter_img) = c(dim(iter_img), 1)
            dim(slice_lbl) = c(dim(slice_lbl), 1)
            # str(as.nifti(iter_img))
            
            writeNIfTI(
                nim = as.nifti(iter_img),
                filename = temp_im_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )
            writeNIfTI(
                nim = as.nifti(slice_lbl),
                filename = temp_lbl_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )
        }
        # print(paste0("train sample #:" , i, " || slice #:", s))
        
        temp_im_name = paste0("case_", paste0(make_correct_label(i), make_correct_label(s)), ".nii.gz")
        temp_lbl_name = paste0("case_", paste0(make_correct_label(i), make_correct_label(s)), ".nii.gz")
        train = I(list(
            "image" = paste0("./imageTr/", temp_im_name),
            "label" = paste0("./labelTr/", temp_lbl_name)
        ))
        
        json_train[[length(json_train)+1]] = train
        print(train$image)
    }
}

# write json to summarize the data
# json_train
# json_train = list()
# for(i in 0:2){#(length(train_input_img)-1)){
# 
#     temp_img = readNIfTI(fname = paste0(orig_file_path, "/imagesTr/", train_input_img[i+1]), reorient = F)
#     
#     for(s in 1:dim(temp_img)[3]) {
#         temp_im_name = paste0("case_", paste0(make_correct_label(i), make_correct_label(s)), ".nii.gz")
#         temp_lbl_name = paste0("case_", paste0(make_correct_label(i), make_correct_label(s)), ".nii.gz")
#         train = I(list(
#             "image" = paste0("./imageTr/", temp_im_name),
#             "label" = paste0("./labelTr/", temp_lbl_name)
#         ))
# 
#         json_train[[length(json_train)+1]] = train
#         print(train$image)
#     }
# }
# message(paste0("Generate json train"))

json_list = list(
    name = "KiTS2021_2D_RGB", 
    description = "see https://kits21.kits-challenge.org/ additional tranformation was preformed: each slice of each sample was transformed to an RGB image with imagenet preprocessing",
    reference = "https://www.sciencedirect.com/science/article/abs/pii/S1361841520301857, https://kits21.kits-challenge.org/",
    licence = "NA",
    release = "0",
    tensorImageSize = "4D",
    modality = I(list( 
        `0` = "R",
        `1` = "G",
        `2` = "B"
    )),
    labels = I(list(
        `0` = "background",
        `1` = "kidney",
        `2` = "tumor",
        `3` = "cyst"
    )),
    numTraining = length(train_input_img),
    numTest = 0,
    training = json_train,
    test = I(list())
)

json_list = jsonlite::toJSON(json_list, auto_unbox = TRUE, pretty = TRUE)

write_json_data_index(json_list, paste0(task_dir, "/dataset", ".json"))
message(paste0("Save json dataset"))
