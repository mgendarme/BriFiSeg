library(keras)
# use_implementation("tensorflow")
# tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)

# set seed for reproducibility
set.seed(11)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = '/home/gendarme/Documents/UNet'
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
TASK = "Task018_RWA_A549_TEST"

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
list_scripts = list("Params_convert4nnunet.r", "PreprocessingAndTransformation.r") %>%
    map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
    source(list_scripts[[l]])
}

CELL = "A549"

# set path for nnunet training folder
task_dir = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/", TASK) 
# Train_Img_dir = paste0(task_dir, "/imagesTr")
# Train_Lbl_dir = paste0(task_dir, "/labelsTr")
Test_Img_dir = paste0(task_dir, "/imagesTsRWA")
Test_Lbl_dir = paste0(task_dir, "/labelsTsRWA")
for(i in c(task_dir, 
            # Train_Img_dir, Train_Lbl_dir,
            Test_Img_dir, Test_Lbl_dir)){
    dir.create(i)
}

# load data
# source(paste0(Unet_script_dir,"/nnUNet/hela_DataAssembly_nnunet_3channels_cxyz_imagenetPreproc.r"))
ZSTACK = 0
## Prepare training images and metadata
TRAIN_PATH = paste0(Unet_dir,"/RWA_",CELL,"_TEST/Train") 
TEST_PATH = paste0(Unet_dir,"/RWA_",CELL,"_TEST/Test")

# all_sample_out <- map_df(list.files(TRAIN_PATH,
#                                     pattern = "_noBorder_shapedisc_thickness5_rwa.csv",
#                                     full.names = TRUE,
#                                     recursive = TRUE),
#                          read_csv, col_types = cols())
files_list = list.files(TRAIN_PATH,
                        pattern = "_noBorder_shapedisc_thickness5_rwa.csv",
                        full.names = TRUE,
                        recursive = TRUE)
all_sample = tibble()
for(i in files_list){
    print(i)
    tmp = read_csv(i, col_types = cols())
    all_sample = rbind(all_sample, tmp)
}
unique_well = unique(sapply(str_split(sapply(str_split(all_sample$ImageId, "Well"), "[", 2), "_"), "[", 1))
unique_well
all_sample = mutate(all_sample, ID = sapply(str_split(sapply(str_split(ImageId, "Well"), "[", 2), "_"), "[", 1))

for(well in unique_well){

    ptm <- proc.time()

    print(paste0("########################### Well: ", well, " ###########################"))
    WELLS = well

    all_sample_out = all_sample %>% filter(ID == well) %>% select(-c(ID))

    source(paste0(Unet_script_dir,"/nnUNet/DataAssembly_rwa4nnunet_3channels_TESTSET_ONLY.r"))
    # str(input, list.len = 2)

    input_crop = input %>%
        mutate(X = map(X, ~ crop_mut(.x, cropping = 512)),
            X = map(X, EBImage::normalize, separate = T),
            X = map(X, ~ preprocess_input(.x, settings = imagenet_settings)),
            R = map(X, ~ select_channels(.x, 1, 1)),
            G = map(X, ~ select_channels(.x, 2, 2)),
            B = map(X, ~ select_channels(.x, 3, 3)),
            X = map(X, ~ add_dim(.x, 1)),
            R = map(R, ~ add_dim(.x, 1)),
            G = map(G, ~ add_dim(.x, 1)),
            B = map(B, ~ add_dim(.x, 1)),
            Y = map(Y, ~ crop_mut(.x, cropping = 512)),
            Y = map(Y, ~ select_channels(.x, 1, 1)),
            Y = map(Y, ~ add_dim(.x, 1))) %>% 
        select(X, R, G, B, Y, ID)
            # Y = map(Y, ~ add_dim_rgb_nifti(.x, 1)))
    # glimpse(input_crop)

    input$X[[1]][1:3, 1:3, 1:3]
    input_crop$X[[1]][1:3, 1:3, 1:3, 1]

    # Convert to nifti
    input_nifti = input_crop %>%
        mutate(X = as.nifti(X),
            Y = as.nifti(Y))
    # str(input_nifti, list.len = 2)

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

    test_input = input_nifti

    for(i in 1:nrow(test_input)){
        for (j in 0:2) {
            temp_im_name = paste0(Test_Img_dir, "/", well, "_", CELL, "_", make_correct_label(i), "_000", j)
                        
            if(j == 0){
                temp_im = test_input$R[[i]]
            } else if(j == 1){
                temp_im = test_input$G[[i]]
            } else if(j == 2){
                temp_im = test_input$B[[i]]
            }
            
            writeNIfTI(
                nim = temp_im,
                filename = temp_im_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )
            
        }

        temp_lbl_name = paste0(Test_Lbl_dir, "/", well, "_", CELL, "_", make_correct_label(i))
        writeNIfTI(
                nim = test_input$Y[[i]],
                filename = temp_lbl_name,
                onefile = TRUE,
                gzipped = TRUE,
                verbose = FALSE,
                warn = -1,
                compression = 1
            )

        print(paste0("test sample #:" , i))
    }

    print(paste0("Computation time: ", round((proc.time() - ptm)[3] , 2),
                    " s || Current condition:",  
                    " -- Well: ", well#,
                    # " -- Position: ", Position
                    ))

}
###############################################################################################
##################### run in terminal for preprocessing wihtout norm ##########################
# NO PREPROCESSING NECESSARY NNUNET WILL DO THAT AUTOMATICALLY AS WE JUST WANT TO PREDICT 
###############################################################################################