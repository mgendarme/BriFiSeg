library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts

# set seed for reproducibility
set.seed(11)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')

# Load custom functions
list_scripts = list("Params_nnUNet_comparaison.r", "PreprocessingAndTransformation.r",
                    "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus.r",
                    "Backbone_Zoo.r",
                    # "Loss.r",
                    # "CustomGenerator_CropFix.r",
                    "Inspection.r",
                    "Postprocessing.r") %>%
    map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
    source(list_scripts[[l]])
}

## Z-stack == 4
# PredPath = paste0(RelPath, "/BF_Data/BATCH_ALL/Prediction/ImgBF512_1Class--5/")
# Sub_dir = "--Arch_fpn--Enc_seresnext101--Drop_5--Class_1--LR_1e-04--PT_8--RedFct_8"
## Z-stack == c(3, 4, 5)

PredPath = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_1Class--26/fpn--seresnext101--FOLD_1")
Sub_dir = ""

ModelPath = paste0(PredPath, Sub_dir)
Save_dir = paste0(PredPath, "_NewPlots")
dir.create(Save_dir)

mod_name = paste0("unet_model_", Sub_dir, ".hdf5")
model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                  backbone = ENCODER,
                  nlevels = NULL,
                  output_activation = ACTIVATION,
                  output_channels = CLASS,
                  decoder_skip = FALSE,
                  dropout = DROPOUT
                  )
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)

weight_name = paste0(Sub_dir, "model_weights.hdf5")
model %>% load_model_weights_hdf5(filepath = paste0(ModelPath, "/", weight_name))

# for(i in c(
#     # "A549",   # DONE
#     "HELA",   # DONE
#     "MCF7",   # DONE
#     "RPE1"    # DONE
#     # "BATCH_THP1"  # PROBLEMATIC
#     )){
#     CELL = i
#     Current_i = CELL
    print(message(paste0("\n##################\n",
                    "Current samples: ", CELL,
                    "\n####################")))
    ## load and pre-process data
    source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_nnUNet_comparaison.r"))

    loop_id = paste0(CELL)

    message(paste0("\n######################################################\n",
                    loop_id,
                    "\n######################################################"))

    # build loop dir, plot dir and image dir
    Save_loop = paste0(Save_dir, "/", loop_id) 
    dir.create(Save_loop)
    Save_plot = paste0(Save_loop, "/Plot")
    dir.create(Save_plot)
    Save_image = paste0(Save_loop, "/Image")
    dir.create(Save_image)

    ### Generate single instances and perform measurements #####################################################
    ## generate plots for IOU and DICE
    source(paste0(RelPath, "/Scripts/FunctionCompilation/IOU_DICE_plot.r"))

    ## generate sample images
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images.r"))

    ## generate instance segmentation
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Postprocessing_1c.r"))

    ## generate plots for single IOU comparison ## can take quite some time
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/IOU_single.r"))

# }