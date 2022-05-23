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
TASK = 2
source(paste0(RelPath, "/Scripts/FunctionCompilation/PreprocessingAndTransformation.r"))

nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
pred_folder = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task00", TASK, "_A549/nnUNetTrainerV2_200ep__nnUNetPlansv2.1/fold_")
label_folder = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/labelsTr")

arraynum = function(image){
    return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

# image_data = tibble()

for(i in c(0, 1, 2, 3, 4)){
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
                Y_hat = map(Y_hat, ~ add_dim(.x, 1))
                )
    # image_data = rbind(image_data, image_data_temp)
    # image_data = image_data_temp

    Y_val = list2tensor(image_data$Y)
    Y_hat_val = list2tensor(image_data$Y_hat)
    
    ## display one representative image
    display(paintObjects(Y_val[2,,,], rgbImage(green = Y_hat_val[2,,,]), col = 'red', thick = T))
    
    ## DICE COEFICIENT
    dice_coef <- function(y_true, y_pred, smooth = 1) {
      y_true_f <- k_flatten(y_true)
      y_pred_f <- k_flatten(y_pred)
      intersection <- k_sum(y_true_f * y_pred_f)
      k_mean((2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth))
    }
    
    dice_metric <- function(y_true, y_pred){
      if(!is.list(y_true)){
        num_imgs <- dim(y_true)[1]
      }
      scores <- array(0, num_imgs)
      for(i in 1:num_imgs){
          scores[i] = dice_coef(y_true[i,,], y_pred[i,,]) %>% as.numeric()
      }
      return(scores)
    }
    
    Save_dir = paste0(pred_folder, i)
    # Save_plot_semantic = paste0(Save_dir, "/Plot_semantic")
    # dir.create(Save_plot_semantic, showWarnings = F)
    # Save_image_semantic = paste0(Save_dir, "/Image_semantic")
    # dir.create(Save_image_semantic, showWarnings = F)
    Save_plot_instance = paste0(Save_dir, "/Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
    Save_image_instance = paste0(Save_dir, "/Image_instance")
    dir.create(Save_image_instance, showWarnings = F)

    # dice_val = dice_metric(Y_val[,,,1], Y_hat_val[,,,1])
    # median(dice_val)
    
    # ## val
    # png(paste0("~/Desktop", "/nnUNet_DICE_2_Boxplot_val_raw_postproc",
    #           # "fold", i,
    #           ".png"), width = 800, height = 1200, res = 300)
    # boxplot(dice_val, 
    #         main = c(paste0("F1 score \nmedian = ", format(round(median(dice_val, na.rm = T), 3), nsmall = 3))),
    #         xlab = "Validation",
    #         ylab = "F1 score per image",
    #         ylim = c(0, 1))
    # dev.off()

    ## generate plots for single dice (boxplot dice instances, lin regression)
    ## generate rep. images for instance generation
    source(paste0(RelPath, "/Scripts/FunctionCompilation/DICE_single.r"))


}
  # image_data
  


# }
