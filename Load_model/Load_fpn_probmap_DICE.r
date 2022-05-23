library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)
library(tensorflow)

# set to run tensorflow on cpu if GPU is busy
# tf$config$set_visible_devices(list(), 'GPU')

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
TASK = 2
source(paste0(RelPath, "/Scripts/FunctionCompilation/PreprocessingAndTransformation.r"))

model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_2Class--5/fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_1")
# model_folder = paste0(RelPath, "/BF_Data/A549/Prediction/ImgBF512_1Class--36/fpn--seresnext101--FOLD_1")
image_folder = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/imagesTs")
label_folder = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", TASK, "_A549/labelsTs")

arraynum = function(image){
  return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

# Load custom functions
list_scripts = list("Params_nnUNet_comparaison.r", "PreprocessingAndTransformation.r",
                    "Model_FPN.r", "Model_UNET.r", "Model_PSPNET.r", "Model_Deeplabv3plus.r",
                    "Backbone_Zoo.r",
                    "Inspection.r",
                    "Postprocessing.r") %>%
  map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
  source(list_scripts[[l]])
}

Save_dir = paste0(model_folder, "/NewPlot")
dir.create(Save_dir)

weight_name = paste0(
  # "best_",
  "model_weights.hdf5")

# build the model
enc = ENCODER 
redfct = 8L
arc = ARCHITECTURE
dropout = DROPOUT
source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
# model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
#                   backbone = ENCODER,
#                   nlevels = NULL,
#                   output_activation = ACTIVATION,
#                   output_channels = CLASS,
#                   decoder_skip = FALSE,
#                   dropout = DROPOUT
# )
# if model saved
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
model %>% load_model_weights_hdf5(filepath = paste0(model_folder, "/", weight_name))

image_data = tibble()

# for(i in c(1)){
image_data_temp <- tibble(Image_Path = list.files(paste0(image_folder), recursive = T),
                          Label_Path = list.files(paste0(label_folder), recursive = T)) %>%
  filter(str_detect(Image_Path, "nii")) %>%
  mutate(Image_Path = paste0(image_folder, "/", Image_Path),
         Label_Path = paste0(label_folder, "/", Label_Path),
         X = map(Image_Path, ~ readNIfTI(fname = .x)),
         X = map(X, ~ arraynum(.x)),
         X = map(X, ~ add_dim(.x, 1)),
         X = map(X, transform_gray_to_rgb_rep),
         X = map(X, EBImage::normalize, separate = T, ft = c(0, 255)),
         X = map(X, imagenet_preprocess_input, mode = "torch"),
         Y = map(Label_Path, ~ readNIfTI(fname = .x)),
         Y = map(Y, ~ arraynum(.x)),
         Y = map(Y, to_categorical),
         Class = CLASS,
         Y = map_if(Y, Class == 1, ~ add_dim(.x, 1))
         # Y_hat = map(Y_hat, flip),
         # Y_hat = map(Y_hat, ~ add_dim(.x, 1))
  )
# image_data = rbind(image_data, image_data_temp)
image_data = image_data_temp
# }
# image_data
# str(image_data$X)
# str(image_data$Y)

X = list2tensor(image_data$X)
Y = list2tensor(image_data$Y)

pred_batch_size = 1

Y_hat = predict(model,
                X,
                batch_size = pred_batch_size)
str(Y_hat)

## display one representative image
display(abind(Y[1,,,2], normalize(X[1,,,1]), along = 1))
display(abind(
  paintObjects(Y[1,,,2], rgbImage(green = Y_hat[1,,,2]), col = 'red', thick = T),
  paintObjects(Y[1,,,3], rgbImage(green = Y_hat[1,,,3]), col = 'red', thick = T),
  along = 1),
  bg = "black")

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

## test
for(i in ifelse(ACTIVATION == "softmax", 2, 1):ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS)){
  
  dice_test = dice_metric(Y[,,,i], Y_hat[,,,i])
  test_dice_med = format(round(median(dice_test, na.rm = T), 3), nsmall = 3)
  
  png(paste0("~/Desktop", "/FPN_DICE_2_CLASS_", ,"_Boxplot_test",
            # "fold", i,
            ".png"), width = 800, height = 1200, res = 300)
  boxplot(dice_test, 
          main = c(paste0("F1 score \nmedian = ", test_dice_med)),
          xlab = "Test",
          ylab = "F1 score per image",
          ylim = c(0, 1.0))
  dev.off()

  assign(paste0("test_iou_class_", ifelse(ACTIVATION == "softmax", i-1, i)), test_dice_med)
}
test_iou_class_1
test_iou_class_2
test_iou_class_3
  

# load original image crop it
origimage = readImage(files = "/home/gendarme/Documents/UNet/BF_Data/A549/Image/20210126_142039_038/WellE03_PointE03_0000_ZStack0004_ChannelDIA_Seq0005.tiff")
smallimage = origimage[1:512, 1:512]
display(normalize(smallimage))  

# write the different images on disk
writeNIfTI(smallimage,
           filename = "~/Desktop/testnifti_comp1",
           onefile = TRUE,
           gzipped = TRUE,
           verbose = FALSE,
           warn = -1,
           compression = 1)
writeNIfTI(smallimage,
           filename = "~/Desktop/testnifti_comp0",
           onefile = TRUE,
           gzipped = TRUE,
           verbose = FALSE,
           warn = -1,
           compression = 0)
writeNIfTI(smallimage,
           filename = "~/Desktop/testnifti_comp9",
           onefile = TRUE,
           gzipped = TRUE,
           verbose = FALSE,
           warn = -1,
           compression = 9)
writeImage(smallimage,
           files = "~/Desktop/testtiff.tiff",
           type = "tiff",
           quality = 100)

# load back the images to compare
testnifti_comp0 = readNIfTI("~/Desktop/testnifti_comp0")
testnifti_comp1 = readNIfTI("~/Desktop/testnifti_comp1")
testtiff = readImage("~/Desktop/testtiff.tiff")

crop = 256
combined = normalize(abind(testtiff[1:crop,1:crop],
                           testnifti_comp0[1:crop,1:crop],
                           testnifti_comp1[1:crop,1:crop],
                           along = 1))
display(combined)


img_data = tibble(
  tiff = array(testtiff, dim = dim(testtiff)[1]*dim(testtiff)[2]),
  nifti_0 = array(testnifti_comp0, dim = dim(testnifti_comp0)[1]*dim(testnifti_comp0)[2]),
  nifti_1 = array(testnifti_comp1, dim = dim(testnifti_comp1)[1]*dim(testnifti_comp1)[2])
)

apply(img_data, 2, mean)

img_data_reshape = tibble()

for(i in 1:ncol(img_data)){
  temp = tibble(Value = select(img_data, i) %>% pull,
                Image = as.character(colnames(img_data[,i])))
  img_data_reshape = rbind(img_data_reshape, temp)
}


img_hist_plot = 
  ggplot(img_data_reshape, aes(x = Value, fill = Image)) + 
  geom_histogram(position = "identity", alpha = 0.2, bins = 50) #+
# scale_x_continuous(limits = 0.1, 0.15)

img_hist_plot

ggsave(filename = paste0("img_hist_loss_caffe.png"), plot = img_hist_plot,
       width = 6, height = 6, dpi = 1000, path = "~/Desktop/")



