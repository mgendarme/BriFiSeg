library(tidyverse)

#folders of experiments to investigate
folder = list(
  'fpn_seresnext101' = '/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgBF512_1Class--60',
  'unet_seresnext101' = '/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgBF512_1Class--100',
  'unet_null' = '/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgBF512_1Class--118'
  )

#loads an RData file, and returns it
loadRData <- function(fileName){
  load(fileName)
  get(ls()[ls() != "fileName"])
}

#generate of means per fold and the mean of the 5 folds per condition
list_mean = list(
  'fpn_seresnext101' = NULL,
  'unet_seresnext101' = NULL,
  'unet_null' = NULL
  )
for (f in 1:length(folder)) {
  files = list.files(folder[[f]], recursive = T) %>% 
    as_tibble() %>% 
    filter(str_detect(value, "rdata") & str_detect(value, "HTR_DICE")) %>%
    pull
  for (i in 1:5) {
    history_dice = loadRData(paste0(folder[[f]], "/", files[i]))
    dice_temp = history_dice$dice_val_post_proc
    list_mean[[f]][[i]] = mean(dice_temp)
  }
  list_mean[[f]]['mean'] = mean(unlist(list_mean[[f]]))
}

# get mean of each condition
cat(paste0(
    'mean dice val unet_null         = ', round(unlist(list_mean$unet_null$mean), 4), "\n",
    'mean dice val fpn_seresnext101  = ' , round(unlist(list_mean$fpn_seresnext101$mean), 4), "\n",
    'mean dice val unet_seresnext101 = ',  round(unlist(list_mean$unet_seresnext101$mean), 4)
))