library(tidyverse)
library(oro.nifti)
library(EBImage)

# path
## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
source(paste0(RelPath, "/Scripts/FunctionCompilation/Inspection.r"))

TASK = 17
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
Save_dir = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/WS")

# rwa
image_rwa = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK+1, "_RWA_A549_TEST/imagesTsRWA")
pred_rwa = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/imagesTsRWA_predicted_ensemble")
label_rwa = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK+1, "_RWA_A549_TEST/labelsTsRWA")
# ctr
image_ctr = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK, "_RWA_A549/imagesTsCtr")
pred_ctr = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/imagesTsCtr_predicted_ensemble")
label_ctr = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK, "_RWA_A549/labelsTsCtr")
# ts
image_ts = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK, "_RWA_A549/imagesTs")
pred_ts = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/imagesTs_predicted_ensemble")
label_ts = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task0", TASK, "_RWA_A549/labelsTs")

# set saving directories for plots 
Save_image = paste0(Save_dir, "/Image_montage")
dir.create(Save_image, showWarnings = F)

# bind all data together 
image = tibble(path = c(list.files(path = image_rwa, pattern = ".nii.gz", full.names = TRUE),
            list.files(path = image_ctr, pattern = ".nii.gz", full.names = TRUE),
            list.files(path = image_ts, pattern = ".nii.gz", full.names = TRUE)))
label_true = tibble(path = c(list.files(path = label_rwa, pattern = ".nii.gz", full.names = TRUE),
                list.files(path = label_ctr, pattern = ".nii.gz", full.names = TRUE),
                list.files(path = label_ts, pattern = ".nii.gz", full.names = TRUE)))
label_pred = tibble(path = c(list.files(path = pred_rwa, pattern = ".nii.gz", full.names = TRUE),
                list.files(path = pred_ctr, pattern = ".nii.gz", full.names = TRUE),
                list.files(path = pred_ts, pattern = ".nii.gz", full.names = TRUE)))

get_id_well_label = function(data){
    return(
        data %>% 
            mutate(
                Ref = sub(".*/", "", path),
                ID = sub(".*_A549_", "", Ref),
                ID = as.numeric(sub(".nii.gz.*", "", ID)),
                WELL = sub("_A549.*", "", Ref),
                WELL = sub(".nii.gz.*", "", WELL)
                ) %>% 
            mutate(
                WELL = sub(".*A549_", "", WELL),
                WELL = sub("_.*", "", WELL)
            )
    )
}

get_id_well_image = function(data){
    return(
        data %>% 
            mutate(
                Ref = sub(".*/", "", path),
                ID = sub(".*_A549_", "", Ref),
                ID = as.numeric(sub("_.*", "", ID)),
                ID = as.numeric(sub(".nii.gz.*", "", ID)),
                WELL = sub("_A549.*", "", Ref)
                ) %>% 
            mutate(
                WELL = sub(".*A549_", "", WELL),
                WELL = sub("_.*", "", WELL)
            )
    )
}

image = get_id_well_image(image) 
image %>% filter(is.na(ID))
label_true = get_id_well_label(label_true) 
label_pred = get_id_well_label(label_pred) 
# unique(image$WELL)
# unique(label_pred$WELL)

ts_wells = sort(rep(c(paste0(LETTERS[4:9], "05"),
                      paste0(LETTERS[4:9], "06")), 4))
ts_ID = as.character(225:272)

correct_well = function(data){
    for(i in 1:length(ts_ID)){
        # print(paste0(ts_ID[i], " ", ts_wells[i]))
        temp = data %>% 
            filter(WELL == ts_ID[i]) %>%
            mutate(WELL = str_replace(WELL, ts_ID[i], ts_wells[i]))
        data = data %>% 
            filter(WELL != ts_ID[i]) %>%
            rbind(temp)
    }
    return(data)
}

image = correct_well(image) 
label_true = correct_well(label_true) 
label_pred = correct_well(label_pred) 
# unique(image$WELL)
# unique(label_pred$WELL)

arraynum = function(image){
        return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

pred2label = function(image, thresh){
  pred_label_init = image > thresh
  pred_label_temp = erode(pred_label_init, kern = makeBrush(size = 15, shape = c("disc")))
  pred_label_temp = bwlabel(pred_label_temp)
  pred_label_temp_ws = watershed(distmap(pred_label_temp))
  pred_label_temp2 = propagate(x = image, seeds = pred_label_temp_ws, mask = image > thresh)
  return(pred_label_temp2)
}

make_montage = function(img, true, pred, Well, Id){

    # for testing
    # img = image
    # true = label_true
    # pred = label_pred
    # Id = 1
    # Well = "C07"

    source = readNIfTI(fname = img %>% filter(WELL == Well & ID == Id) %>% slice_head(n=1) %>% select(path) %>% pull)
    gt = readNIfTI(fname = true %>% filter(WELL == Well & ID == Id) %>% select(path) %>% pull)
    pd = readNIfTI(fname = pred %>% filter(WELL == Well & ID == Id) %>% select(path) %>% pull)

    source = arraynum(source)
    gt = arraynum(gt)
    pd = arraynum(pd)

    gt = bwlabel(gt)
    pd = EBImage::flip(pred2label(pd, 0.5))

    source_rgb = combine_col(image_1 = normalize(source),
                             color_1 = "gray",
                             dimension = dim(source))
    
    im1 = source_rgb + colorLabels(gt)/3.5
    im2 = source_rgb + colorLabels(pd)/3.5

    montage = abind(source_rgb, im1, im2, along = 1)
    writeImage(montage, files = paste0(Save_image, "/", "Montage_", Well,"_",Id,".png"),
                    quality = 100, type = "png")

}

# drug = list("Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin", "Vehicle")
# generate relevant montage:
for(ii in 1:5){
    # vehicle
    make_montage(image, label_true, label_pred, "M05", ii)
    # Saturosporin
    make_montage(image, label_true, label_pred, "C20", ii)
    make_montage(image, label_true, label_pred, "D16", ii)
    make_montage(image, label_true, label_pred, "D20", ii)
    # Doxorubicin
    make_montage(image, label_true, label_pred, "E16", ii)
    make_montage(image, label_true, label_pred, "E20", ii)
    # Nocodazole
    make_montage(image, label_true, label_pred, "G16", ii)
    make_montage(image, label_true, label_pred, "G20", ii)
    # Vorinostat
    make_montage(image, label_true, label_pred, "I16", ii)
    make_montage(image, label_true, label_pred, "I20", ii)
    # Nigericin
    make_montage(image, label_true, label_pred, "K20", ii)
    make_montage(image, label_true, label_pred, "K16", ii)
    # Vehicle again
    make_montage(image, label_true, label_pred, "N07", ii)
}

library(EBImage)
library(ggpubr)
## for fluorescent images
mnt = "/media/gendarme/PMI_EXT4/Dataset/RWA_A549/Image"
Range = 512:(1024)
# Vehicle
vehicle = readImage(paste0(mnt, "/", "Plate1_2000_WellM05_Point1_2000_M05_0001_ChannelDAPI_Seq2427.tiff"))[Range, Range]
# doxorubicin decrease intensity
doxo = readImage(paste0(mnt, "/", "Plate1_2000_WellE16_Point1_2000_E16_0001_ChannelDAPI_Seq0639.tiff"))[Range, Range]
# Nocodazole start of massive cell death
noco = readImage(paste0(mnt, "/", "Plate1_2000_WellG16_Point1_2000_G16_0000_ChannelDAPI_Seq1116.tiff" ))[Range, Range]
# stauro
stauro = readImage(paste0(mnt, "/", "Plate1_2000_WellC20_Point1_2000_C20_0000_ChannelDAPI_Seq0204.tiff"))[Range, Range]
# niger
niger = readImage(paste0(mnt, "/", "Plate1_2000_WellL20_Point1_2000_L20_0003_ChannelDAPI_Seq2193.tiff"))[Range, Range]


hist(vehicle)
montage = normalize(abind(vehicle, stauro, doxo, noco, niger, along = 1), inputRange = c(0.007, 0.019))
str(montage)
writeImage(montage, files = paste0(Save_image, "/", "Montage_","Fluo2",".png"),
                    quality = 100, type = "png")

img_size_here = 512
length_condition_here = 5
img_here = montage
png(paste0(Save_image, "/Task017_Montage_Fluo", "2.png"),
    width = img_size_here*(length_condition_here), height = img_size_here + 300, units = "px", pointsize = 1, res = 600)
plot(img_here, all = T) + 
#   text(x = seq(img_size_here/2, ((length_condition_here+2) * img_size_here) - (img_size_here/2), img_size_here),
#        y = rep(img_size_here*2 + 75, (length_condition_here+2)),
#        labels = c(" ", " ", "UNet \n /"),
#        cex = 7,
#        col = "black", 
#        adj = 0.5) +
  text(x = seq(img_size_here/2, ((length_condition_here) * img_size_here) - (img_size_here/2), img_size_here),
       y = rep(512+75, (length_condition_here)),
       labels = c("Vehicle\ncontrol", "Staurosporin\n125 nM", "Doxorubicin\n125 nM", "Nocodazole\n125 nM", 'Nigericin\n5 µM'),
       cex = 7,
       col = "black", 
       adj = 0.5)
dev.off()



dat = data.frame(Intensity = c(vehicle, doxo, noco, stauro, niger),
                 Compound = rep(c("Vehicle", "Staurosporin", "Doxorubicin", "Nocodazole", "Nigericin"), each = length(c(vehicle))))
dat$Compound <- factor(dat$Compound, levels = c("Vehicle", "Staurosporin", "Doxorubicin", "Nocodazole", "Nigericin"))


plt = ggdensity(dat, x = "Intensity",
#    add = "mean", rug = TRUE,
   color = "Compound", fill = "Compound",
#    ylim = c(0, 9000),
   xlim = c(0.00585, 0.007),
   ylab = "Density",
   legend = "right",
   alpha = 0.2

#    ,
#    palette = c("#00AFBB", "#E7B800")
   ) +
   scale_x_continuous(breaks = seq(0.006, 0.007, 0.0005))
    # ggplot(dat,aes(x=xx, fill=yy)) + 
    # geom_histogram(fill = yy, alpha = 0.2) +
    # geom_histogram(data=subset(dat,yy == 'Vehicle'),fill = "red", alpha = 0.2) +
    # geom_histogram(data=subset(dat,yy == 'Doxorubicin'),fill = "blue", alpha = 0.2) +
    # geom_histogram(data=subset(dat,yy == 'Nocodazole'),fill = "green", alpha = 0.2) 
    # guides(fill=guide_legend(title=yy))
ggsave(filename = paste0(Save_dir, "/Hist_Fluo_Density.png"),
       plot = plt, width = 5, height = 3)

