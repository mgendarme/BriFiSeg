library(tidyverse)
library(oro.nifti)
library(EBImage)

# path
## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
source(paste0(RelPath, "/Scripts/FunctionCompilation/Inspection.r"))
combine_col
# TASK = 17
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
Save_dir = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/CellLines")
dir.create(Save_dir, showWarnings = F)

# A549
image_A549 = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task001_A549/imagesTr")
label_A549 = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task001_A549/labelsTr")
# HELA
image_HELA = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task010_HELA/imagesTr")
label_HELA = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task010_HELA/labelsTr")
# MCF7
image_MCF7 = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task011_MCF7/imagesTr")
label_MCF7 = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task011_MCF7/labelsTr")
# RPE1
image_RPE1 = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task015_RPE1/imagesTr")
label_RPE1 = paste0(nnunetpath, "/nnUNet_raw/nnUNet_raw_data/Task015_RPE1/labelsTr")

# set saving directories for plots 
Save_image = paste0(Save_dir, "/Image_montage")
dir.create(Save_image, showWarnings = F)


get_id_well_label = function(data){
    return(
        data %>% 
            mutate(
                Ref = sub(".*/", "", path),
                ID = sub(".*_", "", Ref),
                # ID = as.numeric(sub("_.*", "", ID)))
                ID = as.numeric(sub(".nii.gz.*", "", ID))
    )
    )
}

get_id_well_image = function(data){
    return(
        data %>% 
            mutate(
                Ref = sub(".*/", "", path),
                ID = sub("_0000.nii.gz.*", "", Ref),
                # ID = sub("*_.", "", Ref))#,
                ID = as.numeric(sub(".*_", "", ID)))#,
                # ID = sub(".*_", "", ID))#,
                # ID = as.numeric(sub(".nii.gz.*", "", ID)))
    )
}

image = tibble()
label_true = tibble()

for(i in c("A549", "HELA", "MCF7", "RPE1")){
    # bind all data together 
    image_temp = tibble(path = c(list.files(path = get(paste0("image_", i)), pattern = ".nii.gz", full.names = TRUE)))
    label_true_temp = tibble(path = c(list.files(path = get(paste0("label_", i)), pattern = ".nii.gz", full.names = TRUE)))

    image_temp = get_id_well_image(image_temp) %>%
        mutate(Cell = i)
    label_true_temp = get_id_well_label(label_true_temp) %>%
        mutate(Cell = i)

    image = rbind(image, image_temp)
    label_true = rbind(label_true, label_true_temp)
}

unique(image$Cell)

arraynum = function(image){
        return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

# pred2label = function(image, thresh){
#   pred_label_init = image > thresh
#   pred_label_temp = erode(pred_label_init, kern = makeBrush(size = 15, shape = c("disc")))
#   pred_label_temp = bwlabel(pred_label_temp)
#   pred_label_temp_ws = watershed(distmap(pred_label_temp))
#   pred_label_temp2 = propagate(x = image, seeds = pred_label_temp_ws, mask = image > thresh)
#   return(pred_label_temp2)
# }

make_montage = function(img, true, cell, Id, border = FALSE){

    # for testing
    # img = image
    # true = label_true
    # pred = label_pred
    # Id = 1
    # Well = "C07"

    source = readNIfTI(fname = img %>% filter(Cell == cell & ID == Id) %>% slice_head(n=1) %>% select(path) %>% pull)
    gt = readNIfTI(fname = true %>% filter(Cell == cell & ID == Id) %>% select(path) %>% pull)
    # pd = readNIfTI(fname = pred %>% filter(Cell == cell & ID == Id) %>% select(path) %>% pull)

    source = arraynum(source)
    gt = arraynum(gt)
    # pd = arraynum(pd)

    gt = bwlabel(gt)
    # pd = EBImage::flip(pred2label(pd, 0.5))

    source_rgb = combine_col(image_1 = EBImage::normalize(source),
                             color_1 = "gray",
                             dimension = dim(source))
    
    if(border == FALSE){
        im1 = source_rgb + colorLabels(gt)/6
        # im2 = source_rgb + colorLabels(pd)/3.5
    } else {
        im1 = paintObjects(gt, source_rgb, col = "red")
        # im2 = source_rgb + colorLabels(pd)/3.5
    }
    

    montage = abind(source_rgb, im1, 
    # im2,
     along = 1)
    writeImage(montage, files = paste0(Save_image, "/", "Montage_", cell,"_",Id,"_", ifelse(border == TRUE, "border", "mask"), ".png"),
                    quality = 100, type = "png")

}

# idx = 2
for(idx in 1:10){

    print(idx)

    make_montage(image, label_true, "A549", idx)
    make_montage(image, label_true, "A549", idx, border = TRUE)

    make_montage(image, label_true, "HELA", idx)
    make_montage(image, label_true, "HELA", idx, border = TRUE)

    make_montage(image, label_true, "MCF7", idx)
    make_montage(image, label_true, "MCF7", idx, border = TRUE)

    make_montage(image, label_true, "RPE1", idx)
    make_montage(image, label_true, "RPE1", idx, border = TRUE)
}


# drug = list("Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin", "Vehicle")
# generate relevant montage:
# # vehicle
# make_montage(image, label_true, label_pred, "M05", 4)
# # Saturosporin
# make_montage(image, label_true, label_pred, "C20", 1)
# make_montage(image, label_true, label_pred, "D16", 1)
# make_montage(image, label_true, label_pred, "D20", 1)
# # Doxorubicin
# make_montage(image, label_true, label_pred, "E16", 1)
# make_montage(image, label_true, label_pred, "E20", 1)
# # Nocodazole
# make_montage(image, label_true, label_pred, "G16", 1)
# make_montage(image, label_true, label_pred, "G20", 1)
# # Vorinostat
# make_montage(image, label_true, label_pred, "I16", 1)
# make_montage(image, label_true, label_pred, "I20", 1)
# # Nigericin
# make_montage(image, label_true, label_pred, "K20", 1)
# make_montage(image, label_true, label_pred, "K16", 1)
# # Vehicle again
# make_montage(image, label_true, label_pred, "N07", 1)

# library(EBImage)
# library(ggpubr)
# ## for fluorescent images
# mnt = "/media/gendarme/PMI_EXT4/Dataset/RWA_A549/Image"
# Range = 512:(1024)
# # Vehicle
# vehicle = readImage(paste0(mnt, "/", "Plate1_2000_WellM05_Point1_2000_M05_0001_ChannelDAPI_Seq2427.tiff"))[Range, Range]
# # doxorubicin decrease intensity
# doxo = readImage(paste0(mnt, "/", "Plate1_2000_WellE16_Point1_2000_E16_0001_ChannelDAPI_Seq0639.tiff"))[Range, Range]
# # Nocodazole start of massive cell death
# noco = readImage(paste0(mnt, "/", "Plate1_2000_WellG16_Point1_2000_G16_0000_ChannelDAPI_Seq1116.tiff" ))[Range, Range]
# # stauro
# stauro = readImage(paste0(mnt, "/", "Plate1_2000_WellC20_Point1_2000_C20_0000_ChannelDAPI_Seq0204.tiff"))[Range, Range]
# # niger
# niger = readImage(paste0(mnt, "/", "Plate1_2000_WellL20_Point1_2000_L20_0003_ChannelDAPI_Seq2193.tiff"))[Range, Range]


# hist(vehicle)
# montage = normalize(abind(vehicle, stauro, doxo, noco, niger, along = 1), inputRange = c(0.007, 0.019))
# str(montage)
# writeImage(montage, files = paste0(Save_image, "/", "Montage_","Fluo2",".png"),
#                     quality = 100, type = "png")

# img_size_here = 512
# length_condition_here = 5
# img_here = montage
# png(paste0(Save_image, "/Task017_Montage_Fluo", "2.png"),
#     width = img_size_here*(length_condition_here), height = img_size_here + 300, units = "px", pointsize = 1, res = 600)
# plot(img_here, all = T) + 
# #   text(x = seq(img_size_here/2, ((length_condition_here+2) * img_size_here) - (img_size_here/2), img_size_here),
# #        y = rep(img_size_here*2 + 75, (length_condition_here+2)),
# #        labels = c(" ", " ", "UNet \n /"),
# #        cex = 7,
# #        col = "black", 
# #        adj = 0.5) +
#   text(x = seq(img_size_here/2, ((length_condition_here) * img_size_here) - (img_size_here/2), img_size_here),
#        y = rep(512+75, (length_condition_here)),
#        labels = c("Vehicle\ncontrol", "Staurosporin\n125 nM", "Doxorubicin\n125 nM", "Nocodazole\n125 nM", 'Nigericin\n5 µM'),
#        cex = 7,
#        col = "black", 
#        adj = 0.5)
# dev.off()



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

