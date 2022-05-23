library(EBImage)
options(EBImage.display = "raster")
library(tidyverse)
library(tiff)
library(doParallel)
  

# Settings#############################################################################################################################################
## rel path requires full directory path because of tiff saving functions from EBImage
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/BFSeg', '~/Documents/BFSeg')
CELL = "A549"                         # c("THP1", "A549", "RPE1", "HELA", "MDM")
Unet_dir = paste0(RelPath, "/BF_Data/", CELL)
Train_dir = paste0(Unet_dir, "/Train")
dir.create(Train_dir, showWarnings = F)
TRAIN_PATH = Train_dir
HEIGHT = 1000   # 512     # make bigger
WIDTH = 1000    # 512     # make bigger
CHANNELS = 1    # only grayscale
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
kernel_shape = "disc"
# Param for eroding individual masks
THICKNESS = 5
# SHAPE = "disc"

currenti = paste0("noBorder_shape", kernel_shape, "_thickness", THICKNESS, "_old")
segm_dir = paste0(Train_dir, "/", currenti)
dir.create(segm_dir, showWarnings = F)
# crop_range <- 501:(501+512-1)
crop_range <- 1000

source(paste0(RelPath, "/Scripts/FunctionCompilation/FunctionCompilation_Kernels_Encoding.r"))


## Prepare training images and metadata
ImageMaskTrain <- tibble::tibble(ImageId = list.files(paste0(Unet_dir, "/Image"), pattern = ".tif", recursive = T))
ImageTrain <- filter(ImageMaskTrain, str_detect(ImageId, ""))

train_data <- map_dfr(seq_len(4), ~ ImageMaskTrain) %>%
  ######
  mutate(ImageFile = file.path(paste0(Unet_dir, "/Image"), ImageId), # rep 4 times
         NucEncodedPixels = NA,
         CytoEncodedPixels = NA,
         ImageShape = c(rep_len(list(c(1, crop_range, 1, crop_range)), nrow(ImageMaskTrain)),
                        rep_len(list(c(1, crop_range, crop_range + 1, crop_range * 2)), nrow(ImageMaskTrain)),
                        rep_len(list(c(crop_range + 1, crop_range * 2, 1, crop_range)), nrow(ImageMaskTrain)),
                        rep_len(list(c(crop_range + 1, crop_range * 2, crop_range + 1, crop_range * 2)), nrow(ImageMaskTrain))),
         ShapeIndex = c(rep_len("1 1000 1 1000", nrow(ImageMaskTrain)),
                        rep_len("1 1000 1001 2000", nrow(ImageMaskTrain)),
                        rep_len("1001 2000 1 1000", nrow(ImageMaskTrain)),
                        rep_len("1001 2000 1001 2000", nrow(ImageMaskTrain))),
         Plate = 1,
         # Plate = sub(".*data/", "", Plate),
         # Plate = as.numeric(sub("--W.*", "", Plate)),
         ID = ImageId,
         ID = sub(".*Well", "", ID),
         ID = sub("_Point.*", "", ID),
         Position = sub(".*_Point", "", ImageId),
         Position = sub(".*_0", "", Position),
         Position = as.numeric(sub("_.*", "", Position)),
         Channel = sub(".*Channel", "", ImageId),
         Channel = sub("_.*", "", Channel),
         Channel = map(Channel, ~ ifelse(.x == "20x Phase", "Ph2", .x) ),
         Channel = unlist(Channel),
         ZStack = sub(".*ZStack", "", ImageId),
         ZStack = as.numeric(sub("_.*", "", ZStack)),
         ZStack = map(ZStack, ~ ifelse(is.na(.x), 0, .x) ),
         ZStack = unlist(ZStack)
         ######
         )
str(train_data, list.len = 2)
unique(train_data$ZStack)

IdToMap <- function(FileList){
  rN = c(1:16)
  cN = c(1:24)
  cID = c(1:24)
  rID = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P')
  Row <- data.frame(rID, rN)
  ID = data.frame(cID, cN)
  ID$rID <- NA
  for(i in rID){
    IDtemp <- ID[1:24,]
    IDtemp$rID <- as.character(i) 
    ID <- rbind(ID, IDtemp)
  }
  ID <- merge(ID[25:nrow(ID),], Row, by = "rID", all.x = T)
  ID$ID <- paste(ID$rID, ID$cID, sep = "")
  WellID <- unique(ID$ID)
  for (i in 1:length(WellID)) {
    if(nchar(WellID[i]) == 2){
      ID$ID[i] <- paste0(substring(WellID[i], 1, 1),
                                 "0",
                                 substring(WellID[i], 2, 2))
    }
  }
  FileList <- left_join(x = FileList, y = ID, by = c("ID"), all.x = T)
}
train_data <- IdToMap(train_data)
train_data %>% glimpse()
# train_data <- train_data %>% filter(ID %in% c("A2", "A4", "A6", "A11",
#                                               "B03", "B02", "B04", "B10",# B02 B04 B10 not control but looks fine
#                                               # "C21", # out of focus
#                                               "D23",
#                                               "F18", "F20",
#                                               "G13", "G14",
#                                               "H13", "H14", "H15", "H22",
#                                               "I24",
#                                               "J06",
#                                               "K14",
#                                               "M01", "M20",
#                                               "O13", "O19", "O23",
#                                               "P08", "P20"))

unique(train_data$ID) # check empty wells on Ben's layout
length(unique(train_data$ID))

# train_data = filter(train_data, ID %in% c("E03", "E04"))

# rm(Plate)
# rm(Well)
# rm(Position)
# rm(Shape)
# rm(Boolean)

# Read images and extract features#############################################################################################
ImageProcessingAndEncoding <- function(Image, Plate, Well, Position, Shape){
  
  ##For testing only:
  # Plate = 1
  # Well = "E03"
  # Position = 3
  # Shape = "1 1000 1001 2000"
  Boolean <- train_data$Plate == Plate &
    train_data$ID == Well &
    train_data$Position == Position &
    train_data$ShapeIndex == Shape
 
  #---------------Rescale pixel intensity-----------
  ######
  # Rescale <- function(x){
  #   Bottom = 2^15/(2^16-1) 
  #   Top = (2^15+4095)/(2^16-1)
  #   (x - Bottom) / (Top - Bottom)   
  # }
  ######
  # CROP = unlist(shape)
  # CROP = as.numeric(unlist(str_split(unique(train_data$ShapeIndex)[1], " ", 4)))
  CROP = as.numeric(unlist(str_split(Shape, " ", 4)))
  
  #---------------Load Images------------------------
  Img405 = readImage(as.character(train_data[train_data$Channel == "DAPI" &
                                               Boolean, "ImageFile"]))[CROP[1]:CROP[2], CROP[3]:CROP[4]]
  NImg405 = EBImage::normalize(Img405) * 2
  col405 = toRGB(NImg405)
  
  ImgTL0 = readImage(as.character(train_data[train_data$Channel == "DIA" &
                                                train_data$ZStack == 5 &
                                                Boolean, "ImageFile"]))[CROP[1]:CROP[2], CROP[3]:CROP[4]]
  NImgTL0 = EBImage::normalize(ImgTL0)
  colTL0 = toRGB(NImgTL0)
      
  #---------------smooth and threshold nuleus--------------------------------
  ## Blur slightly for better foreground detection 
  Img405smooth = Gaus(Img405, 25, 2)
  nucThrManual = thresh(Img405smooth, w = 201, h = 201, offset = 1e-3)
  # DPO(nucThrManual, NImg405)

  ## removal of small objects and smoothes out objects
  nucOpening = opening(nucThrManual, kern = makeBrush(11, shape = "disc"))
  nucOpening = bwlabel(nucOpening)
  nucOpening = fillHull(nucOpening)
  nucOpening = erode(nucOpening, kern = makeBrush(size = 1, shape = c("disc")))
  #  display(paintObjects(nucOpening, col405 * 2, col = "red"), method = "raster")
  boxEr = indObj(bwlabel(nucOpening), "erode", 11, "disc", label = T)
  # watTest = indWat(boxErTest2, split = T)
  
  # DPO(boxEr, col405)
  boxWat = watershed(distmap(boxEr), 1)
  #  display(colorLabels(boxWat))
  #  display(col405)
  #  DPO(boxWat, col405)
  boxEr2 = indObj(boxWat, "erode", 5, "disc", label = T)
   # DPO(boxEr2, col405)
  boxEr3 = indObj(boxEr2, "erode", 5, "disc", label = T)
  # boxEr3 = erode(boxEr2 , kern = makeBrush(size = 5, shape = c("disc")))
  boxEr3op = indObj(boxEr2, "opening", 7, "gaussian", label = T)
   # DPO(boxEr3op, col405)
  # boxEr3op = bwlabel(boxEr3op, label = T)
  boxProp = propagate(Img405, boxEr3op, mask = nucOpening, lambda = 1e-8)
   # DPO(boxProp, col405)
  # ini param = 1
  # cellEr = indObj(boxProp, "erode", 1, "gaussian", label = F)
  # DPO(cellEr, col405)
  maskEr = SingObjOp(boxProp, "erode", 1, "gaussian")
   # DPO(cellEr, col405)
  maskErOp = SingObjOp(maskEr, "opening", 7, "gaussian")
  mask = maskErOp
    DPO(mask, col405)
  # cellErOpDi = SingObjOp(cellErOp, "dilate", 2, "gaussian")
  # DPO(cellErOpDi, col405)
  
  ## to include for deep watershed transform
  # DM = SingObjUnit(cellErOp, unit = F)
  # DM = indWat(mask, unit = F, split = F)
  # unit_DM = SingObjUnit(cellErOp)
  # unit_DM = indWat(mask, unit = T, split = F)
  
 mmt = as_tibble(computeFeatures.moment(mask, Img405)) %>% 
      select(m.cx, m.cy) %>% 
      round() %>% 
      mutate(m.cx = as.integer(m.cx),
             m.cy = as.integer(m.cy))
    
    center = array(0, dim = dim(Img405))
    for (i in 1:dim(center)[1]) {
      center[mmt$m.cx[i], mmt$m.cy[i]] = 1
    }
    # display(ifelse(mask > 0, 1, 0) - center)
    # DPO(cellErOp, col405)
    centerDil3 = dilate(center, kern = makeBrush(3, shape = "gaussian")) %>% 
      as.Image()

    mask_m3 = indObj(mask, "erode", 3, "disc")
    mask_m5 = indObj(mask, "erode", 5, "disc")
    mask_m7 = indObj(mask, "erode", 7, "disc")
    border = ifelse(mask > 0, 1, 0) - ifelse(mask_m3 > 0, 1, 0) 
    # display(border)
    
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############
    ############
    # generated 3 masks: -3 -5 -7 pixels eroded masks                             easy done
    # border (probably mask - mask_erode_3_pixel or mask_dilate_3_pixel - mask)   easy done
    # center                                                                      diff done
    # unit vector                                                                 diff done
    
    nucDilate = dilate(mask, kern = makeBrush(3, shape = "disc"))
    #display(combine(colorLabels(nucMask), colorLabels(nucDilate)), all = T)
    nucBorder = nucDilate - mask
     # display(colorLabels(nucBorder))
    InterfaceBorderNuc = ifelse(mask > 0, 1, 0) - ifelse(nucBorder > 0, 2, 0)
    InterfaceBorderNuc = ifelse(InterfaceBorderNuc == -2, 2, InterfaceBorderNuc)
    InterfaceBorderNuc = ifelse(InterfaceBorderNuc == -1, 3, InterfaceBorderNuc)
    #  display(InterfaceBorderNuc == 3 )
    
    Segm405 = paintObjects(mask, col405, col = 'red', thick = F)
    SegmTL0 = paintObjects(mask, colTL0, col = 'red', thick = F)
    
    writeImage(Segm405, files = paste0(segm_dir,
                                       "/Segm405_",
                                       "Plate_", Plate,
                                       "--Well_", Well,
                                       "--Position_", Position,
                                       "--Shape_", Shape,
                                       currenti, ".tiff"),
               quality = 100, type = "tiff")
    
    writeImage(SegmTL0, files = paste0(segm_dir,
                                       "/SegmTL0_",
                                       "Plate_", Plate,
                                       "--Well_", Well,
                                       "--Position_", Position,
                                       "--Shape_", Shape,
                                       currenti, ".tiff"),
               quality = 100, type = "tiff")
    
    dim(InterfaceBorderNuc) <- c(dim(InterfaceBorderNuc), 1)
    dim(mask) <- c(dim(mask), 1)
    dim(mask_m3) <- c(dim(mask_m3), 1)
    dim(mask_m5) <- c(dim(mask_m5), 1)
    dim(mask_m7) <- c(dim(mask_m7), 1)
    dim(border) <- c(dim(border), 1)
    dim(center) <- c(dim(center), 1)
    dim(centerDil3) <- c(dim(centerDil3), 1)
    # dim(unit_DM) <- c(dim(unit_DM), 1)
    # dim(nucMask) <- c(dim(nucMask), 1)
  
  postprocess_image(as.array(ifelse(InterfaceBorderNuc == 3, 1, 0)), c(1000, 1000))
  postprocess_image(as.array(ifelse(InterfaceBorderNuc == 2, 1, 0)), c(1000, 1000))


    ## Encoding nucleus #####################################################################################
  n_sample_nuc <- as.tibble(train_data[Boolean & train_data$Channel == "DAPI",]) %>% 
    mutate(ActualShape = list(c(HEIGHT, WIDTH)),
           BorderEncodedPixels = list(as.array(ifelse(InterfaceBorderNuc == 2, 1, 0))),
           InterfaceEncodedPixels = list(as.array(ifelse(InterfaceBorderNuc == 3, 1, 0))),
           MaskEncodedPixels = list(as.array(ifelse(InterfaceBorderNuc == 1, 1, 0))),
           Mask = list(as.array(ifelse(mask > 0, 1, 0))),
           Mask_m3 = list(as.array(mask_m3)),
           Mask_m5 = list(as.array(mask_m5)),
           Mask_m7 = list(as.array(mask_m7)),
           Border = list(as.array(border)),
           Center = list(as.array(centerDil3))
           ) %>%
    mutate(BorderEncodedPixels = map2(BorderEncodedPixels, ActualShape, postprocess_image),
           InterfaceEncodedPixels = map2(InterfaceEncodedPixels, ActualShape, postprocess_image),
           MaskEncodedPixels = map2(MaskEncodedPixels, ActualShape, postprocess_image),
           Mask = map2(Mask, ActualShape, postprocess_image),
           Mask_m3 = map2(Mask_m3, ActualShape, postprocess_image),
           Mask_m5 = map2(Mask_m5, ActualShape, postprocess_image),
           Mask_m7 = map2(Mask_m7, ActualShape, postprocess_image),
           Border = map2(Border, ActualShape, postprocess_image),
           Center = map2(Center, ActualShape, postprocess_image),
           Crop = Shape) %>% 
    unnest(#c(
             BorderEncodedPixels,
             InterfaceEncodedPixels,
             MaskEncodedPixels,
             Mask,
             Mask_m3,
             Mask_m5,
             Mask_m7,
             Border,
             Center#)
    ) %>% 
    select(ImageId, BorderEncodedPixels, InterfaceEncodedPixels, MaskEncodedPixels,
           Mask, Mask_m3, Mask_m5, Mask_m7,
           Border, Center,
           Crop)
    # select(-BorderMasks, -InterfaceMasks, -Masks, -ImageShape)   
  # display(preprocess_masks(n_sample_nuc$Mask, c(1000, 1000), c(1000, 1000)))
    # n_sample_nuc %>% glimpse()
  
    # display(preprocess_masks(n_sample_out_nuc$Mask, c(1000, 1000), c(1000, 1000)))
    ## Export encoding
    write_csv(n_sample_nuc, paste0(segm_dir,
                                   "/Plate_", Plate,
                                   "--Well_", Well,
                                   "--Position_", Position,
                                   "--Shape_", Shape,
                                   "--n_sample_out_nuc_", currenti,".csv"))

}

#----------------Run ImageProcessing() to everyfield---------------------------------------------------------------------------------------------------
   
registerDoParallel(8)
# library(doMC)
# registerDoMC(4)

for (Plate in unique(train_data[, ]$Plate)) {
  for(Well in unique(train_data[train_data$Plate == Plate, ]$ID)) {
    ptm <- proc.time()
    # for(Position in unique(train_data[train_data$Plate == Plate &
                                    #   train_data$ID == Well, ]$Position)) {
      ptm <- proc.time()
     foreach(Position = unique(train_data$Position), .packages = c("EBImage", "tidyverse")) %:%
      foreach (Shape = unique(train_data[train_data$Plate == Plate &
                                           train_data$ID == Well &
                                           train_data$Position == Position, ]$ShapeIndex), .packages = c("EBImage", "tidyverse")) %dopar% {
        # for (Shape in unique(train_data$ShapeIndex)) {
        # print(paste0("START -- Plate: ", Plate, "; Well: ", Well, "; Position:", Position,  "; Crop:", Shape,"; Time: ", Sys.time()))
        # Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position & train_data$ShapeIndex == Shape
        
        # run image proc and encoding for 1 field of view
        ImageProcessingAndEncoding(Image = train_data,
                                   Plate = Plate,
                                   Well = Well,
                                   Position = Position,
                                   Shape = Shape)
        # 
        # check iteration
        # print(paste0("END   -- Plate: ", Plate, "; Well: ", Well, "; Position:", Position,  "; Crop:", Shape,"; Time: ", Sys.time()))
      }
      print(paste0("Computation time: ", round((proc.time() - ptm)[3] , 2),
                    " s || Current condition:",  
                    " -- Well: ", Well#,
                    # " -- Position: ", Position
                    ))
    # }
  }
}

# as.numeric(unlist(str_split(unique(train_data$ShapeIndex)[1], " ", 4)))
