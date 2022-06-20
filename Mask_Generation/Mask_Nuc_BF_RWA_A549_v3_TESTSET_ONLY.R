library(EBImage)
options(EBImage.display = "raster")
library(tidyverse)
library(tiff)
library(doParallel)

# Settings#############################################################################################################################################
## rel path requires full directory path because of tiff saving functions from EBImage
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/GitHub/BriFiSeg', '~/Documents/GitHub/BriFiSeg')
CELL = "RWA_A549"                         # c("THP1", "A549", "RPE1", "HELA", "MDM")
Unet_dir = paste0(RelPath, "/BF_Data/", CELL)
Train_dir = paste0(Unet_dir, "_TEST/Train")
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

currenti = paste0("noBorder_shape", kernel_shape, "_thickness", THICKNESS, "_rwa")
segm_dir = paste0(Train_dir, "/", currenti)
dir.create(segm_dir, showWarnings = F)
ctrl_dir = paste0(Unet_dir, "_TEST/ctrl")
dir.create(ctrl_dir, showWarnings = F)
# crop_range <- 501:(501+512-1)
crop_range <- 1000

source(paste0(RelPath, "/Scripts/FunctionCompilation/Kernels_Encoding.r"))


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
         ZStack = sub("_.*", "", ZStack),
         ZStack = map(ZStack, ~ ifelse(str_detect(.x, "Well") == TRUE, "0000", .x)),
         ZStack = map(ZStack, ~ ifelse(is.na(.x), 0, .x) ),
         ZStack = as.numeric(ZStack),
         ZStack = unlist(ZStack),
         ZStack = 0
         ######
         )

str(train_data, list.len = 2)
unique(train_data$ID)
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

## Filter Data for testing
train_data <- train_data %>% filter(str_detect(ID, "03|04|05|06", negate = TRUE))

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
  # Well = "E06"
  # Position = 1
  # Shape = "1001 2000 1001 2000"
  Boolean <- train_data$Plate == Plate &
    train_data$ID == Well &
    train_data$Position == Position &
    train_data$ShapeIndex == Shape
 

  CROP = as.numeric(unlist(str_split(Shape, " ", 4)))
  
  #---------------Load Images------------------------
  Img405 = readImage(as.character(train_data[train_data$Channel == "DAPI" &
                                               Boolean, "ImageFile"]))[CROP[1]:CROP[2], CROP[3]:CROP[4]]
  NImg405 = EBImage::normalize(Img405) 
  col405 = toRGB(NImg405)
  
  ImgTL0 = readImage(as.character(train_data[train_data$Channel == "DIA" &
                                                train_data$ZStack == 0 &
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
  boxEr = indObj(bwlabel(nucOpening), "erode", 11, "disc", label = T)
  # watTest = indWat(boxErTest2, split = T)
  
  boxWat = watershed(distmap(boxEr), 1)
  boxEr2 = indObj(boxWat, "erode", 5, "disc", label = T)
  boxEr3 = indObj(boxEr2, "erode", 5, "disc", label = T)
  boxEr3op = indObj(boxEr2, "opening", 7, "gaussian", label = T)
  boxProp = propagate(Img405, boxEr3op, mask = nucOpening, lambda = 1e-8)
  maskEr = SingObjOp(boxProp, "erode", 1, "gaussian")
   # DPO(cellEr, col405)
  maskErOp = SingObjOp(maskEr, "opening", 7, "gaussian")
  mask = maskErOp
  
  if(Position == 1){
    writeImage(
      paintObjects(mask, col405, col = "yellow"),
      files = paste0(ctrl_dir,   "/Well_", Well,
                                 "--Position_", Position,
                                 "--Shape_", Shape,
                                 "_Ctrl_seg_image.png")
    )
  }
  
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

  centerDil3 = dilate(center, kern = makeBrush(3, shape = "gaussian")) %>% 
    as.Image()

  dim(mask) <- c(dim(mask), 1)
  dim(center) <- c(dim(center), 1)
  dim(centerDil3) <- c(dim(centerDil3), 1)

  maskm3 = indObj(mask, "erode", 3, "disc")
  maskm5 = indObj(mask, "erode", 5, "disc")
  maskm7 = indObj(mask, "erode", 7, "disc")
  maskm9 = indObj(mask, "erode", 9, "disc")
  
  maskp3 = indObj(mask, "dilate", 3, "disc")
  maskp5 = indObj(mask, "dilate", 5, "disc")
  maskp7 = indObj(mask, "dilate", 7, "disc")
  maskp9 = indObj(mask, "dilate", 9, "disc")

  border3 = ifelse(mask > 0, 1, 0) - ifelse(maskm3 > 0, 1, 0) 
  border5 = ifelse(mask > 0, 1, 0) - ifelse(maskm5 > 0, 1, 0) 
  border7 = ifelse(mask > 0, 1, 0) - ifelse(maskm7 > 0, 1, 0) 
  border9 = ifelse(mask > 0, 1, 0) - ifelse(maskm9 > 0, 1, 0) 

  interface3 = ifelse(maskp3 == 2, 1, 0)
  interface5 = ifelse(maskp5 == 2, 1, 0)
  interface7 = ifelse(maskp7 == 2, 1, 0)
  interface9 = ifelse(maskp9 == 2, 1, 0)

  Segm405 = paintObjects(mask, col405, col = 'red', thick = F)
  SegmTL0 = paintObjects(mask, colTL0, col = 'red', thick = F)

  writeImage(Segm405, files = paste0(segm_dir,
                                     "/Segm405_",
                                     "Plate_", Plate,
                                     "--Well_", Well,
                                     "--Position_", Position,
                                     "--Shape_", Shape,
                                     currenti, ".png"),
             quality = 100, type = "png")
  
  writeImage(SegmTL0, files = paste0(segm_dir,
                                     "/SegmTL0_",
                                     "Plate_", Plate,
                                     "--Well_", Well,
                                     "--Position_", Position,
                                     "--Shape_", Shape,
                                     currenti, ".png"),
             quality = 100, type = "png")
  
  
  # dim(unit_DM) <- c(dim(unit_DM), 1)
  # dim(nucMask) <- c(dim(nucMask), 1)

## Encoding nucleus #####################################################################################
n_sample_nuc <- as_tibble(train_data[Boolean & train_data$Channel == "DAPI",]) %>% 
  mutate(ActualShape = list(c(HEIGHT, WIDTH)),
         Mask = list(as.array(ifelse(mask > 0, 1, 0))),
         Maskm3 = list(as.array(ifelse(maskm3 > 0, 1, 0))),
         Maskm5 = list(as.array(ifelse(maskm5 > 0, 1, 0))),
         Maskm7 = list(as.array(ifelse(maskm7 > 0, 1, 0))),
         Maskm9 = list(as.array(ifelse(maskm9 > 0, 1, 0))),
         Maskp3 = list(ifelse(maskp3 > 0, 1, 0)),
         Maskp5 = list(ifelse(maskp5 > 0, 1, 0)),
         Maskp7 = list(ifelse(maskp7 > 0, 1, 0)),
         Maskp9 = list(ifelse(maskp9 > 0, 1, 0)),
         Border3 = list(as.array(border3)),
         Border5 = list(as.array(border5)),
         Border7 = list(as.array(border7)),
         Border9 = list(as.array(border9)),
         Interface3 = list(as.array(interface3)),
         Interface5 = list(as.array(interface5)),
         Interface7 = list(as.array(interface7)),
         Interface9 = list(as.array(interface9)),
         Center = list(as.array(centerDil3))
         ) %>%
  mutate(Mask = map2(Mask, ActualShape, postprocess_image),
         Maskm3 = map2(Maskm3, ActualShape, postprocess_image),
         Maskm5 = map2(Maskm5, ActualShape, postprocess_image),
         Maskm7 = map2(Maskm7, ActualShape, postprocess_image),
         Maskm9 = map2(Maskp9, ActualShape, postprocess_image),
         Maskp3 = map2(Maskp3, ActualShape, postprocess_image),
         Maskp5 = map2(Maskp5, ActualShape, postprocess_image),
         Maskp7 = map2(Maskp7, ActualShape, postprocess_image),
         Maskp9 = map2(Maskp9, ActualShape, postprocess_image),
         Border3 = map2(Border3, ActualShape, postprocess_image),
         Border5 = map2(Border5, ActualShape, postprocess_image),
         Border7 = map2(Border7, ActualShape, postprocess_image),
         Border9 = map2(Border9, ActualShape, postprocess_image),
         Interface3 = map2(Interface3, ActualShape, postprocess_image),
         Interface5 = map2(Interface5, ActualShape, postprocess_image),
         Interface7 = map2(Interface7, ActualShape, postprocess_image),
         Interface9 = map2(Interface9, ActualShape, postprocess_image),
         Center = map2(Center, ActualShape, postprocess_image),
         Crop = Shape) %>% 
  unnest(Mask,
         Maskm3, Maskm5, Maskm7, Maskm9,
         Maskp3, Maskp5, Maskp7, Maskp9,
         Border3, Border5, Border7, Border9, 
         Interface3, Interface5, Interface7, Interface9,
         Center,
         Crop
  ) %>% 
  select(ImageId,
         Mask,
         Maskm3, Maskm5, Maskm7, Maskm9,
         Maskp3, Maskp5, Maskp7, Maskp9,
         Border3, Border5, Border7, Border9, 
         Interface3, Interface5, Interface7, Interface9,
         Center,
         Crop)

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
                                      # train_data$ID == Well, ]$Position)) {
      # ptm <- proc.time()
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
