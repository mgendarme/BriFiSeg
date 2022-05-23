  library(EBImage)
  options(EBImage.display = "raster")
  library(tidyverse)
  library(tiff)
  library(doParallel)
    
  # Settings#############################################################################################################################################
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
  
  currenti = paste0("noBorder_shape", kernel_shape, "_thickness", THICKNESS)
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
  
  # Read images and extract features#############################################################################################
  ImageProcessingAndEncoding <- function(Image, Plate, Well, Position, Shape){
    
    ##For testing only:
    Plate = 1
    Well = "E03"
    Position = 1
    Shape = "1001 2000 1001 2000"
    Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position & train_data$ShapeIndex == Shape
   
    #---------------Rescale pixel intensity-----------
    ######
    # Rescale <- function(x){
    #   Bottom = 2^15/(2^16-1) 
    #   Top = (2^15+4095)/(2^16-1)
    #   (x - Bottom) / (Top - Bottom)   
    # }
    ######
    CROP = unlist(unique(train_data[Boolean, "ImageShape"]))
    
    #---------------Load Images------------------------
    Img405 = readImage(as.character(train_data[train_data$Channel == "DAPI" &
                                                 Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    NImg405 = EBImage::normalize(Img405) * 2
    col405 = toRGB(NImg405)
    #  display(as.array(NImg405))
    # ImgTLm1 = readImage(as.character(train_data[train_data$Channel == "DIA" &
    #                                             train_data$ZStack == 1 &
    #                                             Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    # NImgTLm1 = EBImage::normalize(ImgTLm1)
    ImgTL0 = readImage(as.character(train_data[train_data$Channel == "DIA" &
                                                  train_data$ZStack == 5 &
                                                  Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    NImgTL0 = EBImage::normalize(ImgTL0)
    colTL0 = toRGB(NImgTL0)
    # ImgTLp1 = readImage(as.character(train_data[train_data$Channel == "DIA" &
                                                  # train_data$ZStack == 3 &
                                                  # Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    # NImgTLp1 = EBImage::normalize(ImgTLp1)
    # display(colTL0)   
    # # "fake" rgb image composed of 3 TL images (stack -1, 0, 1 ?m from dapi focal plan)
    # RgbTL = rgbImage(blue = ImgTL0,
    #                  red = ImgTLm1,
    #                  green = ImgTLp1)
    #  # display(normalize(RgbTL))
    
    # ImgPhm1 = readImage(as.character(train_data[train_data$Channel == "Ph2" &
    #                                               train_data$ZStack == 1 &
    #                                               Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    # NImgPhm1 = EBImage::normalize(ImgPhm1)
    # ImgPh0 = readImage(as.character(train_data[train_data$Channel == "Ph2" &
    #                                              train_data$ZStack == 2 &
    #                                              Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    # NImgPh0 = EBImage::normalize(ImgPh0)
    # ImgPhp1 = readImage(as.character(train_data[train_data$Channel == "Ph2" &
    #                                               train_data$ZStack == 3 &
    #                                               Boolean, "ImageFile"]))[CROP[[1]]:CROP[[2]], CROP[[3]]:CROP[[4]]]
    # NImgPhp1 = EBImage::normalize(ImgPhp1)
    
    # # "fake" rgb image composed of 3 PC images (stack -1, 0, 1 ?m from dapi focal plan)
    # RgbPh = rgbImage(blue = ImgPh0,
    #                  red = ImgPhm1,
    #                  green = ImgPhp1)
     # display(normalize(NImgTL0))
    
    #---------------smooth and threshold nuleus--------------------------------
    # Blur slightly for better foreground detection 
    Img405smooth = Gaus(Img405, 25, 2)
    nucThrManual = thresh(Img405smooth, w = 201, h = 201, offset = 1e-3)
    # DPO(nucThrManual, NImg405)
    # removal of small objects and smoothes out objects
    nucOpening = opening(nucThrManual, kern = makeBrush(11, shape = "disc"))
    nucOpening = bwlabel(nucOpening)
    nucOpening = fillHull(nucOpening)
    nucOpening = erode(nucOpening, kern = makeBrush(size = 1, shape = c("disc")))
    #  display(paintObjects(nucOpening, col405 * 2, col = "red"), method = "raster")
    boxEr = indObj(bwlabel(nucOpening), "erode", 11, "disc", label = T)
    # watTest = indWat(boxErTest2, split = T)
     # display(colorLabels(boxEr))
    # display(colorLabels(boxErTest2))
    # str(boxErTest2)
    
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
    #  DPO(mask, col405)
    # cellErOpDi = SingObjOp(cellErOp, "dilate", 2, "gaussian")
    # DPO(cellErOpDi, col405)
    
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
    display(centerDil3)
    maskm3 = indObj(mask, "erode", 3, "disc")
    maskm5 = indObj(mask, "erode", 5, "disc")
    maskm7 = indObj(mask, "erode", 7, "disc")
    maskm9 = indObj(mask, "erode", 9, "disc")
    
    maskp3 = indObj(mask, "dilate", 3, "disc")
    maskp5 = indObj(mask, "dilate", 5, "disc")
    maskp7 = indObj(mask, "dilate", 7, "disc")
    maskp9 = indObj(mask, "dilate", 9, "disc")
    display(maskp9 - mask)
    # border = ifelse(mask > 0, 1, 0) - ifelse(mask_m3 > 0, 1, 0) 
    #  display(border)
    # tt = paintObjects(border7, col405, col = "yellow")
    # # tt = fillHull(tt)
    # display(tt)
    # writeImage(border7, files = paste0("/home/gendarme/Desktop/testseg.tiff"), quality = 100, type = "tiff") 
    border3 = ifelse(mask > 0, 1, 0) - ifelse(maskm3 > 0, 1, 0) 
    border5 = ifelse(mask > 0, 1, 0) - ifelse(maskm5 > 0, 1, 0) 
    border7 = ifelse(mask > 0, 1, 0) - ifelse(maskm7 > 0, 1, 0) 
    border9 = ifelse(mask > 0, 1, 0) - ifelse(maskm9 > 0, 1, 0) 
    # display(ifelse(mask > 0, 1, 0))
    # display(ifelse(mask_m7 > 0, 1, 0))
    # display(ifelse(mask > 0, 1, 0) - ifelse(mask_m7 > 0, 1, 0) )
    # display(abind(
    #   DPO(border3, col405),
    #   DPO(border5, col405),
    #   DPO(border7, col405),
    #   along = 1
    # ))
    


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
    
    # display(nucDilate)
    # hist(nucDilate)
    # display(nucDilate)
    # NI = NImg405
    # dim(NI) = c(dim(NI), 1)
    # display(rgbImage(blue = NI,
    #                  green = nucDilate))
    # DPO(nucDilate == 2, col405)
    # nucDilate = dilate(mask, kern = makeBrush(15, shape = "disc"))
    # display(nucDilate)
    # hist(nucDilate)
    interface3 = ifelse(maskp3 == 2, 1, 0)
    interface5 = ifelse(maskp5 == 2, 1, 0)
    interface7 = ifelse(maskp7 == 2, 1, 0)
    interface9 = ifelse(maskp9 == 2, 1, 0)
    # interim = abind(interface3, interface5, interface7, interface9, along = 2)
    # display(interim)
    # writeImage(interim, files = paste0("/home/gendarme/Desktop/interim.tiff"), quality = 100, type = "tiff") 
    # nucBorder = nucDilate - mask
    # display(nucDilate9)
    # nucBorder3 = nucDilate3 - mask
    # nucBorder5 = nucDilate5 - mask
    # nucBorder7 = nucDilate7 - mask
    # nucBorder9 = nucDilate9 - mask

    # InterfaceBorderNuc = ifelse(mask > 0, 1, 0) - ifelse(nucBorder > 0, 2, 0)
    #  display(InterfaceBorderNuc)
    # InterfaceBorderNuc = ifelse(InterfaceBorderNuc == -2, 2, InterfaceBorderNuc)
    #  display(InterfaceBorderNuc == 2)
    # InterfaceBorderNuc = ifelse(InterfaceBorderNuc == -1, 3, InterfaceBorderNuc)
    #  display(InterfaceBorderNuc == 3)
    # hist(InterfaceBorderNuc)
    #  display(rgbImage(green = (InterfaceBorderNuc == 3),
    #                   red = border))
    # YBI = ifelse(mask > 0, 1, 0) + ifelse(border3 > 0, 2, 0) + ifelse(InterfaceBorderNuc > 0, 3, 0)
    # hist(YBI)    
    # InterfaceBorderNuc3 = ifelse(mask > 0, 1, 0) - ifelse(nucBorder3 > 0, 2, 0)
    # InterfaceBorderNuc3 = ifelse(InterfaceBorderNuc3 == -2, 2, InterfaceBorderNuc3)
    # InterfaceBorderNuc3 = ifelse(InterfaceBorderNuc3 == -1, 3, InterfaceBorderNuc3)

    # InterfaceBorderNuc5 = ifelse(mask > 0, 1, 0) - ifelse(nucBorder5 > 0, 2, 0)
    # InterfaceBorderNuc5 = ifelse(InterfaceBorderNuc5 == -2, 2, InterfaceBorderNuc5)
    # InterfaceBorderNuc5 = ifelse(InterfaceBorderNuc5 == -1, 3, InterfaceBorderNuc5)
    #  display(rgbImage(green = (InterfaceBorderNuc5 == 3),
    #                   red = border))
    # InterfaceBorderNuc7 = ifelse(mask > 0, 1, 0) - ifelse(nucBorder7 > 0, 2, 0)
    # InterfaceBorderNuc7 = ifelse(InterfaceBorderNuc7 == -2, 2, InterfaceBorderNuc7)
    # InterfaceBorderNuc7 = ifelse(InterfaceBorderNuc7 == -1, 3, InterfaceBorderNuc7)

    # InterfaceBorderNuc9 = ifelse(mask > 0, 1, 0) - ifelse(nucBorder9 > 0, 2, 0)
    # InterfaceBorderNuc9 = ifelse(InterfaceBorderNuc9 == -2, 2, InterfaceBorderNuc9)
    # InterfaceBorderNuc9 = ifelse(InterfaceBorderNuc9 == -1, 3, InterfaceBorderNuc9)
    #      display(rgbImage(green = (InterfaceBorderNuc7 == 3),
    #                   red = border))

    #  display(InterfaceBorderNuc == 3 )
    #  hist(as.array(InterfaceBorderNuc))

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
    
    dim(mask) <- c(dim(mask), 1)
    dim(mask3) <- c(dim(mask3), 1)
    dim(mask5) <- c(dim(mask5), 1)
    dim(mask7) <- c(dim(mask7), 1)
    dim(mask9) <- c(dim(mask9), 1)
    dim(border) <- c(dim(border), 1)
    # dim(nucBorder) <- c(dim(nucBorder), 1)
    dim(border3) <- c(dim(border3), 1)
    dim(border5) <- c(dim(border5), 1)
    dim(border7) <- c(dim(border7), 1)
    dim(border9) <- c(dim(border9), 1)
    dim(interface) <- c(dim(interface), 1)
    dim(interface3) <- c(dim(interface3), 1)
    dim(interface5) <- c(dim(interface5), 1)
    dim(interface7) <- c(dim(interface7), 1)
    dim(interface9) <- c(dim(interface9), 1)
    dim(center) <- c(dim(center), 1)
    dim(centerDil3) <- c(dim(centerDil3), 1)
    # dim(unit_DM) <- c(dim(unit_DM), 1)
    # dim(nucMask) <- c(dim(nucMask), 1)
  
    ## Encoding nucleus #####################################################################################
  n_sample_nuc <- as.tibble(train_data[Boolean & train_data$Channel == "DAPI",]) %>% 
    mutate(ActualShape = list(c(HEIGHT, WIDTH)),
           Mask = list(as.array(ifelse(mask > 0, 1, 0))),
           Mask3 = list(as.array(mask_m3)),
           Mask5 = list(as.array(mask_m5)),
           Mask7 = list(as.array(mask_m7)),
          #  Border = list(as.array(border)),
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
           Maskm3 = map2(Mask3, ActualShape, postprocess_image),
           Maskm5 = map2(Mask5, ActualShape, postprocess_image),
           Maskm7 = map2(Mask7, ActualShape, postprocess_image),
           Maskm9 = map2(Mask9, ActualShape, postprocess_image),
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
   
  for (Plate in unique(train_data$Plate)) {
    for(Well in unique(train_data$ID)) {
      for(Position in unique(train_data$Position)) {
      # foreach(Position = unique(train_data$Position)) %:%
        # foreach (Shape = unique(train_data$ShapeIndex)) %dopar% {
        for (Shape in unique(train_data$ShapeIndex)) {
          Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position & train_data$ShapeIndex == Shape
          
          # run image proc and encoding for 1 field of view
          ImageProcessingAndEncoding(Image = train_data, Plate = Plate, Well = Well, Position = Position, Shape = Shape)
          
          # check iteration
          print(paste0("Plate: ", Plate, "; Well: ", Well, "; Position:", Position,  "; Crop:", Shape,"; Time: ", Sys.time()))
        }
      }
    }
  }
