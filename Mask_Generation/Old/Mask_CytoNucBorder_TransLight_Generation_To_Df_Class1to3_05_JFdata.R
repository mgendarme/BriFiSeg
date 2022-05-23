library(EBImage)
library(tidyverse)
library(tiff)

# Settings#############################################################################################################################################
UnetTrain_dir <- "~/Documents/UNet/BF_Data/JF_Data/Train"
base_dir <- "~/Documents/UNet/BF_Data/JF_Data/JF_2019_08"
TRAIN_PATH = UnetTrain_dir
HEIGHT = 1000 # 512
WIDTH = 1000 # 512
CHANNELS = 1          # only grayscale
SHAPE = c(WIDTH, HEIGHT, CHANNELS)

# Param for eroding individual masks
THICKNESS = 5
SHAPE = "disc"

currenti = paste0("noBorder_shape", SHAPE, "_thickness",THICKNESS)
segm_dir = paste0(UnetTrain_dir, "/", currenti)
dir.create(segm_dir)
# crop_range <- 501:(501+512-1)
crop_range <- 1:1000

## Prepare training images and metadata
ImageMaskTrain <- list.files(base_dir, pattern = ".tif", recursive = T) %>% str_replace(".tif", "") %>%  as.data.frame() 
colnames(ImageMaskTrain)[1] <- "ImageId"
ImageTrain <- filter(ImageMaskTrain, str_detect(ImageId, ""))#|NImg405|Img568|Img405"))
# ImageIdSimp <- str_split(ImageTrain$ImageId, "/", simplify = T)[,2]

#MaskTrain <- filter(ImageMaskTrain, str_detect(ImageId, "cytoMask"))
train_data <- data.frame(ImageId = ImageTrain$ImageId) %>%
  #mutate(MaskId = MaskTrain$ImageId) %>%
  mutate(ImageFile = file.path(base_dir, paste0(ImageId, ".tif")),
         # ImageIdSimp = ImageIdSimp,
         NucEncodedPixels = NA,
         CytoEncodedPixels = NA,
         ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file)[crop_range, crop_range])[1:2]),
         Plate = 1,
         # Plate = sub(".*data/", "", Plate),
         # Plate = as.numeric(sub("--W.*", "", Plate)),
         ID = sub(".*data/", "", ImageId),
         ID = sub("--W.*", "", ID),
         Position = sub(".*--P", "", ImageId),
         Position = sub("--.*", "", Position),
         Channel = sub(".*--", "", ImageId),
         ImageId = paste0("JF_2019_08/",ImageId)
  )

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
  FileList <- merge(x = FileList, y = ID, by = c("ID"), all.x = T)
}
train_data <- IdToMap(train_data)
# train_data <- filter(train_data, rN == 1)
str(train_data)

#################################### Encoding functions ######################################################################
# Encoding of mask in data frame for CNN: ####
## image2rle encodes the image pixels values
image2rle <- function(image){
  
  labels <- 1:max(image) ## assuming background  == 0
  
  x <- as.vector(t(image))
  
  encoding <- rle(x)
  
  ## Adding start positions
  encoding$positions <- 1 + c(0, cumsum(encoding$lengths[-length(encoding$lengths)]))
  
  ## encodes every individual mask bsaed on label
  mask2rle <- function(label, enc) {
    indices <- enc$values == label
    list(position = enc$positions[indices][1],
         encoding = paste(enc$positions[indices], enc$lengths[indices], collapse=" "))
  }
  
  ##return encodings with increasing positions
  map_df(labels, mask2rle, encoding) %>%
    arrange(position) %>%
    pull(encoding)
}

## postprocess_image resize the masks images and runs image2rle, eventually adds labels if not existing
postprocess_image <- function(image, shape){
  image <- resize(image[,,1], w = shape[1], h = shape[2])
  #image <- bwlabel(image > .5) ##binarize and label if no labels to start with
  image2rle(image)
}

## preprocess_image performs operations on images e.g. clahe, EBImage::normalize, resize ...
preprocess_image <- function(file, shape){
  image <- readImage(file, type = "tiff")
  #image <- to_gray_scale(image)                       ## convert to gray scale  
  image <- resize(image, w = shape[1], h = shape[2])  ## make all images of dimensions
  image <- clahe(image)                               ## local adaptive contrast enhancement
  image <- EBImage::normalize(image)                           ## standardize between [0, 1]
  #image <- invert(image)                              ## invert brightfield
  dim(image) <- c(dim(image), 1)
  imageData(image)                                    ## return as array
} ### Not necessary here

# Read images and extract features#############################################################################################
ImageProcessingAndEncoding <- function(Image, Plate, Well, Position){
  
  ##For testing only:
  # # ##Image = train_data
  # Plate = 1
  # Well = "A10"
  # Position = 1
  # Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position
  #---------------Rescale pixel intensity-----------
  Rescale <- function(x){
    Bottom = 2^15/(2^16-1) 
    Top = (2^15+4095)/(2^16-1)
    (x - Bottom) / (Top - Bottom)   
  }
  
  #---------------Load Images------------------------
  Img405 = readImage(train_data[train_data$Channel == "dapi" & Boolean, "ImageFile"])[crop_range, crop_range]
  # str(Img405)
  Img405 = Rescale(Img405)
  NImg405 = EBImage::normalize(Img405, inputRange = c(range(Img405)[1], range(Img405)[2]))
  # display(NImg405)
  # Img488 = readImage(train_data[train_data$Channel == "GFP" & Boolean, "ImageFile"])
  # Img488 = Rescale(Img488)
  # NImg488 = EBImage::normalize(Img488)
  #display(NImg488)
  ImgTL = readImage(train_data[train_data$Channel == "Transmission" & Boolean, "ImageFile"])[crop_range, crop_range]
  ImgTL = Rescale(ImgTL)
  # filter_high <- matrix(1, nc = 3, nr = 3)
  # filter_high[2, 2] <- -10
  # Img568_high <- filter2(Img568, filter_high)
  # display(normalize(Img568_high * 1))
  
  # Img568 = Rescale(Img568)
  # NImg568 = EBImage::normalize(Img568, inputRange = c(range(Img568)[1], range(Img568)[2]))
  NImgTL = EBImage::normalize(ImgTL, inputRange = c(range(ImgTL)[1], range(ImgTL)[2]))
  # display(NImgTL)
  # Img647 = readImage(train_data[train_data$Channel == "647" & Boolean, "FileNames"])
  # Img647 = Rescale(Img647)
  # NImg647 = EBImage::normalize(Img647, inputRange = c(range(Img647)[1], range(Img647)[2]))
  #display(NImg647)
  
  #---------------smooth and threshold nuleus--------------------------------
  FilterNuc = makeBrush(size = 51, shape = "gaussian", sigma = 2)
  Img405smooth = filter2(Img405, filter = FilterNuc)
  nucThrManual = thresh(Img405smooth, w = 100, h = 100, offset = 0.001)
  #display(nucThrManual)
  nucOpening = nucThrManual
  nucSeed = bwlabel(nucOpening)
  nucFill = fillHull(thresh(Img405smooth, w = 20, h = 20, offset = 0.005))
  nucRegions = propagate(Img405smooth, nucSeed, mask = nucFill)
  nucMask = watershed(distmap(nucRegions), tolerance = 1, ext = 1)
  #display(colorLabels(nucMask))
  NImgCol405 = rgbImage(blue = NImg405*3)
  #display(NImgCol405)
  nucSegm = paintObjects(nucMask, NImgCol405, col = 'red')
  # display(nucSegm, all = T)
  
  #------------------Generate voronoi-----------------------------------------
  VoR = propagate(nucMask, nucMask, lambda = 100000)
  # display(colorLabels(VoR))

  #---------------smooth and threshold Cytoplasm------------------------------
  # FilterCyto = makeBrush(size = 201, shape = "gaussian", sigma = 5)
  # Img568smooth = filter2(Img568, filter = FilterCyto)
  # sharpen <- function(img){
  #   FilterHighPass = matrix(-1/40, nrow = 3, ncol = 3)
  #   FilterHighPass[2, 2] = 1
  #   img = filter2(img, FilterHighPass)
  # }
  # Img568sharpen = sharpen(Img568)
  # #display(EBImage::normalize(Img568sharpen))
  # Img568sharpenSmooth = filter2(Img568sharpen, filter = makeBrush(size = 201, shape = "gaussian", sigma = 2))
  # Img568sharpen = Img568sharpenSmooth
  # # display(EBImage::normalize(Img568smooth))
  # cytoThr = thresh(Img568smooth, w = 400, h = 400, offset = 1e-6)
  # #display(cytoThr)
  # ctmask = opening(Img568sharpen > 0.115, makeBrush(11, shape='disc'))
  # cytoMask = propagate(Img568, seeds = nucMask, mask = ctmask)
  # NImgCol568 = rgbImage(red = NImg568 * 2)
  # #NImgCol405_568 = rgbImage(blue = NImg405 * 3, red = NImg568 * 1.5)
  # display(NImgCol568)
  #display(Segm, all = T)
  
  #--------------Perform erode on individual masks --------------------------
  # Empty = Image(data = array(0, dim = c(1000 ,1000)))
  # for (i in 1:(max(cytoMask) + 1)) {
  #   cytoMaskTemp <- erode(cytoMask == i, kern = makeBrush(11, shape = "disc"))
  #   Empty = Empty + cytoMaskTemp
  # }
  # EmptyInv = 1 - Empty
  # cytoMaskBw = ifelse(cytoMask == 0 , 1, 0)
  # #display(cytoMaskBw)
  # # Generate edges of cytoplasms
  # Edges = EmptyInv - cytoMaskBw 
  # #display(Edges)
  # ctmaskNoEdges = ctmask - Edges
  # #display(ctmaskNoEdges)
  # cytoMaskNoEdges <- bwlabel(ctmaskNoEdges)
  # #display(colorLabels(cytoMaskNoEdges))
  # 
  # # filter out objects with s.area < 20
  # F_cytoMaskNoEdges <- as.data.frame(computeFeatures.shape(cytoMaskNoEdges)) %>% rownames_to_column
  # ids <- filter(as.data.frame(F_cytoMaskNoEdges), s.area < 50)
  # cytoMaskNoEdgesFilter <- rmObjects(cytoMaskNoEdges, index = ids)
  # #display(cytoMaskNoEdgesFilter)
  # 
  # #cytoMask <- cytoMaskNoEdgesFilter
  # #display(Edges)
  # 
  # ## Generate masks for border and boudaries between cells/nuclei: 
  # cytoDilate = dilate(cytoMask, kern = makeBrush(THICKNESS, shape = SHAPE))
  # #display(combine(colorLabels(cytoMask), colorLabels(cytoDilate)), all = T)
  # cytoBorder = cytoDilate - cytoMask
  # display(colorLabels(cytoBorder))
  # InterfaceBorderCyto = ifelse(cytoMask > 0, 1, 0) - ifelse(cytoBorder > 0, 2, 0)
  # InterfaceBorderCyto = ifelse(InterfaceBorderCyto == -2, 2, InterfaceBorderCyto)
  # InterfaceBorderCyto = ifelse(InterfaceBorderCyto == -1, 3, InterfaceBorderCyto)
  # ## Border = -2; Interface =-1; Background = 0; Cyto = 1
  # display(EBImage::normalize(InterfaceBorderCyto))
  
  # display(EBImage::normalize(ifelse(InterfaceBorderCyto < 0, 1, 0)))
  # display(colorLabels(InterfaceBorderCyto))
  # #Mask to compute IoU on:
  # display(colorLabels(bwlabel(ifelse(InterfaceBorderCyto == 1, 1, 0))))
  
  nucDilate = dilate(nucMask, kern = makeBrush(THICKNESS, shape = SHAPE))
  #display(combine(colorLabels(nucMask), colorLabels(nucDilate)), all = T)
  nucBorder = nucDilate - nucMask
  #display(colorLabels(nucBorder))
  InterfaceBorderNuc = ifelse(nucMask > 0, 1, 0) - ifelse(nucBorder > 0, 2, 0)
  InterfaceBorderNuc = ifelse(InterfaceBorderNuc == -2, 2, InterfaceBorderNuc)
  InterfaceBorderNuc = ifelse(InterfaceBorderNuc == -1, 3, InterfaceBorderNuc)
  ## Border = -2; Interface =-1; Background = 0; Cyto = 1
  # display(EBImage::normalize(InterfaceBorderNuc))
  # str(InterfaceBorderNuc)
  
  #display(EBImage::normalize(ifelse(InterfaceBorderNuc == -1, 1, 0)))
  # hist(InterfaceBorderNuc)

  Segm = paintObjects(nucMask, NImgTL, col = 'red', thick = T)
  # Segm = paintObjects(nucMask, Segm, col = 'blue')
  # display(Segm)
  # #---------------Remove cells at the edge-------------------------------------------------------------------------------
  # #subset for boundary pixels
  # dims = dim(nucMask) 
  # #display(colorLabels(nucMask))
  # border = c(nucMask[1:dims[1],1],
  #            nucMask[1:dims[1],dims[2]],
  #            nucMask[1,1:dims[2]],
  #            nucMask[dims[1],1:dims[2]]
  #            )
  # #extract object identifiers at the boundary
  # ids = unique(border[which(border != 0)])
  # #Filter our ids from Fc or add info to new col.
  # # Fc$Border <- ifelse(Fc[,"rowname"] %in% ids, "Border", "Inner") #if the border objects are of interest and need to be kept
  # Fc = filter(Fc, !rowname %in% ids)
  # 
  # if (nrow(Fc) == 0) {
  #   Fc = data.frame(0, nrow = 0, ncol = 141*12+1)
  # }
  # 
  # write.csv(Fc, paste0("IF005","_p--",Plate,"_w--",Well,"_pos--",Position,".csv"))
  
  # #--------------- Save images and masks ---------------------------------------------------------------------------------
  # writeImage(nucMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--nucMask.tif"))
  # writeImage(cytoMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMask.tif"))
  # #writeImage(cytoMaskNorm, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMaskNorm.tif"))
  # #writeImage(NImg405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg405.tif"))
  # writeImage(Img405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Img405.tif"))
  # #writeImage(NImg568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg568.tif"))
  # writeImage(Img568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Img568.tif"))
  
  # writeTIFF(ImgTL, paste0(UnetTrain_dir, "/p--",Plate,"--",Well,"--P",Position,"--ImgTL.tiff"))
  # writeTIFF(Segm, paste0(segm_dir, "/test.tiff"))
                          # "p_",Plate,"--W_",Well,"--P_",Position,"-Segm.tif"), type = "tiff")
  
  # hist(InterfaceBorderCyto)
  # hist(InterfaceBorderNuc)
  # display(InterfaceBorderCyto == 1)
  # display(InterfaceBorderNuc == 1)
  
  ##Change dimensions:
  dim(nucMask) <- c(dim(nucMask), 1)
  # dim(cytoMask) <- c(dim(cytoMask), 1)
  dim(InterfaceBorderNuc) <- c(dim(InterfaceBorderNuc), 1)
  # dim(InterfaceBorderCyto) <- c(dim(InterfaceBorderCyto), 1)
  # str(InterfaceBorderCyto)
  # str(InterfaceBorderNuc)
  
  ## Encoding nucleus #####################################################################################
  n_sample_nuc <- as.tibble(train_data[Boolean & train_data$Channel == "dapi",]) %>% 
    add_column(BorderMasks = list(as.array(ifelse(InterfaceBorderNuc == 2, 1, 0))),
               InterfaceMasks = list(as.array(ifelse(InterfaceBorderNuc == 3, 1, 0))),
               #BackgroundMasks = list(as.array(ifelse(InterfaceBorderNuc == 0, 1, 0))),
               #InsideMasks = list(as.array(ifelse(InterfaceBorderNuc == 1, 1, 0))),
               Masks = list(as.array(ifelse(InterfaceBorderNuc == 1, 1, 0)))) %>%
    mutate(BorderEncodedPixels = map2(BorderMasks, ImageShape, postprocess_image),
           InterfaceEncodedPixels = map2(InterfaceMasks, ImageShape, postprocess_image),
           #InsideEncodedPixels = map2(InsideMasks, ImageShape, postprocess_image),
           MaskEncodedPixels = map2(Masks, ImageShape, postprocess_image)) %>% 
    select(-BorderMasks, -InterfaceMasks, -Masks, -ImageShape) #InsideMasks,
  
  n_sample_out_nuc <- n_sample_nuc %>% 
    unnest(BorderEncodedPixels,
           InterfaceEncodedPixels,
           #InsideEncodedPixels,
           MaskEncodedPixels) %>% 
    mutate(BorderEncodedPixels = as.character(BorderEncodedPixels),
           InterfaceEncodedPixels= as.character(InterfaceEncodedPixels),
           #InsideEncodedPixels= as.character(InsideEncodedPixels),
           MaskEncodedPixels= as.character(MaskEncodedPixels)) %>% 
    select(ImageId, BorderEncodedPixels, InterfaceEncodedPixels, MaskEncodedPixels) #InsideEncodedPixels,
  #str(n_sample_out_nuc)
  
  ## Encoding cytoplasm ##################################################################################
  n_sample_cyto <- as.tibble(train_data[Boolean & train_data$Channel == "Transmission",]) %>% 
    add_column(BorderMasks = list(as.array(ifelse(InterfaceBorderNuc == 2, 1, 0))),
               InterfaceMasks = list(as.array(ifelse(InterfaceBorderNuc == 3, 1, 0))),
               #BackgroundMasks = list(as.array(ifelse(InterfaceBorderCyto == 0, 1, 0))),
               #InsideMasks = list(as.array(ifelse(InterfaceBorderCyto == 1, 1, 0))),
               Masks = list(as.array(ifelse(InterfaceBorderNuc == 1, 1, 0)))) %>%
    mutate(BorderEncodedPixels = map2(BorderMasks, ImageShape, postprocess_image),
           InterfaceEncodedPixels = map2(InterfaceMasks, ImageShape, postprocess_image),
           #InsideEncodedPixels = map2(InsideMasks, ImageShape, postprocess_image),
           MaskEncodedPixels = map2(Masks, ImageShape, postprocess_image)) %>% 
    select(-BorderMasks, -InterfaceMasks, -Masks, -ImageShape) #InsideMasks,
  
  n_sample_out_cyto <- n_sample_cyto %>% 
    unnest(BorderEncodedPixels,
           InterfaceEncodedPixels,
           #InsideEncodedPixels,
           MaskEncodedPixels) %>% 
    mutate(BorderEncodedPixels = as.character(BorderEncodedPixels),
           InterfaceEncodedPixels= as.character(InterfaceEncodedPixels),
           #InsideEncodedPixels= as.character(InsideEncodedPixels),
           MaskEncodedPixels= as.character(MaskEncodedPixels)) %>% 
    select(ImageId, BorderEncodedPixels, InterfaceEncodedPixels, MaskEncodedPixels) #InsideEncodedPixels,
  # str(n_sample_out_cyto)
  # display(rle2masks(n_sample_out_cyto$BorderEncodedPixels, shape = c(2424, 2424, 1)))
  
  ## Export encoding
  write_csv(n_sample_out_nuc, paste0(segm_dir,"/Plate_", Plate, "--Well_", Well, "--Position_", Position,"--n_sample_out_nuc_",currenti,".csv"))
  write_csv(n_sample_out_cyto, paste0(segm_dir,"/Plate_", Plate, "--Well_", Well, "--Position_", Position,"--n_sample_out_cyto_",currenti,".csv"))
  #write_csv(n_sample_out_InterfaceBorderCyto, paste0(UnetTrain_dir,"Plate_", Plate, "--Well_", Well, "--Position_", Position,"--n_sample_out_InterfaceBorderCyto.csv"))
  #write_csv(n_sample_out_InterfaceBorderNuc, paste0(UnetTrain_dir,"Plate_", Plate, "--Well_", Well, "--Position_", Position,"--n_sample_out_InterfaceBorderNuc.csv"))
  
  ##Check if cytomask encodes for labels
  # cytoMaskNorm <- EBImage::normalize(cytoMask)
  # ftsCyto = computeFeatures.moment(cytoDilate)
  # display(colorLabels(cytoDilate), method="raster")
  # text((ftsCyto[,"m.cx"]), (ftsCyto[,"m.cy"]), labels = rownames(ftsCyto), col="Green", cex=.8)
  # # range(cytoMaskNorm)
  # # hist(cytoMaskNorm)
  # ftsBorder = computeFeatures.moment(cytoBorder)
  # display(colorLabels(cytoBorder), method="raster")
  # text((ftsBorder[,"m.cx"]), (ftsBorder[,"m.cy"]), labels = rownames(ftsBorder), col="Green", cex=.8)
}

#----------------Run ImageProcessing() to everyfield---------------------------------------------------------------------------------------------------
for (Plate in unique(train_data$Plate)) {
  for(Well in unique(train_data$ID)) {
    for(Position in unique(train_data$Position)) {
      Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position
      # run image proc and encoding for 1 field of view
      ImageProcessingAndEncoding(Image = train_data, Plate = Plate, Well = Well, Position = Position)
      
      # check iteration
      print(paste0("Plate: ", Plate, "; Well: ", Well, "; Position:", Position, "; Time: ", Sys.time()))
    }
  }
}
