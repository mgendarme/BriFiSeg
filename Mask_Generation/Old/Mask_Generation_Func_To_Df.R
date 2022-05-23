WD <- as.character('/home/gendarme/Documents/U-Net/PMI_Data/Original_IF012')
#setwd(WD)
library(EBImage)
library(tidyverse)

# Settings#############################################################################################################################################
UnetTrain_dir <- "/home/gendarme/Documents/U-Net/PMI_Data/Original_IF012/Train/"
base_dir <- "/home/gendarme/Documents/U-Net/PMI_Data/Original_IF012"
TRAIN_PATH = paste0(base_dir,"/Train--Normalize-Range_1-2") 
HEIGHT = 1000
WIDTH = 1000
CHANNELS = 1          # only grayscale
SHAPE = c(WIDTH, HEIGHT, CHANNELS)

## Prepare training images and metadata
ImageMaskTrain <- list.files(base_dir, pattern = ".tif", recursive = T) %>% str_replace(".tif", "") %>%  as.data.frame()
colnames(ImageMaskTrain)[1] <- "ImageId"
ImageTrain <- filter(ImageMaskTrain, str_detect(ImageId, "mCherry|DAPI"))#|NImg405|Img568|Img405"))
ImageIdSimp <- str_split(ImageTrain$ImageId, "/", simplify = T)[,3]

#MaskTrain <- filter(ImageMaskTrain, str_detect(ImageId, "cytoMask"))
train_data <- data.frame(ImageId = ImageTrain$ImageId) %>%
  #mutate(MaskId = MaskTrain$ImageId) %>%
  mutate(ImageFile = file.path(base_dir, paste0(ImageId, ".tif")),
         ImageIdSimp = ImageIdSimp,
         #MaskFile = file.path(TRAIN_PATH, paste0(MaskId, ".tif")),
         NucEncodedPixels = NA,
         CytoEncodedPixels = NA,
         ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]),
         Plate = as.numeric(str_sub(ImageId, 16, 18)),
         ID = sub("--W.*", "", ImageIdSimp),
         Position = as.numeric(sapply(str_split(sapply(str_split(ImageIdSimp, "--P"), "[", 2), "--"), "[", 1)),
         Channel = sub(".*--", "", ImageId)
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
train_data <- filter(train_data, cN == 1 & ID == "C1")
enc_list <- list()

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

## preprocess_image performs operations on images e.g. clahe, normalize, resize ...
preprocess_image <- function(file, shape){
  image <- readImage(file, type = "tiff")
  #image <- to_gray_scale(image)                       ## convert to gray scale  
  image <- resize(image, w = shape[1], h = shape[2])  ## make all images of dimensions
  image <- clahe(image)                               ## local adaptive contrast enhancement
  image <- normalize(image)                           ## standardize between [0, 1]
  #image <- invert(image)                              ## invert brightfield
  dim(image) <- c(dim(image), 1)
  imageData(image)                                    ## return as array
} ### Not necessary here

# # ##For testing:
# # ##Image = train_data
Plate = 1
Well = "C1"
Position = 1
Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position

# Read images and extract features#############################################################################################
Rescale <- function(x){
  Bottom = 2^15/(2^16-1) 
  Top = (2^15+4095)/(2^16-1)
  (x - Bottom) / (Top - Bottom)   
}

NucSegmentation <- function(data, Boolean, Channel){
  Img405 = readImage(data[data$Channel == Channel & Boolean, "ImageFile"])
  Img405 = Rescale(Img405)
  NImg405 = normalize(Img405, inputRange = c(range(Img405)[1], range(Img405)[2]))
  
  FilterNuc = makeBrush(size = 51, shape = "gaussian", sigma = 2)
  Img405smooth = filter2(Img405, filter = FilterNuc)
  nucThrManual = thresh(Img405smooth, w = 100, h = 100, offset = 0.001)
  #display(nucThrManual)
  nucOpening = nucThrManual
  #display(nucOpening)
  nucSeed = bwlabel(nucOpening)
  #display(colorLabels(nucSeed))
  nucFill = fillHull(thresh(Img405smooth, w = 20, h = 20, offset = 0.005))
  #display(nucFill)
  nucRegions = propagate(Img405smooth, nucSeed, mask = nucFill)
  #display(nucRegions)
  Mask = watershed(distmap(nucRegions), tolerance = 1, ext = 1)
  #display(colorLabels(nucMask))
  #NImgCol405 = rgbImage(blue = NImg405*3)
  #display(NImgCol405)
  #nucSegm = paintObjects(nucMask, NImgCol405, col = 'red')
  #display(nucSegm, all = T)
}
nucMask = NucSegmentation(train_data, Boolean, Channel =  "DAPI")
str(nucMask)
#display(colorLabels(nucMask))

CytoSegmentation <- function(data, Boolean, Channel, nucleus){
  nucMask = nucleus
  FilterCyto = makeBrush(size = 201, shape = "gaussian", sigma = 5)
  Img568smooth = filter2(Img568, filter = FilterCyto)
  sharpen <- function(img){
    FilterHighPass = matrix(-1/40, nrow = 3, ncol = 3)
    FilterHighPass[2, 2] = 1
    img = filter2(img, FilterHighPass)
  }
  Img568sharpen = sharpen(Img568)
  #display(normalize(Img568sharpen))
  Img568sharpenSmooth = filter2(Img568sharpen, filter = makeBrush(size = 201, shape = "gaussian", sigma = 2))
  Img568sharpen = Img568sharpenSmooth
  # display(normalize(Img568smooth))
  cytoThr = thresh(Img568smooth, w = 400, h = 400, offset = 0.0000001)
  #display(cytoThr)
  #cytoOpening = opening(cytoThr, kern = makeBrush(1, shape = "disc"))
  #display(cytoOpening)
  ctmask = opening(Img568sharpen > 0.15, makeBrush(5, shape='disc'))
  #display(ctmask)
  cytoMask = propagate(Img568, seeds = nucMask, mask = ctmask)
}
cytoMask = NucSegmentation(train_data, Boolean, Channel =  "mCherry", nucleus = nucMask)
display(colorLabels(cytoMask))

Encoded2df <- function(data, nucleus, cytoplasm){

    #add encoding to df:
  dim(nucMask) <- c(dim(nucMask), 1)
  dim(cytoMask) <- c(dim(cytoMask), 1)
  #EncodedNuc <- postprocess_image(nucMask, shape = SHAPE)
  #EncodedCyto <- postprocess_image(cytoMask, shape = SHAPE)
  
  n_sample <- as.tibble(data[Boolean & train_data$Channel == "mCherry",]) %>%
    add_column(nucMasks = list(as.array(nucMask)), cytoMasks = list(as.array(cytoMask))) %>% # remove those 2 columns in
    mutate(NucEncodedPixels = map2(nucMasks, ImageShape, postprocess_image), CytoEncodedPixels = map2(cytoMasks, ImageShape, postprocess_image)) %>%
    select(-nucMasks, -cytoMasks, -ImageShape)
  #str(n_sample)
  
  n_sample_out <- n_sample %>%
    unnest(NucEncodedPixels, CytoEncodedPixels) %>%
    mutate(NucEncodedPixels = as.character(NucEncodedPixels), CytoEncodedPixels = as.character(CytoEncodedPixels))
  str(n_sample_out)

}

n_sample_out <- Encoded2df(train_data, nucMask, cytoMask)

ImageProcessingAndEncoding <- function(Image, Plate, Well, Position){
  
  
  # #--------------- Save images and masks ---------------------------------------------------------------------------------
  # writeImage(nucMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--nucMask.tif"))
  # writeImage(cytoMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMask.tif"))
  # #writeImage(cytoMaskNorm, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMaskNorm.tif"))
  # #writeImage(NImg405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg405.tif"))
  # writeImage(Img405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Img405.tif"))
  # #writeImage(NImg568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg568.tif"))
  # writeImage(Img568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Img568.tif"))
  # writeImage(Segm, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Segm.tif"))
  
}

#----------------Run ImageProcessing() to everyfield---------------------------------------------------------------------------------------------------
# run full script
for (Plate in unique(train_data$Plate)) {
  for(Well in unique(train_data$ID)) {
    for(Position in unique(train_data$Position)) {
      Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position
      n_sample <- as.tibble(train_data[Boolean & train_data$Channel == "mCherry",]) 
      
      # run image proc and encoding for 1 field of view
      ImageProcessingAndEncoding(Image = train_data, Plate = Plate, Well = Well, Position = Position) 
      
      # add encoded information to list
      n_sample %>%
        add_column(nucMasks = list(as.array(nucMask)), cytoMasks = list(as.array(cytoMask))) %>% # remove those 2 columns in
        mutate(NucEncodedPixels = map2(nucMasks, ImageShape, postprocess_image), CytoEncodedPixels = map2(cytoMasks, ImageShape, postprocess_image)) %>%
        select(-nucMasks, -cytoMasks, -ImageShape)
      #str(n_sample)
      
      n_sample_out <- n_sample %>%
        unnest(NucEncodedPixels, CytoEncodedPixels) %>%
        mutate(NucEncodedPixels = as.character(NucEncodedPixels), CytoEncodedPixels = as.character(CytoEncodedPixels))
      enc_list[[paste0("p--", Plate, "--", Well, "--P", Position)]] <- n_sample_out
      
      # check iteration
      print(paste0("Plate: ", Plate, "; Well: ", Well, "; Position:", Position, "; Time: ", Sys.time()))
    }
  }
}

# transform then write output
enc_df = do.call(rbind, enc_list)
str(enc_df)
#write_csv(train_data_enc, paste0(UnetTrain_dir,"train_data_enc.csv"))
#read_test <- read_csv(paste0(UnetTrain_dir,"train_data_enc.csv"))
#str(read_test)
