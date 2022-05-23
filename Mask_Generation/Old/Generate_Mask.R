library(keras)
library(dplyr)
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster") 

# Defining some parameters:
## General parameters
base_dir <- "/home/gendarme/Documents/U-Net/PMI_Data/UnetTrain"
TRAIN_PATH = paste0(base_dir,"/Train--Normalize-Range_1-2") 
TEST_PATH = paste0(base_dir,"/Test--Normalize-Range_1-2")
HEIGHT = 256
WIDTH = 256
CHANNELS = 1          # only grayscale
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
## U-Net specific parameters
BATCH_SIZE = 16       # BATCH_SIZE = Number of training samples in (1 Forward / 1 Backward) pass
EPOCHS = 25           # 1 EPOCH = 1 Forward pass + 1 Backward pass for ALL training samples

## Prepare training images and metadata
ImageMaskTrain <- list.files(TRAIN_PATH) %>% str_replace(".tif", "") %>% as.data.frame()
colnames(ImageMaskTrain)[1] <- "ImageId"
ImageTrain <- filter(ImageMaskTrain, str_detect(ImageId, "-Img568"))#|NImg405|Img568|Img405"))
MaskTrain <- filter(ImageMaskTrain, str_detect(ImageId, "cytoMask"))
train_data <- data.frame(ImageId = ImageTrain$ImageId) %>%
  mutate(MaskId = MaskTrain$ImageId) %>%
  mutate(ImageFile = file.path(TRAIN_PATH, paste0(ImageId, ".tif")),
         MaskFile = file.path(TRAIN_PATH, paste0(MaskId, ".tif")),
         EncodedPixels = NA,
         ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]))
head(train_data)

# Load images for testing
CytoMask <- readImage(paste0(TRAIN_PATH,"/",filter(TabImageTrain, str_detect(ListImageTrain, "cytoMask"))[1,]))
Img405 <- readImage(paste0(TRAIN_PATH,"/",filter(TabImageTrain, str_detect(ListImageTrain, "Img405"))[1,]))
Img568 <- readImage(paste0(TRAIN_PATH,"/",filter(TabImageTrain, str_detect(ListImageTrain, "Img568"))[1,]))
Img568 <- readImage(paste0(TRAIN_PATH,"/",filter(TabImageTrain, str_detect(ListImageTrain, "Img568"))[1,]))
display(Img405)
display(Img568)
display(CytoMask)

#--------------segment nucleus---------------------------------------------
FilterNuc = makeBrush(size = 51, shape = "gaussian", sigma = 2)
Img405smooth = filter2(Img405, filter = FilterNuc)
nucThrManual = thresh(Img405smooth, w = 100, h = 100, offset = 0.001)
nucOpening = nucThrManual
nucSeed = bwlabel(nucOpening)
nucFill = fillHull(thresh(Img405smooth, w = 20, h = 20, offset = 0.005))
nucRegions = propagate(Img405smooth, nucSeed, mask = nucFill)
nucMask = watershed(distmap(nucRegions), tolerance = 1, ext = 1)
str(nucMask)
nucMask
NImgCol405 = rgbImage(blue = Img405*3)
nucSegm = paintObjects(nucMask, NImgCol405, col = 'red')
display(nucSegm, all = T)

#---------------smooth and threshold Cytoplasm------------------------------
FilterCyto = makeBrush(size = 201, shape = "gaussian", sigma = 5)
Img568smooth = filter2(Img568, filter = FilterCyto)
sharpen <- function(img){
  FilterHighPass = matrix(-1/40, nrow = 3, ncol = 3)
  FilterHighPass[2, 2] = 1
  img = filter2(img, FilterHighPass)
}
Img568sharpen = sharpen(Img568)
Img568sharpenSmooth = filter2(Img568sharpen, filter = makeBrush(size = 201, shape = "gaussian", sigma = 2))
Img568sharpen = Img568sharpenSmooth
cytoThr = thresh(Img568smooth, w = 400, h = 400, offset = 0.0000001)
ctmask = opening(Img568sharpen > 0.15, makeBrush(5, shape='disc'))
cytoMask = propagate(Img568, seeds = nucMask, mask = ctmask)
display(colorLabels(cytoMask))
# NImgCol405_568 = rgbImage(blue = Img405 * 3, red = Img568 * 1.5)
# Segm = paintObjects(cytoMask, NImgCol405_568, col = 'orange')
# Segm = paintObjects(nucMask, Segm, col = 'red')
# display(Segm)

## Transform mask for U-Net ####
# for testing:
image <- cytoMask
display(colorLabels(image))
dim(cytoMask) <- c(dim(cytoMask), 1)
SHAPE <- c(1000, 1000, 1)

##########################################################################################################################################
################################################### Steps for ENCODING masks #############################################################
##########################################################################################################################################

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

# cytoMaskPProcess <- postprocess_image(cytoMask, SHAPE)
# str(cytoMaskPProcess)
# DF_cytoMaskPProcess <- as.data.frame(cytoMaskPProcess)

# #Here Masks=cytoMask, ImageShape = c(1000, 1000, 1), postprocess_image same as org:
# train_data_complete <- train_data %>%
#   #add_column(Masks = array_branch(cytoMask, 1)) #%>%
#   mutate(EncodedPixels = map2(ImageShape, postprocess_image))
  
##########################################################################################################################################
################################################### Steps for DECODING masks #############################################################
##########################################################################################################################################

#rle2masks generates single mask from encoded pixels:
rle2masks <- function(encodings, shape, labels = TRUE) {
  
  ## Convert rle encoded mask to image
  rle2mask <- function(encoding, shape){
    
    #encoding = cytoMaskPProcess
    #shape = SHAPE
    splitted <- as.integer(str_split(encoding, pattern = "\\s+", simplify = TRUE))
    dfsplit <- as.data.frame(splitted)
    positions <- splitted[seq(1, length(splitted), 2)]
    lengths <- splitted[seq(2, length(splitted), 2)] - 1
    
    ## decode
    mask_indices <- unlist(map2(positions, lengths, function(pos, len) seq.int(pos, pos+len)))
    
    ## shape as 2D image
    mask <- numeric(prod(shape))
    mask[mask_indices] <- 1
    mask <- matrix(mask, nrow=shape[1], ncol=shape[2], byrow=TRUE)
    mask
  }
  
  if(!labels) {     ##reduce to one image
    masks <- matrix(0, nrow=shape[1], ncol=shape[2])
    for(i in 1:length(encodings))
      masks <- masks + rle2mask(encodings[i], shape)
  }
  else {           ##each mask in channel
    masks <- array(0, dim = c(shape[1], shape[2], length(encodings)))
    for(i in 1:length(encodings))
      masks[,,i] <- i*rle2mask(encodings[i], shape)
  }

  masks
}

# #For testing:
# cytoMaskPProcess_rle2masks <- rle2masks(cytoMaskPProcess, SHAPE, labels = T)
#  display(normalize(cytoMaskPProcess_rle2masks), all=T) 

#prepocess_masks puts all mask from one image together:
preprocess_masks <- function(encoding, old_shape, new_shape, labels = TRUE){
  require(EBImage)
  
  masks <- rle2masks(encoding, old_shape, labels = labels)
  
  if(any(old_shape[1:2] != new_shape[1:2])) {
    masks <- resize(masks, w = new_shape[1], h = new_shape[2])
    if(labels) {
      for(i in 1:dim(masks)[3]) ##recover labeling after reshaping
        masks[,,i] <- i*(masks[,,i] > 0)
    }
  }
  
  masks <- Reduce("+", getFrames(masks))
  
  dim(masks) <- c(dim(masks), 1) ##masks have no color channels
  masks
}

# # For testing:
 # cytoMaskPProcess_rle2masks_preprocess_masks <- preprocess_masks(cytoMaskPProcess, shape, shape, labels = T)
 # display(colorLabels(cytoMaskPProcess_rle2masks_preprocess_masks))
