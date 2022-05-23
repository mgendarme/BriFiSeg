library(keras)
#install_keras()
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
ImageTrain <- filter(ImageMaskTrain, str_detect(ImageId, "NImg"))#|NImg405|Img568|Img405"))
MaskTrain <- filter(ImageMaskTrain, str_detect(ImageId, "Mask"))
ImageMaskMetaTrain <- data.frame(ImageId = ImageTrain$ImageId) %>%
  mutate(MaskId = MaskTrain$ImageId) %>%
  mutate(ImageFile = file.path(TRAIN_PATH, paste0(ImageId, ".tif")),
         MaskFile = file.path(TRAIN_PATH, paste0(MaskId, ".tif")),
         EncodedPixels = NA,
         ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]))
head(ImageMaskMetaTrain)


## Load Images for testing
CytoMask <- readImage(paste0(ImageMaskMetaTrain$MaskFile)[1])
NucMask <- readImage(paste0(ImageMaskMetaTrain$MaskFile)[2])
Img568 <- readImage(paste0(ImageMaskMetaTrain$ImageFile)[2])
Img405 <- readImage(paste0(ImageMaskMetaTrain$ImageFile)[1])
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
range(cytoMask)
hist(cytoMask)
normalize(cytoMask, inputRange = c(0, range(cytoMask)[2]))
display(normalize(cytoMask))
#NImgCol405_568 = rgbImage(blue = Img405 * 3, red = Img568 * 1.5)
#Segm = paintObjects(cytoMask, NImgCol405_568, col = 'orange')
#Segm = paintObjects(nucMask, Segm, col = 'red')
#display(Segm)
#writeImage(cytoMask, "/home/gendarme/Documents/mask.tiff", bits.per.sample = 16, type = "tif", quality = 100L)
#mask <- readImage("/home/gendarme/Documents/mask.tiff")
#display(normalize(mask))


## Transform mask for U-Net ####
str(cytoMask)
# split.along.dim <- function(a, n){
#   setNames(lapply(split(a, arrayInd(seq_along(a), dim(a))[, n]),
#                   array, dim = dim(a)[-n], dimnames(a)[-n]),
#            dimnames(a)[[n]])
# }

Img568List <- list(as.array(abind(Img568, rev.along = 0)))
cytoMaskList <- list(as.array(abind(cytoMask, rev.along = 0)))
ImgMaskList <- list(X = Img568List, Y = cytoMaskList)
#ImgMaskVec <- unlist(ImgMaskList, , use.names=F)
ImgMaskTbl <- as.tibble(ImgMaskList)
str(ImgMaskList)

a <- normalize(ImgMaskTbl$Y[[1]])
b <- ImgMaskTbl$X[[1]]
display(combine(a, b), all = T)

image = abind(Img568, rev.along = 0) # for testing
str(image)

shape = c(1000, 1000, 1)

image2rle <- function(image){
  
  labels <- 1:max(image) ## assuming background  == 0
  
  x <- as.vector(t(image))
  
  encoding <- rle(x)
  
  ## Adding start positions
  encoding$positions <- 1 + c(0, cumsum(encoding$lengths[-length(encoding$lengths)]))
  
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

postprocess_image <- function(image, shape){
  
  image <- resize(image[,,1], w = shape[1], h = shape[2])
  image <- bwlabel(image > 0) ##binarize and label
  image2rle(image)
}

df <- data.frame(ID = 1)
str(df)
a1 <- df %>% add_column(Masks = cytoMaskList) %>% 
  mutate(Y = map2(Masks, shape, postprocess_image))
str(a1)
?mutate
?map2

#is.array(cytoMask) # T
#s.matrix(cytoMask) # T
#is.list(cytoMask) # F
#tibble(as.data.frame(cytoMask)) # DNW
#tibble(cytoMask) # DNW

image = cytoMask
dim(image) <- c(dim(image), 1)
shape = c(1000, 1000, 1)
image2rle <- function(image){
  
  labels <- 1:max(image) ## assuming background  == 0
  
  x <- as.vector(t(image))
  
  encoding <- rle(x)
  
  ## Adding start positions
  encoding$positions <- 1 + c(0, cumsum(encoding$lengths[-length(encoding$lengths)]))
  
  ################################################################################################# label? enc?
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

postprocess_image <- function(image, shape){
  image <- resize(image[,,1], w = shape[1], h = shape[2])
  image <- bwlabel(image > .5) ##binarize and label
  image2rle(image)
}

a <- postprocess_image(image, shape)
