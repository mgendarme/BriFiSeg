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


## kernels for processing
#####
LoadKernels <- function(){
  sharp <<- matrix(c(0, -1, 0,
                     -1, 8, -1,
                     0, -1, 0), nrow = 3, ncol = 3)
  
  outline <<- matrix(c(-1, -1, -1,
                       -1, 8, -1,
                       -1, -1, -1), nrow = 3, ncol = 3)
  
  emboss <<- matrix(c(-2, -1, 0,
                      -1, 1, 1,
                      0, 1, 2), nrow = 3, ncol = 3)
  
  Identity <<- matrix(c(0, 0, 0,
                        0, 1, 0,
                        0, 0, 0), nrow = 3, ncol = 3)
}
LoadKernels()

UnSharpMasking <- function(image, weight){
  Gaus = makeBrush(size = 51, shape = "gaussian", sigma = 1)
  sharpimage = image + (image - filter2(image, Gaus)) * weight
}

sobel <- function(image){
  sobel_x = matrix(c(1, 0, -1,
                     2, 0, -2,
                     1, 0, -1), nrow = 3, ncol = 3)
  sobel_y = matrix(c(1, 2, 1,
                     0, 0, 0,
                     -1, -2, -1), nrow = 3, ncol = 3)
  sobelimage = sqrt(filter2(image, sobel_x)^2 + filter2(image, sobel_y)^2)
}

Gaus <- function(image, size, sigma){
  gausImage = filter2(image, filter = makeBrush(size = size, shape = "gaussian", sigma = sigma))
}

DPO = function(Mask, Image, col = NULL){
  display(paintObjects(Mask, Image, col = ifelse(is.null(col), "yellow", col)))
}

indObj = function(Image, operation, size, shape, label = F){

  op = function(Mask, Image, operation, size, shape){
    CurObj = do.call(operation, list(Image == Mask,
                                     kern = makeBrush(size = size, shape = shape)))  
  }

  NewImage = map(1:max(Image), ~ op(.x, Image, operation, size, shape)) %>% 
    reduce(`+`) %>% 
    {if(label == T) bwlabel(.) else . }
  NewImage

}

SingObjOp = function(Image, operation, size, shape){
  Image = map(1:max(Image), ~ ifelse(Image == .x,
                                     do.call(operation, list(Image == .x,
                                                             kern = makeBrush(size = size, shape = shape))),
                                     0))
  sumAll = function(Image){
    ImageTemp = Image[[1]]
    for (i in 1:length(Image)) {
      ImageTemp = ImageTemp + ifelse(Image[[i]] > 0, i, 0)
    }
    Image = ImageTemp
  }
  Image = sumAll(Image)
  Image
}

unit_vec <- function(obj, vec){
  return(obj / (sqrt(sum(vec^2, na.rm = T))))
}

indDM = function(Mask, Image, operation){
  CurObj = do.call(operation, list(Image == Mask))
}

relabel = function(Image, Mask){
  Image = bwlabel(Image == Mask)
}

array2list = function(array, length){
  array = map(1:max(array), ~ array)
}

indWat = function(Image, split = F, unit = F, tol = NULL, ext = NULL){

  orig = Image
  if(is.null(tol)){ tol = 1 } else { tol = tol } 
  if(is.null(ext)){ ext = 1 } else { ext = ext }
  
  if(unit == F){
    if(split == F){ # new distance map
    Image = map(1:max(Image), ~ indDM(.x, Image, distmap)) %>% 
      reduce(`+`)
    
    } else if(split == T) { # new instances
      NI = map(1:max(Image), ~ indDM(.x, Image, distmap)) %>% 
        map(watershed, tol, ext) %>% 
        reduce(`+`)
          
      oldOnly = ifelse(NI == 1, NI, 0) %>% 
        bwlabel()
          
      newOnly = ifelse(NI > 1, NI, 0)
      
      newOnlySeq = map(1:max(newOnly), ~ relabel(newOnly, .x)) 
      newOnlySeq = map2(newOnlySeq, cumsum(lapply(newOnlySeq, max)), ~ ifelse(.x > 0, .x + .y, 0) ) %>% 
        reduce(`+`) %>% 
        {ifelse(. > 0, . + max(oldOnly), 0)}
    
      Image = oldOnly + newOnlySeq
          
    } else if (unit == T){
      NDM = map(1:max(Image), ~ indDM(.x, Image, distmap)) %>% 
        map(unit_vec, orig) %>% 
        reduce(`+`)
    }
  } else {
    NDM = map(1:max(Image), ~ indDM(.x, Image, distmap)) %>% 
      map(unit_vec, orig) %>% 
      reduce(`+`)
  }
}

SingObjWat = function(Image, distmap, tol, ext){
  Image = map(1:max(Image), ~ ifelse(Image == .x, .x, 0))
  if(distmap == T){
    Image = base::lapply(Image, distmap)
  }
  Image = lapply(Image, watershed, tol, ext)
  
  sumAll = function(Image){
    ImageTemp = Image[[1]]
    for (i in 1:length(Image)) {
      ImageTemp = ImageTemp + Image[[i]]
    }
    Image = ImageTemp
  }
  Image = sumAll(Image)
  Image
}

SingObjUnit = function(Image, unit = T){
  orig = Image
  Image = map(1:max(Image), ~ ifelse(Image == .x, .x, 0))
  Image = lapply(Image, distmap)
  if(unit == T){
    Image = lapply(Image, unit_vec, orig)
  }
  sumAll = function(Image){
    ImageTemp = Image[[1]]
    for (i in 1:length(Image)) {
      ImageTemp = ImageTemp + Image[[i]]
    }
    Image = ImageTemp
  }
  Image = sumAll(Image)
  Image
}

preprocess_masks <- function(encoding, old_shape, new_shape, labels = FALSE){
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

rle2masks <- function(encodings, shape, labels = FALSE, cropping = NULL) {
  
  ## Convert rle encoded mask to image
  rle2mask <- function(encoding, shape){
    
    splitted <- as.integer(str_split(encoding, pattern = "\\s+", simplify=TRUE))
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
  
  if(!is.null(cropping)){
    masks <- masks[cropping, cropping]
  } 
  
  masks
}

#####