### Preprocessing

# A little preprocessing is performed on the original images, i.e. resizing, inversion of brightfield images and normalization:
Rescale <- function(x){
  Bottom = 2^15/(2^16-1) 
  Top = (2^15+4095)/(2^16-1)
  (x - Bottom) / (Top - Bottom)   
}

ctr_std <- function(x,
                    ctr = TRUE,
                    ctr_input = NULL,
                    std = TRUE,
                    std_input = NULL
                    ){
  if(ctr == TRUE){
    if(!is.null(ctr_input)){
      x = x - ctr_input
    } else {
      x = x - mean(x)
    }
  }
  if(std == TRUE){
    if(!is.null(std_input)){
      x = x / std_input
    } else {
      x = x / sd(x)
    }
  }
}

invert <- function(x) {
  if(mean(x) > .5)
    x <- 1 - x
  x
}

to_gray_scale <- function(x) {
  y <- rgbImage(red = getFrame(x, 1),
                green = getFrame(x, 2),
                blue = getFrame(x, 3))
  y <- channel(y, mode="luminance")
  dim(y) <- c(dim(y), 1)
  y
}

preprocess_image <- function(file, shape){
  image <- readImage(file, type = "tiff")
  image <- Rescale(image)
  #image <- to_gray_scale(image)                        ## convert to gray scale  
  image <- resize(image, w = shape[1], h = shape[2])    ## make all images of dimensions
  #image <- clahe(image)                                ## local adaptive contrast enhancement
  #image <- normalize(image)                            ## standardize between [0, 1]
  #image <- invert(image)                               ## invert brightfield
  dim(image) <- c(dim(image), 1)                        ## because grayscale only 1d
  imageData(image)                                      ## return as array
}

preprocess_image_TL <- function(file, shape, cropping = NULL, rescaling = NULL){
  # file = train_data$img1[1]
  # shape = c(1000, 1000)
  if(!is.null(cropping)){
    if(length(cropping) == 1){
    image <- readImage(file, type = "tiff")[1:cropping, 1:cropping]
    } else if(length(cropping) == 4){
      image <- readImage(file, type = "tiff")[cropping[1]:cropping[2], cropping[3]:cropping[4]]
    }
  } else {
    image <- readImage(file, type = "tiff")
  }
  # display(normalize(image))
  # str(image)
  if(!is.null(rescaling)){
    image <- Rescale(image)
  }
  # image <- Rescale(image)
  #image <- to_gray_scale(image)                        ## convert to gray scale  
  image <- resize(image, w = shape[1], h = shape[2])    ## make all images of dimensions
  #image <- clahe(image)                                ## local adaptive contrast enhancement
  #image <- normalize(image)                            ## standardize between [0, 1]
  #image <- invert(image)                               ## invert brightfield
  dim(image) <- c(dim(image), 1)                        ## because grayscale only 1d
  imageData(image)                                      ## return as array
}

swap_channel = function(image, new_channel, channel_index){
  image[,,channel_index] = new_channel
  image
}

combine3c = function(file1, file2, file3, shape){
  image = array(0, dim = shape)
  image[,,1] <- readImage(file1, type = "tiff")
  image[,,2] <- readImage(file2, type = "tiff")
  image[,,3] <- readImage(file3, type = "tiff")
  image
} 

crop_mut = function(image, cropping){
  if(!is.na(cropping[1])| is.na(cropping[1])){ #  | is.na(cropping[[1]]
    if(length(dim(image)) == 2){
      if(length(cropping) == 1){
        image <- image[1:cropping, 1:cropping]
        } else if(length(cropping) == 4){
          image <- image[cropping[1]:cropping[2], cropping[3]:cropping[4]]
      }
    } else {
      if(length(cropping) == 1){
        image <- image[1:cropping, 1:cropping, ]
        } else if(length(cropping) == 4){
          image <- image[cropping[1]:cropping[2], cropping[3]:cropping[4], ]
      }
    }
  }
  return(image)
}

## make an RGB image with the 3 ztacks
preprocess_image_TL_3c <- function(file1, file2, file3, shape, cropping = NULL, rescaling = NULL){

  image = array(0, dim = shape)

  file1 = train_data$img1[1]
  file2 = train_data$img2[1]
  file3 = train_data$img3[1]
  if(!is.null(cropping)){
    if(length(cropping) == 1){
    image[,,1] <- readImage(file1, type = "tiff")[1:cropping, 1:cropping]
    image[,,2] <- readImage(file2, type = "tiff")[1:cropping, 1:cropping]
    image[,,3] <- readImage(file3, type = "tiff")[1:cropping, 1:cropping]
    } else if(length(cropping) == 4){
      image[,,1] <- readImage(file1, type = "tiff")[cropping[1]:cropping[2], cropping[3]:cropping[4]]
      image[,,2] <- readImage(file2, type = "tiff")[cropping[1]:cropping[2], cropping[3]:cropping[4]]
      image[,,3] <- readImage(file3, type = "tiff")[cropping[1]:cropping[2], cropping[3]:cropping[4]]
    }
  } else {
    image[,,1] <- readImage(file1, type = "tiff")
    image[,,2] <- readImage(file2, type = "tiff")
    image[,,3] <- readImage(file3, type = "tiff")
  }
  if(!is.null(rescaling)){
    image <- Rescale(image)
  }
  
  # image <- Rescale(image)
  # image <- to_gray_scale(image)                        ## convert to gray scale  
  # image <- resize(image, w = shape[1], h = shape[2])    ## make all images of dimensions
  
  # image <- clahe(image)                                ## local adaptive contrast enhancement
  # image <- normalize(image)                            ## standardize between [0, 1]
  # image <- invert(image)                               ## invert brightfield

  # dim(image) <- c(dim(image), 1)                        ## because grayscale only 1d

  imageData(image)                                      ## return as array

}

custom_resize = function(img, new_width, new_height) {
  new_img = apply(img, 2, function(y){return (spline(y, n = new_height)$y)})
  new_img = t(apply(new_img, 1, function(y){return (spline(y, n = new_width)$y)}))

  new_img[new_img < 0] = 0
  new_img = round(new_img)

  return (new_img)
}


# Image and Mask transformation
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

resize_masks <- function(masks, shape, labels = FALSE){
  
  masks <- resize(masks, w = shape[1], h = shape[2])
  
  # if(labels) {
  #   for(i in 1:dim(masks)[3]) ##recover labeling after reshaping
  #     masks[,,i] <- i*(masks[,,i] > 0)
  # }
  
  #masks <- Reduce("+", getFrames(masks))
  
  #dim(masks) <- c(dim(masks), 1) ##masks have no color channels
  
  masks
  
}

add_dim <- function(image, dim3){
  dim(image) <- c(dim(image), dim3)
  return(image)
}

add_dim_rgb_nifti <- function(image, dim4){
  dim(image) <- c(dim(image)[1:2], dim4, dim(image)[3])
  return(image)
}

sum_channels <- function(masks, k, i, j){
  masks[,,k] = masks[,,i] + masks[,,j]
  return(masks)
}

diff_channels <- function(masks, k, i, j){
  masks[,,k] = masks[,,i] - masks[,,j]
  return(masks)
}

keep_channels <- function(masks, i, j){
  masks[,,1:2] = masks[,,c(i,j)]
  return(masks)
}

shuffle_channels <- function(masks, chan){
  masks = masks[,,chan]
  return(masks)
}

select_channels <- function(masks, i, j){
  masks = masks[,,i:j]
  return(masks)
}

softmax_transf_channel <- function(masks, i){
  masks[,,i] = 1 - masks[,,i]
  return(masks)
}

transform_gray_to_rgb_rep <- function(old_image){
  new_image = array(0, dim = c(dim(old_image)[1], dim(old_image)[2], 3))
  #str(new_image)
  new_image[,,1:3] = old_image
  #display(abind(new_image[,,1], new_image[,,2], new_image[,,3], along = 1))
  old_image = new_image
  return(old_image)
}

set_dimension <- function(image, channel){
  dim(image) <- c(dim(image)[1], dim(image)[2], channel) 
  imageData(image)  
}

### General transformations

# I use the encoded masks instead of reading them. Note that for decoding the original image dimensions are required.
# Here is a function to decode the run length encoded masks into images, combining all masks in one image (Optionally labelled image is returned):
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
    masks <- masks[1:cropping, 1:cropping]
  } 
  
  masks
}

# for transforming to tensors
list2tensor <- function(xList) {
  xTensor <- simplify2array(xList)
  aperm(xTensor, c(4, 1, 2, 3))    
}

### Data augmentation:
# Data augmentation with ONE parameter to apply to image and label
data_aug_XandY_1_param <- function(data_aug, operation, parameter, original_input){
  if(original_input == TRUE){
    # for geometric, shape trasnformation:
    if(is.null(parameter)){
      data_temp <- input %>% mutate(X = map(X, operation),
                                    Y = map(Y, operation))
      data_aug <- bind_rows(data_aug, data_temp)
    } else {
      data_temp <- input %>% mutate(X = map(X, operation, parameter),
                                    Y = map(Y, operation, parameter))
      data_aug <- bind_rows(data_aug, data_temp)
    }
  } else {
    # for intensity, texture transformation:
    if(is.null(parameter)){
      data_temp <- data_aug %>% mutate(X = map(X, operation),
                                       Y = map(Y, operation))
      data_aug <- bind_rows(data_aug, data_temp)
    } else {
      data_temp <- data_aug %>% mutate(X = map(X, operation, parameter),
                                       Y = map(Y, operation, parameter))
      data_aug <- bind_rows(data_aug, data_temp)
    }
  }
}

# Data augmentation with ONE parameter to apply to image or label only
data_aug_X_1_param <- function(data_aug, operation, parameter, original_input){
  if(original_input == TRUE){
    # for geometric, shape trasnformation:
    if(is.null(parameter)){
      data_temp <- input %>% mutate(X = map(X, operation))
      data_aug <- bind_rows(data_aug, data_temp)
    } else {
      data_temp <- input %>% mutate(X = map(X, operation, parameter))
      data_aug <- bind_rows(data_aug, data_temp)
    }
  } else {
    # for intensity, texture transformation:
    if(is.null(parameter)){
      data_temp <- data_aug %>% mutate(X = map(X, operation))
      data_aug <- bind_rows(data_aug, data_temp)
    } else {
      data_temp <- data_aug %>% mutate(X = map(X, operation, parameter))
      data_aug <- bind_rows(data_aug, data_temp)
    }
  }
}

# Data augmentation with TWO parameter to apply to image or label only
data_aug_X_2_param <- function(data_aug, operation, parameter1, parameter2, original_input){
  if(original_input == TRUE){
    # for geometric, shape trasnformation:
    if(is.null(parameter)){
      data_temp <- input %>% mutate(X = map(X, operation))
      data_aug <- bind_rows(data_aug, data_temp)
    } else {
      data_temp <- input %>% mutate(X = map(X, operation, parameter1, parameter2))
      data_aug <- bind_rows(data_aug, data_temp)
    }
  } else {
    # for intensity, texture transformation:
    if(is.null(parameter)){
      data_temp <- data_aug %>% mutate(X = map(X, operation))
      data_aug <- bind_rows(data_aug, data_temp)
    } else {
      data_temp <- data_aug %>% mutate(X = map(X, operation, parameter1, parameter2))
      data_aug <- bind_rows(data_aug, data_temp)
    }
  }
}

LoadKernels <- function(){
  sharp <<- matrix(c(0, -1, 0,
                     -1, 5, -1,
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

UnSharpMasking <- function(image, weight = 1, blur = 1){
  # Gaus = makeBrush(size = 51, shape = "gaussian", sigma = 1)
  if(weight == 0){
    image
    return(image)
  } else {
    Gaus = gblur(image, blur)
    sharpimage = image + (image - Gaus) * weight
    return(sharpimage)
  }
  
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

blur = function(image, blur_param){
  if(blur_param != 0){
    image = EBImage::gblur(image, blur_param)
  }
  image
}

mul2one = function(array, range_, axis){
  # range_ = c(4,7)
  # range_ = 3
  # array = Y_hat_val[1,,,]
  # axis = 4
  if(length(range_) > 1 | range_[1] > 1){
    if(length(range_) == 1){
      range_ = 1:range_
    }
    if(length(dim(array)) == 4){
      if(axis == 1){
        temp = array[range_[1],,,]
        for(i in (range_[1]+1):range_[length(range_)]){
          temp = abind(temp, array[i,,,], along = 1)
        }
      } else if(axis == 4){
        temp = array[,,range_[1]]
        for(i in (range_[1]+1):range_[length(range_)]){
          temp = abind(temp, array[,,i], along = 1)
        }
      }
    } else if(length(dim(array)) == 3){
      if(axis == 1){
        temp = array[range_[1],,]
        for(i in (range_[1]+1):range_[length(range_)]){
          temp = abind(temp, array[i,,], along = 1)
        }
      } else if(axis == 4){
        temp = array[,,range_[1]]
        for(i in (range_[1]+1):range_[length(range_)]){
          temp = abind(temp, array[,,i], along = 1)
        }
      }
    }
  }
  return(temp)
}