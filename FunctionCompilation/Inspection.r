# Function for systematic inspection of predictions

make_merge <- function(Y, image_number){
  if(CLASS == 1){
    Merge = rgbImage(blue = Y[image_number,,,1])
    Merge
  } else if(CLASS == 2){
    Merge = rgbImage(blue = Y[image_number,,,1],
                     green = Y[image_number,,,2])
    Merge
  } else if(CLASS == 3) {
    Merge = rgbImage(blue = Y[image_number,,,1], 
                     green = Y[image_number,,,2],
                     red = Y[image_number,,,3])
    Merge
  }
}

make_merge_image_mask <- function(X, Y, image_number, InputRange){
  if(CLASS ==1){
    Merge = rgbImage(blue = EBImage::normalize(X[image_number,,,], inputRange = InputRange),
                     green = Y[image_number,,,1])
    Merge
  }else if(CLASS == 2){
    Merge = rgbImage(blue = EBImage::normalize(X[image_number,,,], inputRange = InputRange),
                     green = Y[image_number,,,2])
    Merge
  } else if(CLASS == 3) {
    Merge = rgbImage(blue = EBImage::normalize(X[image_number,,,], inputRange = InputRange), 
                     green = Y[image_number,,,2],
                     red = Y[image_number,,,3])
    Merge
  }
}

make_montage <- function(X, Y, image_number, InputRange = NULL){
  if(CLASS == 1){
    Montage = Image(abind(Y[image_number,,,1],
                          EBImage::normalize(X[image_number,,,], inputRange =  unlist(ifelse(is.null(InputRange), list(list(range(X[image_number,,,])), InputRange)))), along = 1))
    Montage
  } else if(CLASS == 2){
    Montage = Image(abind(Y[image_number,,,1],
                          Y[image_number,,,2],
                          EBImage::normalize(X[image_number,,,], inputRange =  unlist(ifelse(is.null(InputRange), list(list(range(X[image_number,,,])), InputRange)))), along = 1))
    Montage
  } else if(CLASS == 3) {
    Montage = Image(abind(Y[image_number,,,1],
                          Y[image_number,,,2],
                          Y[image_number,,,3],
                          EBImage::normalize(X[image_number,,,], inputRange =  unlist(ifelse(is.null(InputRange), list(list(range(X[image_number,,,])), InputRange)))), along = 1))
    Montage
  }
}

combine_col <- function(image_1 = NULL,
                        image_2 = NULL,
                        image_3 = NULL,
                        image_4 = NULL,
                        color_1,
                        color_2,
                        color_3,
                        color_4,
                        dimension){
  if(!is.null(image_1)){
    jet.colors.b = colorRampPalette(c("black", as.character(color_1)))
    col_b = colormap(image_1, jet.colors.b(256L))
  } else {
    col_b <- Image(data = array(0, dim = c(dimension, 3)), colormode = Color)
  }
  
  if(!is.null(image_2)){
    jet.colors.g = colorRampPalette(c("black", as.character(color_2)))
    col_g = colormap(image_2, jet.colors.g(256L))
  } else {
    col_g <- Image(data = array(0, dim = c(dimension, 3)), colormode = Color)
  }
  
  if(!is.null(image_3)){
    jet.colors.r = colorRampPalette(c("black", as.character(color_3)))
    col_r = colormap(image_3, jet.colors.r(256L))
  } else {
    col_r <- Image(data = array(0, dim = c(dimension, 3)), colormode = Color)
  }
  
  if(!is.null(image_4)){
    jet.colors.c = colorRampPalette(c("black", as.character(color_4)))
    col_c = colormap(image_4, jet.colors.c(256L))
  } else {
    col_c <- Image(data = array(0, dim = c(dimension, 3)), colormode = Color)
  }
  
  image <- Image(data = array(0, dim = c(dimension, 3)), colormode = Color)
  image[,,1] <- col_b[,,1] + col_g[,,1] + col_r[,,1] + col_c[,,1]
  image[,,2] <- col_b[,,2] + col_g[,,2] + col_r[,,2] + col_c[,,2]
  image[,,3] <- col_b[,,3] + col_g[,,3] + col_r[,,3] + col_c[,,3]
  image
}

## source code use for viridis transformation
## https://stackoverflow.com/questions/61460482/converting-and-saving-grayscale-image-to-e-g-viridis-colour-scheme
library(viridisLite)

img_to_viridis <- function(image){

  if(length(dim(image)) == 3){
    intmat = image[,,1] * 255  
  } else {
    intmat = image * 255
  }
  
  virmat = viridisLite::viridis(256)[intmat + 1]
  virmat = c(substr(virmat, 2, 3),
              substr(virmat, 4, 5),
              substr(virmat, 6, 7)
              )
  virmat = as.numeric(as.hexmode(virmat)) / 255
  
  dim(virmat) <- c(dim(intmat), 3)
  virmat = Image(virmat, colormode = "Color")
  return(virmat)

}
