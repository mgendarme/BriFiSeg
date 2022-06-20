# source used for our version
# https://github.com/rayheberer/imgnoise/blob/master/R/gaussian_noise.R
gaussian_noise = function(image, mu = 0.0, factor = 0.1) {

    variance = factor
    standard_deviation = sqrt((range(image)[2] - range(image)[1]) * variance )
    noise = array(rnorm(dim(image)[1] * dim(image)[2] * dim(image)[3],
                    mean = mu,
                    sd = standard_deviation),
                dim = c(dim(image)[1], dim(image)[2], dim(image)[3]))
    image = image + noise
    return(image)

}

# source used for our version
# https://github.com/MIC-DKFZ/batchgenerators
gamma_correction = function(image, factor = 1, epsilon=1e-7){

    ## gama range = c(0.5, 2)
    gamma = factor
    minm = min(image)
    rnge = max(image) - minm
    image = (((image - minm) / (rnge + epsilon)) ^ gamma) * rnge + minm
    return(image)
  # }
}

# source used for our version
# https://github.com/MIC-DKFZ/batchgenerators
contrast_aug = function(image, factor){

    ## contrast_range = c(0.75, 1.25) / c(.5, 1.5)?
    mn = mean(image)
        # if preserve_range:
        #    minm = data_sample[c].min()
        #    maxm = data_sample[c].max()
    image = (image - mn) * factor + mn
    return(image)
  # }
}

brightness_aug = function(image, factor){

    ## multiplier_range=(0.5, 2)
    image = image * factor
    return(image)

}

blur = function(image, factor){

    if(factor != 0){
      image = EBImage::gblur(image, factor)
    }
    return(image)

}

aug_int_list = function(image, param){
    if(is.na(param$identity)){

        if(!is.na(param$gamma_correction)){
            image = gamma_correction(image, param$gamma_correction)
        } 
        if(!is.na(param$brightness_aug)){
            image = brightness_aug(image, param$brightness_aug)
        } 
        if(!is.na(param$contrast_aug)){
            image = contrast_aug(image, param$contrast_aug)
        }
        if(!is.na(param$blur)){
            image = blur(image, param$blur)
        }
        if(!is.na(param$gaussian_noise)){
            image = gaussian_noise(image, param$gaussian_noise)
        }    
    }
  return(image)
}


custom_generator <- function(data,
                             shuffle,
                             scale, # typically 0.2
                             # zoom_out, #c(T or F) add this parameter if necessary/useful
                             intensity_operation = FALSE,
                             batch_size) {
  
  # data = train_input
  # shuffle = TRUE
  # scale = 0.2
  # intensity_operation = T
  # batch_size = BATCH_SIZE
  
  i <- 1
  function() {
    if(shuffle) {
      indices <- sample(1:nrow(data), size = batch_size)
    } else { 
      if (i + batch_size >= nrow(data) ) 
        i <<- 1
      indices <- c(i:min(i + batch_size - 1, nrow(data)))
      i <<- i + length(indices)
    }
    
    ## generate random batch from input data
    input_iter <- data[indices,]
    
    ## generate the range of possible parameters for the augmentation
    # morpholical operations parameters
    random_crop = sample(c(0, 1), batch_size, prob = c(0.8, 0.2), replace = T)
    random_scale = sample(seq(1, 1+scale, by = 0.01), batch_size, replace = T)
    
    # morpholical operations parameters for 512 size images
    random_crop_lim_h_l = seq(0, WIDTH*(scale/2), by = 1)
    random_crop_lim_h_r = seq(round(WIDTH - WIDTH*(scale/2), 0), WIDTH, by = 1)
    random_crop_lim_v_t = seq(0, WIDTH*(scale/2), by = 1)
    random_crop_lim_v_b = seq(round(WIDTH - WIDTH*(scale/2), 0), WIDTH, by = 1)
    
    # morpholical operations parameters
    gaussian_noise_lim = seq(0, 0.1, 0.01)
    gamma_correction_lim = seq(0.7, 1.5, .01)
    brightness_aug_lim = seq(0.7, 1.3, .01)
    contrast_aug_lim = seq(0.65, 1.5, .01)
    blur_lim = seq(1.0, 1.5, 0.01)

    probs =  list(identity = 0.2,
                  crop = 0.2,
                  gaussian_noise = 0.2,     #0.15,
                  gamma_correction = 0.2,   #0.15,
                  brightness_aug = 0.2,     #0.15,
                  contrast_aug = 0.2,       #0.15,
                  blur = 0.2
               )

    param_int_generator = function(){
      return(
        list(  
        identity = sample(c(NA, 1), 1, prob = c(1 - probs$identity, probs$identity), replace = T),
        crop = sample(c(NA, 1), 1, prob = c(1 - probs$crop, probs$crop), replace = T),
        gaussian_noise = sample(c(NA, sample(gaussian_noise_lim, 1, replace = T)), 1,
                            prob = c(1 - probs$gaussian_noise, probs$gaussian_noise), replace = T),
        gamma_correction = sample(c(NA, sample(gamma_correction_lim, 1, replace = T)), 1,
                            prob = c(1 - probs$gamma_correction, probs$gamma_correction), replace = T),
        brightness_aug = sample(c(NA, sample(brightness_aug_lim, 1, replace = T)), 1,
                            prob = c(1 - probs$brightness_aug, probs$brightness_aug), replace = T),
        contrast_aug = sample(c(NA, sample(contrast_aug_lim, 1, replace = T)), 1,
                            prob = c(1 - probs$contrast_aug, probs$contrast_aug), replace = T),
        blur = sample(c(NA, sample(blur_lim, 1, replace = T)), 1,
                            prob = c(1 - probs$blur, probs$blur), replace = T)
        )
      )
    }

    # perform morphological and intensity based transformations
    # input_iter$crop_range[[1]]
    # crop_mut(train_input$X[[1]], input_iter$crop_range[[1]])
    
    input_iter <- input_iter %>%
        mutate(       
            ### generate parameters
            ## for 1 cropping, 2, rotating, 3, flipping and flopping
            crop_shift_h_l = replicate(batch_size, sample(random_crop_lim_h_l, 1)),
            crop_shift_h_r = replicate(batch_size, sample(random_crop_lim_h_r, 1)),
            crop_shift_v_b = replicate(batch_size, sample(random_crop_lim_v_b, 1)),
            crop_shift_v_t = replicate(batch_size, sample(random_crop_lim_v_t, 1)),
            # crop_lim = replicate(batch_size, sample(random_crop, 1)),
            crop_range = pmap(list(crop_shift_h_l,
                                   crop_shift_h_r,
                                   crop_shift_v_t,
                                   crop_shift_v_b), c),
            rotation_shift = replicate(batch_size, sample(c(0, 90, 180, 270), 1)),
            flip_shift = replicate(batch_size, sample(c(0, 1), 1)),
            flop_shift = replicate(batch_size, sample(c(0, 1), 1)),

            # add this if crop only with certain probability
            # crop = param_df$crop,
            # crop_null = list(c(1, 512, 1, 512)),
            # crop_range = map_if(crop_range, crop == 0, function(x) x = c(1, 512, 1, 512)),
            # scale = param_df$scale_param,
            # scale = unlist(map(scale, ~ ifelse(.x < 1, .x, 1))),
            
            ## useful for debugging
            # Xaug = map(X, ~ crop_mut(.x, cropping = 512)),
            # Yaug = map(Y, ~ crop_mut(.x, cropping = 512)),
            # input_iter$crop_range[[1]]
            
            ### perform the augmentation
            ## cropping + zoom in or zoom out
            Xaug = map2(X, crop_range, ~ crop_mut(.x, .y)),
            Xaug = map(Xaug, EBImage::resize, WIDTH, HEIGHT),
            Yaug = map2(Y, crop_range, ~ crop_mut(.x, .y)),
            Yaug = map(Yaug, EBImage::resize, WIDTH, HEIGHT),
            ## rotate      
            Xaug = map2(Xaug, rotation_shift, ~ EBImage::rotate(.x, .y)),
            Yaug = map2(Yaug, rotation_shift, ~ EBImage::rotate(.x, .y)),
            ## flip & flop
            Xaug = map_if(Xaug, flip_shift == 1, EBImage::flip),
            Xaug = map_if(Xaug, flop_shift == 1, EBImage::flop),
            Yaug = map_if(Yaug, flip_shift == 1, EBImage::flip),
            Yaug = map_if(Yaug, flop_shift == 1, EBImage::flop),
            ## if necessary reshape the iamges to appropriate dimensions
            dimXaug = map(Xaug, dim),
            dimXaug = map(dimXaug, length),
            Xaug = map_if(Xaug, dimXaug == 2, ~ add_dim(.x, 1)),
            dimYaug = map(Yaug, dim),
            dimYaug = map(dimYaug, length),
            Yaug = map_if(Yaug, dimYaug == 2, ~ add_dim(.x, 1))            
            ) %>% 
        select(Xaug, Yaug)
    
    ## add operations for source image only
    if(intensity_operation == TRUE) {
        input_iter <- input_iter %>%
            mutate(
                Param = NA,
                Param = map(Param, ~ param_int_generator()),
                Xaug = map2(Xaug, Param, ~ aug_int_list(.x, .y))
                # shuffle channels ?
                # Chan_shuffle = replicate(batch_size, sample(c(1:CHANNELS)), simplify = F),
                # Xaug = map2(Xaug, Chan_shuffle, ~ shuffle_channels(.x, .y)),
                ) %>%
            select(Xaug, Yaug)
    } 

    X <- list2tensor(input_iter$Xaug)
    Y <- list2tensor(input_iter$Yaug)

    list(X, Y)

  }
}

#############################
#############################