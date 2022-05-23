convert_to_instance_seg = function(arr,
                                   spacing = c(0.125, 0.125),
                                   small_center_threshold = 30L,
                                   isolated_border_as_separate_instance_threshold = 15){
    
    # for testing
    # arr = Y_hat_val[1,,,]
    # spacing = c(0.125, 0.125)
    # small_center_threshold = 30L
    # isolated_border_as_separate_instance_threshold = 15

    # we first identify centers that are too small and set them to be border. This should remove false positive instances
    objects = EBImage::bwlabel(arr[,,2] > 0.5)
    for(o in 1:max(objects)){
        if(o > 0 & sum(objects == o) <= small_center_threshold){
            arr[,,3][objects == o] = arr[,,2][objects == o]
            arr[,,2][objects == o] = 0
        }
    }
    
    # 1 is core, 2 is border
    objects = EBImage::bwlabel(arr[,,2] > 0.5)
    final = objects
    remaining_border = (arr[,,3] > 0.5)
    current = objects
    dilated_mm = c(0, 0)
    spacing = spacing

    while(sum(remaining_border) > 0){
        strel_size = c(0, 0)
        maximum_dilation = max(dilated_mm)
        
        for(i in 1:2){
            if(spacing[i] == min(spacing)){
                strel_size[i] = 1
                next
            }
            if((dilated_mm[i] + spacing[i] / 2) < maximum_dilation){
                strel_size[i] = 1
            }
        }
        ball_here = array(EBImage::makeBrush(3, shape = "diamond"), dim = c(3, 3))

        if(strel_size[1] == 0){ 
            ball_here = array(ball_here[2,], dim = c(1, 3))
        }
        if(strel_size[2] == 0){
            ball_here = array(c(1,1,1), dim = c(3, 1))
        }
        
        #print(1)
        dilated = EBImage::dilate(current, kern = ball_here)
        diff = (current == 0) & (dilated != current)
        final[diff & remaining_border] = dilated[diff & remaining_border]
        remaining_border[diff] = 0
        current = dilated

        for(i in 1:2){
            if(strel_size[i] == 1){
                dilated_mm[i] = dilated_mm[i] + spacing[i]
            } else {
                dilated_mm[i] = dilated_mm[i]
            }
        }
        
        # print(paste0(" || remaining border :", sum(remaining_border), " || strel_size: ", strel_size))
    
    }

    # what can happen is that a cell is so small that the network only predicted border and no core. This cell will be
    # fused with the nearest other instance, which we don't want. Therefore we identify isolated border predictions and
    # give them a separate instance id
    # we identify isolated border predictions by checking each foreground object in arr and see whether this object
    # also contains label 1
    max_label = max(final)
    one_hot_arr = ifelse(arr[,,2] > 0.5, 1, 0) + ifelse(arr[,,3] > 0.5, 2, 0)
    foreground_objects = bwlabel(one_hot_arr)  
    
    for(i in 1:max(foreground_objects)){
        if(!(1 %in% unique(one_hot_arr[foreground_objects==i]))){
            size_of_object = sum(foreground_objects==i)
            if(size_of_object >= isolated_border_as_separate_instance_threshold){ 
                final[foreground_objects == i] = max_label + 1
                max_label = max_label + 1
                # display(final == (max_label))
                #print('yeah boi')
            }
            # print(paste0(i, "  size = ", size_of_object))
        }
    }

    return(final)
}

# display(abind(
#     normalize(X_val[1,,,]),
#     colorLabels(Y_val[1,,,2]),
#     colorLabels(final),
#     along = 1
# ))
