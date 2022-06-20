source(paste0(RelPath, "/Scripts/FunctionCompilation/connected_component_analysis.r"))

instance_gen = function(hat){
  temp = apply(hat, 1, convert_to_instance_seg, simplify = "array")
  temp = simplify2array(temp)
  temp = aperm(temp, c(3, 1, 2))
  temp = add_dim(temp, dim3 = 1) 
  return(temp)
}

Y_val_label = instance_gen(Y_val)
# display(colorLabels(Y_val_label[1,,,1]))
Y_hat_val_label = instance_gen(Y_hat_val)
# display(Y_hat_val_label[1,,,1])

idx = 1
dice_metric_instance <- function(y_true, y_pred, pixels_4_split_merge=100){
    
   
    if(!is.list(y_true)){
        num_imgs <- dim(y_true)[1]
    }
    
    scores <- tibble()
    
    # for(i in 1:num_imgs){

        scores_temp = tibble()
        
        y_true_i = array(y_true[idx,,,], dim = dim(y_true)[2]*dim(y_true)[3])
        y_pred_i = array(y_pred[idx,,,], dim = dim(y_pred)[2]*dim(y_pred)[3])
        
        scores_instances_i = array(0, max(y_true_i))
        inst_pred_order = array(0, max(y_true_i))
        inst_true_order = array(0, max(y_true_i))
        
        oversplit = array(0, max(y_true_i))
        undersplit = array(0, max(y_pred_i))
        missed = array(0, max(y_true_i))
        extra = array(0, max(y_pred_i))

        for(j in 1:max(y_true_i)) {
 
            start_time = Sys.time()     
            # print(j)

            # from prediction's side
            # if there isn't a predicted objected or two small get 1 missed
            if(length(y_pred_i[y_true_i == j & y_pred_i > 0]) < pixels_4_split_merge){
                oversplit[j] = 0
                missed[j] = 1
            # if there is more than one predicted label for one true label and object not too small
            } else {
                if( length(unique(y_pred_i[y_true_i == j & y_pred_i > 0])) > 1){
                    unq = unique(y_pred_i[y_true_i == j & y_pred_i > 0])
                    unq1 = length(y_pred_i[y_true_i == j & y_pred_i == unq[1]]) # only checking two first labels for oversplit
                    unq2 = length(y_pred_i[y_true_i == j & y_pred_i == unq[2]])
                    if(unq1 > pixels_4_split_merge & unq2 > pixels_4_split_merge){
                        oversplit[j] = length(unique(y_pred_i[y_true_i == j & y_pred_i > 0]))-1        
                        missed[j] = 0
                    } else {
                        oversplit[j] = 0
                        missed[j] = 0
                    } 
                } else {
                    oversplit[j] = 0
                    missed[j] = 0
                }
            }
            
            scores_instances_j = array(0, dim = c(max(y_true_i), max(y_pred_i)))
            
            for(k in 1:max(y_pred_i)) {
                # k = 1
                if(length(y_true_i[y_pred_i == k & y_true_i > 0]) < pixels_4_split_merge){
                    undersplit[k] = 0
                    extra[k] = 1
                # if there is more than one true label for one pred label and object not too small
                } else {
                    if( length(unique(y_true_i[y_pred_i == k & y_true_i > 0])) > 1){
                        unq = unique(y_true_i[y_pred_i == k & y_true_i > 0])
                        unq1 = length(y_true_i[y_pred_i == k & y_true_i == unq[1]]) # only checking two first labels for oversplit
                        unq2 = length(y_true_i[y_pred_i == k & y_true_i == unq[2]])
                        if(unq1 > pixels_4_split_merge & unq2 > pixels_4_split_merge){
                            undersplit[k] = length(unique(y_pred_i[y_true_i == k & y_pred_i > 0]))-1        
                            extra[k] = 0
                        } else {
                            undersplit[k] = 0
                            extra[k] = 0
                        } 
                    } else {
                        undersplit[k] = 0
                        extra[k] = 0
                    }
                }

                scores_instances_j[j,k] = dice_coef(ifelse(y_true[idx,,,] == j, 1, 0),
                                                    ifelse(y_pred[idx,,,] == k, 1, 0)) %>% as.numeric()
            }
                        
            id_all = ifelse(idx == 1, j, sum(apply(y_true[1:(idx-1),,,1], 1, max)) + j)

            if(sum(y_true[idx,,,] == j) != 0 & max(scores_instances_j) == 0){
                inst_true_order[j] = j
                inst_pred_order[j] = 0
                scores_instances_i[j] = 0
            } else {
                inst_true_order[j] = j
                inst_pred_order[j] = which(scores_instances_j[j,] == max(scores_instances_j[j,]))
                scores_instances_i[j] = max(scores_instances_j[j,])
            }
            
        }

        temp_true = computeFeatures.shape(y_true[idx,,,]) %>% 
            as_tibble() %>% 
            rownames_to_column(var = "inst_true") %>% 
            mutate(image_id = i, 
                   dice_score = scores_instances_i,
                   inst_true = as.double(inst_true),
                   inst_pred = inst_pred_order,
                   oversplit = oversplit,
                   missed = missed)
        temp_pred = computeFeatures.shape(y_pred[idx,,,]) %>% 
            as_tibble() %>% 
            rownames_to_column(var = "inst_pred") %>% 
            mutate(image_id = i,
                   inst_pred = as.double(inst_pred),
                   undersplit = undersplit,
                   extra = extra)

        temp = full_join(temp_true, temp_pred, suffix = c(".true", ".pred"), by = c("image_id" = "image_id",
                                                                                  "inst_pred" = "inst_pred")) %>%
                select(inst_true, inst_pred, dice_score, oversplit, undersplit, missed, extra, everything())
            #    mutate(oversplit = oversplit,
            #           undersplit = undersplit)
        # as.data.frame(temp)
        # if(fov_empty == 1){
        #     temp = select_if(temp, is.numeric) %>% # to create empty table
        #         map(function(x) x = NA)
        # }

        scores = rbind(scores, temp)

        end_time = Sys.time()
        run_time = end_time - start_time

        print(paste0("image # ", i, " out of ", length(list_files), " images analysed",
                 " || well: ", Well, " ID: ", ID, 
                     " || # labels: ",  ifelse(str_length(max(y_true_i)) == 1 , paste0("0",max(y_true_i)), max(y_true_i)),
                     " || compute time: ", round(as.numeric(run_time), digits = 2) * 60, " s"))

        # return(temp)
    # }

    return(scores)
    # return(scores_inst)
}

single_dice_val = dice_metric_instance(Y_val_label, Y_hat_val_label)

save(single_dice_val, file = paste0(Save_data, "/single_dice_", i, "_ID--", ID, "_WELL--", Well ,".RData"))

arraynum = function(image){
        return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}


make_montage = function(img, true, pred, Id, border = FALSE){

    source = img[Id,,,]
    gt = true[Id,,,1]
    pd = pred[Id,,,]

    source_rgb = combine_col(image_1 = EBImage::normalize(source),
                             color_1 = "gray",
                             dimension = dim(source))
    range(source_rgb)
    
    if(border == FALSE){
        im1 = source_rgb + colorLabels(gt)/6
        im2 = source_rgb + colorLabels(pd)/6
    } else {
        im1 = paintObjects(gt, source_rgb, col = "red")
        im2 = paintObjects(pd, source_rgb, col = "red")
    }
 
    montage = abind(source_rgb, im1,  im2,
     along = 1)
    writeImage(montage, files = paste0(Save_image_instance, "/", "Montage_", i,"_", ifelse(border == TRUE, "border", "mask"), ".png"),
                    quality = 100, type = "png")

}

# idx = 2
# for(idx in 1:10){
    make_montage(X_val, Y_val_label, Y_hat_val_label, idx)
    make_montage(X_val, Y_val_label, Y_hat_val_label, idx, border = TRUE)
# }
# for(img_num in 1:5){
    ## Val
    instance_gt_pred_val = abind(
        normalize(colorLabels(Y_val_label[idx,,,1])),
        normalize(colorLabels(Y_hat_val_label[idx,,,])),
        along = 1
    )
    writeImage(instance_gt_pred_val, files = paste0(Save_image_instance, "/instance_gt_pred_", i,"_ID--", ID, "_WELL--", Well ,".png"))
    
    # mask_gt_pred_val = abind(
    #     EBImage::rgbImage(red = Y_val[idx,,,2]),
    #     EBImage::rgbImage(red = Y_hat_val[idx,,,2] > 0.5),
    #     along = 1
    # )
    # writeImage(mask_gt_pred_val, files = paste0(Save_image_instance, "/mask_gt_pred_", i, "_ID--", ID, "_WELL--", Well ,".png"))

    ## Test
    
    # instance_gt_pred_test = abind(
    #     normalize(colorLabels(Y_test_label[img_num,,,1])),
    #     normalize(colorLabels(Y_hat_test_label[img_num,,,])),
    #     along = 1
    # )
    # writeImage(instance_gt_pred_test, files = paste0(Save_image_instance, "/instance_gt_pred_test_", img_num, ".png"))
    
    # mask_gt_pred_test = abind(
    #     EBImage::rgbImage(red = Y_test[img_num,,,2], green = Y_test[img_num,,,3]),
    #     EBImage::rgbImage(red = Y_hat_test[img_num,,,2] > 0.5, green = Y_hat_test[img_num,,,3] > 0.5),
    #     along = 1
    # )
    # writeImage(mask_gt_pred_test, files = paste0(Save_image_instance, "/mask_gt_pred_test_", img_num, ".png"))
# }

