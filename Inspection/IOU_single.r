pred2label = function(image, thresh){
    pred_label_init = image > thresh
    pred_label_temp = erode(pred_label_init, kern = makeBrush(size = 15, shape = c("disc")))
    pred_label_temp = bwlabel(pred_label_temp)
    pred_label_temp_ws = watershed(distmap(pred_label_temp))
    pred_label_temp2 = propagate(x = image, seeds = pred_label_temp_ws, mask = image > thresh)
    return(pred_label_temp2)
}

orig_O_pred = function(y_true_label, y_pred_label){
    orig_over_pred = ifelse(y_true_label > 0, 1, 0) + 
        ifelse(y_pred_label > 0, 2, 0)
    orig_over_pred_rgb = rgbImage(blue = (orig_over_pred == 3),   # Y and Yhat overlap 
                                  red = (orig_over_pred == 1),    # under segmented
                                  green = (orig_over_pred == 2))  # over segmented
    return(orig_over_pred_rgb)
}

iou_thresh <- c(seq(0.01, 0.99, 0.01))

iou <- function(y_true, y_pred){
    # y_true = Y_train[,,,1]
    # y_pred = Y_hat[,,,1]
    
    intersection <- sum((y_true * y_pred)>0)
    union <- sum((y_true + y_pred)>0)
    
    if(union == 0){
        return(union)
    }
    
    return(intersection/union)

}

iou_metric_instance <- function(y_true_, y_pred_){
    
    ## for testing
    # y_true_ = Y_test_label
    # y_pred_ = Y_hat_test_label
    
    y_true = y_true_[,,,1]
    y_pred = y_pred_[,,,1]
    
    if(!is.list(y_true)){
        num_imgs <- dim(y_true)[1]
    }
    
    scores <- tibble()
    
    for(i in 1:num_imgs){
        
        scores_temp = tibble()
        
        y_true_i = array(y_true[i,,], dim = dim(y_true)[2]*dim(y_true)[3])
        y_pred_i = array(y_pred[i,,], dim = dim(y_pred)[2]*dim(y_pred)[3])
        
        scores_instances_i = array(0, max(y_true_i))
        inst_pred_order = array(0, max(y_true_i))
        inst_true_order = array(0, max(y_true_i))
        
        for(j in 1:max(y_true_i)) {
            
            scores_instances_j = array(0, dim = c(max(y_true_i), max(y_pred_i)))
            
            for(k in 1:max(y_pred_i)) {
                scores_instances_j[j,k] = iou(y_true[i,,] == j, y_pred[i,,] == k)
            }
            
            id_all = ifelse(i == 1, j, sum(apply(y_true_[1:(i-1),,,1], 1, max)) + j)
            if(sum(y_true[i,,] == j) != 0 & max(scores_instances_j) == 0){
                inst_true_order[j] = j
                inst_pred_order[j] = 0
                scores_instances_i[j] = 0
            } else {
                inst_true_order[j] = j
                inst_pred_order[j] = which(scores_instances_j[j,] == max(scores_instances_j[j,]))
                scores_instances_i[j] = max(scores_instances_j[j,])
            }
            
        }
        
        temp_true = computeFeatures.shape(y_true[i,,]) %>% 
            as_tibble() %>% 
            rownames_to_column(var = "inst_true") %>% 
            mutate(image_id = i, 
                   iou_score = scores_instances_i,
                   inst_pred = inst_pred_order)
        temp_pred = computeFeatures.shape(y_pred[i,,]) %>% 
            as_tibble() %>% 
            rownames_to_column(var = "inst_pred") %>% 
            mutate(image_id = i,
                   inst_pred = as.double(inst_pred))

        temp = full_join(temp_true, temp_pred, suffix = c(".true", ".pred"), by = c("image_id" = "image_id",
                                                                                  "inst_pred" = "inst_pred"))

        scores = rbind(scores, temp)
    }
    
    return(scores)
}

Y_test_label = Y_test
Y_hat_test_label = Y_hat_test

for (i in 1:dim(Y_test)[1]) {
    Y_test_label[i,,,] = bwlabel(Y_test_label[i,,,])
    Y_hat_test_label[i,,,] = pred2label(Y_hat_test_label[i,,,], thresh = 0.5)
}

## for testing with THP1
# for (i in 1:dim(Y_test)[1]) {
#     Y_test_label[i,,,] = pred2label(Y_test_label[i,,,], thresh = 0.5)
# }

## compute iou per instance
iou_instance = iou_metric_instance(Y_test_label, Y_hat_test_label)


# hist(iou_instance$iou_score, breaks = 50)
# boxplot(iou_instance$iou_score)
# median(iou_instance$iou_score, na.rm = T)

lin_reg = ggscatter(iou_instance, x = "s.area.true", y = "s.area.pred",
          add = "reg.line",                                 # Add regression line
          conf.int = TRUE,                                  # Add confidence interval
          add.params = list(color = "blue",
                            fill = "lightgray"),
          xlim = c(0, 8000),
          ylim = c(0, 8000),
          xlab = "Area of ground truth instances",
          ylab = "Area of predicted instances"
          ) +
    stat_cor(method = "pearson", label.x = 6000, label.y = 100)  # Add correlation coefficient

ggsave(filename = paste0(Save_plot, "/IOU_instances_linear_regression_wsGT.png"),
       plot = lin_reg, width = 15, height = 15, units = "cm", dpi = 600)

# img_num = 1
# testim = 
#     abind(
#     abind(
#     normalize(colorLabels(Y_test_label[img_num,,,1])),
#     normalize(colorLabels(Y_hat_test_label[img_num,,,])),
#     along = 1),
#     abind(
#         EBImage::paintObjects(Y_hat_test_label[img_num,,,], Image(UnSharpMasking(normalize(X_test[img_num,,,])), colormode = "color"), col = "red"),
#         EBImage::paintObjects(Y_hat_test_label[img_num,,,], Image(transform_gray_to_rgb_rep(normalize(X_dapi_test[img_num,,])), colormode = "color"), col = "red"),
#         along = 1),
#     along = 2)
# display(testim)
# writeImage(testim, files = paste0("/home/gendarme/Desktop/testim_instances_", CELL,"_wtGT.png"))