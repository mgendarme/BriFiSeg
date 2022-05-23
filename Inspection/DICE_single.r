source(paste0(RelPath, "/Scripts/FunctionCompilation/connected_component_analysis.r"))

instance_gen = function(hat){
  temp = apply(hat, 1, convert_to_instance_seg, simplify = "array")
  temp = simplify2array(temp)
  temp = aperm(temp, c(3, 1, 2))
  temp = add_dim(temp, dim3 = 1) 
  return(temp)
}

## gen instances for gt and pred
# gt
# Y_label = instance_gen(Y_train)
Y_val_label = instance_gen(Y_val)
Y_test_label = instance_gen(Y_test)

# pred
# Y_hat_label = instance_gen(Y_hat)
Y_hat_val_label = instance_gen(Y_hat_val)
Y_hat_test_label = instance_gen(Y_hat_test)

dice_metric_instance <- function(y_true, y_pred){
    
    ## for testing
    #  y_true = Y_val_label
    #  y_pred = Y_hat_val_label
    
    if(!is.list(y_true)){
        num_imgs <- dim(y_true)[1]
    }
    
    scores <- tibble()
    
    # registerDoParallel(detectCores() - 1)
    
    # tf$config$set_visible_devices(list(), 'GPU')

    # scores_inst = foreach(i = 1:num_imgs, combine = rbind) %dopar% {
    for(i in 1:num_imgs){
        
        scores_temp = tibble()
        
        y_true_i = array(y_true[i,,,], dim = dim(y_true)[2]*dim(y_true)[3])
        y_pred_i = array(y_pred[i,,,], dim = dim(y_pred)[2]*dim(y_pred)[3])
        
        scores_instances_i = array(0, max(y_true_i))
        inst_pred_order = array(0, max(y_true_i))
        inst_true_order = array(0, max(y_true_i))
        
        for(j in 1:max(y_true_i)) {
            
            start_time = Sys.time()

            scores_instances_j = array(0, dim = c(max(y_true_i), max(y_pred_i)))
            
            for(k in 1:max(y_pred_i)) {
                scores_instances_j[j,k] = dice_coef(ifelse(y_true[i,,,] == j, 1, 0),
                                                    ifelse(y_pred[i,,,] == k, 1, 0)) %>% as.numeric()
            }
                        
            id_all = ifelse(i == 1, j, sum(apply(y_true[1:(i-1),,,1], 1, max)) + j)
            if(sum(y_true[i,,,] == j) != 0 & max(scores_instances_j) == 0){
                inst_true_order[j] = j
                inst_pred_order[j] = 0
                scores_instances_i[j] = 0
            } else {
                inst_true_order[j] = j
                inst_pred_order[j] = which(scores_instances_j[j,] == max(scores_instances_j[j,]))
                scores_instances_i[j] = max(scores_instances_j[j,])
            }
            
        }

        temp_true = computeFeatures.shape(y_true[i,,,]) %>% 
            as_tibble() %>% 
            rownames_to_column(var = "inst_true") %>% 
            mutate(image_id = i, 
                   dice_score = scores_instances_i,
                   inst_pred = inst_pred_order)
        temp_pred = computeFeatures.shape(y_pred[i,,,]) %>% 
            as_tibble() %>% 
            rownames_to_column(var = "inst_pred") %>% 
            mutate(image_id = i,
                   inst_pred = as.double(inst_pred))

        temp = full_join(temp_true, temp_pred, suffix = c(".true", ".pred"), by = c("image_id" = "image_id",
                                                                                  "inst_pred" = "inst_pred"))

        scores = rbind(scores, temp)

        end_time = Sys.time()
        run_time_dopar = end_time - start_time

        print(paste0("image # ", ifelse(str_length(i) == 1 , paste0("0",i), i), " out of ", num_imgs, " images analysed",
                     " || # labels: ",  ifelse(str_length(max(y_true_i)) == 1 , paste0("0",max(y_true_i)), max(y_true_i)),
                     " || compute time: ", round(as.numeric(run_time_dopar), digits = 2) * 60, " s"))

        # return(temp)
    }

    return(scores)
}

# single_dice_train = dice_metric_instance(Y_label, Y_hat_label)

print(paste0("---------- Dice instances validation ----------"))
single_dice_val = dice_metric_instance(Y_val_label, Y_hat_val_label)
print(paste0("---------- Dice instances test ----------"))
single_dice_test = dice_metric_instance(Y_test_label, Y_hat_test_label)

## add # of objects per image
for(i in c("val", "test")){
    temp_gt = get(paste0("Y_", i, "_label"))
    temp_pred = get(paste0("Y_hat_", i, "_label"))
    cell_count = array(0, c(nrow(temp_gt), 2))
    for(j in 1:nrow(temp_gt)){
        cell_count[j,1] = length(unique(c(temp_gt[j,,,])))
        cell_count[j,2] = length(unique(c(temp_pred[j,,,])))
    }
    assign(paste0("cell_count_", i), tibble(count_gt = cell_count[,1],
                                            count_pred = cell_count[,2]))
}

for(i in c("val", "test")){
    cc_lin_reg = ggpubr::ggscatter(get(paste0("cell_count_", i)),
            x = "count_gt", y = "count_pred",
            add = "reg.line",                                 # Add regression line
            conf.int = TRUE,                                  # Add confidence interval
            add.params = list(color = "blue",
                                fill = "lightgray"),
            # xlim = c(0, 8000),
            # ylim = c(0, 8000),
            xlab = "Cell count ground truth",
            ylab = "Cell count prediction"
            ) +
        ggpubr::stat_cor(method = "pearson", label.x = 20, label.y = 5, na.rm = TRUE)  # Add correlation coefficient

    ggsave(filename = paste0(Save_plot_instance, "/Cell_count_linear_regression_", i, ".png"),
        plot = cc_lin_reg, width = 15, height = 15, units = "cm", dpi = 600)
}


# train_dice = format(round(median(single_dice_train$dice_score, na.rm = T), 3), nsmall = 3)
# png(paste0(Save_plot_instance, "/DICE_1_instance_boxplot_train.png"),
#         width = 800, height = 1200, res = 300)
# boxplot(single_dice_train$dice_score, 
#         main = c(paste0("F1 score \nmedian = ", train_dice)),
#         xlab = "Train",
#         ylab = "F1 score per image",
#         ylim = c(0,1)
# )
# dev.off()

val_dice = format(round(median(single_dice_val$dice_score, na.rm = T), 4), nsmall = 3)
png(paste0(Save_plot_instance, "/DICE_2_instance_boxplot_val.png"),
        width = 800, height = 1200, res = 300)
boxplot(single_dice_val$dice_score, 
        main = c(paste0("F1 score \nmedian = ", val_dice)),
        xlab = "Validation",
        ylab = "F1 score per image",
        ylim = c(0,1)
)
dev.off()

test_dice = format(round(median(single_dice_test$dice_score, na.rm = T), 4), nsmall = 3)
png(paste0(Save_plot_instance, "/DICE_3_instance_boxplot_test.png"),
        width = 800, height = 1200, res = 300)
boxplot(single_dice_test$dice_score, 
        main = c(paste0("F1 score \nmedian = ", test_dice)),
        xlab = "Test",
        ylab = "F1 score per image",
        ylim = c(0,1)
)
dev.off()

# lmtest = lm(`s.area.true` ~ `s.area.pred`, data = single_dice_test)# %>% filter(!is.na(s.a)))
# test_summary = summary(lmtest)
# test_summary$r.squared
# single_dice_test %>% select(s.area.true, s.area.pred) %>% filter(s.area.pred)
# single_dice_test$s.area.pred

for(i in c("val", "test")){
    lin_reg = ggpubr::ggscatter(get(paste0("single_dice_", i)),
            x = "s.area.true", y = "s.area.pred",
            add = "reg.line",                                 # Add regression line
            conf.int = TRUE,                                  # Add confidence interval
            add.params = list(color = "blue",
                                fill = "lightgray"),
            xlim = c(0, 8000),
            ylim = c(0, 8000),
            xlab = "Area of ground truth instances",
            ylab = "Area of predicted instances"
            ) +
        ggpubr::stat_cor(method = "pearson", label.x = 5000, label.y = 100, na.rm = TRUE)  # Add correlation coefficient

    ggsave(filename = paste0(Save_plot_instance, "/DICE_instances_linear_regression_GT_", i, ".png"),
        plot = lin_reg, width = 15, height = 15, units = "cm", dpi = 600)
}
# img_num = 1

for(img_num in 1:5){
    ## Val
    instance_gt_pred_val = abind(
        normalize(colorLabels(Y_val_label[img_num,,,1])),
        normalize(colorLabels(Y_hat_val_label[img_num,,,])),
        along = 1
    )
    writeImage(instance_gt_pred_val, files = paste0(Save_image_instance, "/instance_gt_pred_val_", img_num, ".png"))
    
    mask_gt_pred_val = abind(
        EBImage::rgbImage(red = Y_val[img_num,,,2], green = Y_val[img_num,,,3]),
        EBImage::rgbImage(red = Y_hat_val[img_num,,,2] > 0.5, green = Y_hat_val[img_num,,,3] > 0.5),
        along = 1
    )
    writeImage(mask_gt_pred_val, files = paste0(Save_image_instance, "/mask_gt_pred_val_", img_num, ".png"))

    ## Test
    
    instance_gt_pred_test = abind(
        normalize(colorLabels(Y_test_label[img_num,,,1])),
        normalize(colorLabels(Y_hat_test_label[img_num,,,])),
        along = 1
    )
    writeImage(instance_gt_pred_test, files = paste0(Save_image_instance, "/instance_gt_pred_test_", img_num, ".png"))
    
    mask_gt_pred_test = abind(
        EBImage::rgbImage(red = Y_test[img_num,,,2], green = Y_test[img_num,,,3]),
        EBImage::rgbImage(red = Y_hat_test[img_num,,,2] > 0.5, green = Y_hat_test[img_num,,,3] > 0.5),
        along = 1
    )
    writeImage(mask_gt_pred_test, files = paste0(Save_image_instance, "/mask_gt_pred_test_", img_num, ".png"))
}
