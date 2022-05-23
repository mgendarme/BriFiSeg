## training set
X_train <- list2tensor(train_input$X)
Y_train <- list2tensor(train_input$Y)
## dev set
X_val <- list2tensor(val_input$X)
Y_val <- list2tensor(val_input$Y)
## test set
X_test <- list2tensor(test_input$X)
Y_test <- list2tensor(test_input$Y)

# X = X[,1:HEIGHT, 1:WIDTH,]
# if(CHANNELS == 1){dim(X) = c(dim(X), 1)}
X_train = X_train[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_train) = c(dim(X_train), 1)}
X_val = X_val[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_val) = c(dim(X_val), 1)}
X_test = X_test[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_test) = c(dim(X_test), 1)}

Y_train = Y_train[,1:HEIGHT, 1:WIDTH,]
Y_val = Y_val[,1:HEIGHT, 1:WIDTH,]
Y_test = Y_test[,1:HEIGHT, 1:WIDTH,]

# for (i in c("Y_train", "Y_val", "Y_test")) {
#   print(paste0("Str: ", i, " ", as.character(str(get(i)))))
# }

## Evaluate the fitted model
## Predict and evaluate on training images:
pred_batch_size = 8
# Y_hat <- predict(model, x = X_train)
Y_hat = predict(model, 
                X_train,
                batch_size = pred_batch_size)
# Y_hat_val <- predict(model, x = X_val)
Y_hat_val = predict(model, 
                    X_val,
                    batch_size = pred_batch_size)
# Y_hat_test <- predict(model, x = X_test)
Y_hat_test = predict(model, 
                     X_test,
                     batch_size = pred_batch_size)

##############
### IOU #2 ###
##############

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

iou_metric <- function(y_true, y_pred){
  if(!is.list(y_true)){
    num_imgs <- dim(y_true)[1]
  }
  
  scores <- array(0, num_imgs)
    
  for(i in 1:num_imgs){
    
    y_true_i = array(y_true[i,,], dim = dim(y_true)[2]*dim(y_true)[3])
    y_pred_i = array(y_pred[i,,], dim = dim(y_pred)[2]*dim(y_pred)[3])
    
    if(sum(y_true[i,,]) == 0 & sum(y_pred[i,,]) == 0){
      scores[i] = 1
    } else {
      scores[i] = mean(iou_thresh <= iou(y_true[i,,], y_pred[i,,]))
    }
  }
  
  return(scores)
}

TRESH_PRED = 0.5

if(CLASS == 1){

  Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0)
  
  Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0)
  
  Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0)
  
} else if(CLASS == 2){

  Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0) -
    ifelse(Y_hat[,,,2] >= TRESH_PRED, Y_hat[,,,2], 0)
  Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)

  Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0) -
    ifelse(Y_hat_val[,,,2] >= TRESH_PRED, Y_hat_val[,,,2], 0)
  Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)

  Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0) -
    ifelse(Y_hat_test[,,,2] >= TRESH_PRED, Y_hat_test[,,,2], 0)
  Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)

} else if(CLASS == 3){

  Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0) -
    ifelse(Y_hat[,,,2] >= TRESH_PRED, Y_hat[,,,2], 0) -
    ifelse(Y_hat[,,,3] >= TRESH_PRED, Y_hat[,,,3], 0)
  Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)

  Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0) -
    ifelse(Y_hat_val[,,,2] >= TRESH_PRED, Y_hat_val[,,,2], 0) -
    ifelse(Y_hat_val[,,,3] >= TRESH_PRED, Y_hat_val[,,,3], 0)
  Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)

  Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0) -
    ifelse(Y_hat_test[,,,2] >= TRESH_PRED, Y_hat_test[,,,2], 0) -
    ifelse(Y_hat_test[,,,3] >= TRESH_PRED, Y_hat_test[,,,3], 0)
  Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)

}

iou_train_post_proc = iou_metric(Y_train[,,,1], Y_hat_train_ab_p5)
iou_val_post_proc   = iou_metric(Y_val[,,,1], Y_hat_val_ab_p5)
iou_test_post_proc  = iou_metric(Y_test[,,,1], Y_hat_test_ab_p5)

history_iou = list(iou_train_post_proc = iou_train_post_proc,
                   iou_val_post_proc = iou_val_post_proc,
                   iou_test_post_proc = iou_test_post_proc)
                
save(history_iou, file = paste0(Save_dir, "/HTR_IOU_", Current_i, "_", loop_id, ".rdata"))

#######################
####   Boxplots    ####
#######################
## train
png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_1_Boxplot_train_postproc_", loop_id, ".png"), width = 800, height = 1200, res = 300)
boxplot(iou_train_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", round(median(iou_train_post_proc, na.rm = T), 2))),
        xlab = "Train",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

## val
png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_2_Boxplot_val_postproc_", loop_id,".png"),
    width = 800, height = 1200, res = 300)
boxplot(iou_val_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ",
                 round(median(iou_val_post_proc, na.rm = T), 2))),
        xlab = "Validation",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

## test
png(paste0(str_replace(Save_dir, "/Image", ""), "/IOU_3_Boxplot_test_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(iou_test_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", round(median(iou_test_post_proc, na.rm = T), 2))),
        xlab = "Test",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

dice_metric <- function(y_true, y_pred){
  if(!is.list(y_true)){
    num_imgs <- dim(y_true)[1]
  }
  
  scores <- array(0, num_imgs)
    
  for(i in 1:num_imgs){
    
    scores[i] = dice_coef(y_true[i,,], y_pred[i,,]) %>% as.numeric()
    
  }
  
  return(scores)
}

dice_train_post_proc = dice_metric(Y_train[,,,1], Y_hat_train_ab_p5)
dice_val_post_proc = dice_metric(Y_val[,,,1], Y_hat_val_ab_p5)
dice_test_post_proc = dice_metric(Y_test[,,,1], Y_hat_test_ab_p5)

history_dice = list(dice_train_post_proc = dice_train_post_proc,
                   dice_val_post_proc = dice_val_post_proc,
                   dice_test_post_proc = dice_test_post_proc)
                
save(history_dice, file = paste0(Save_dir, "/HTR_DICE_", Current_i, "_", loop_id, ".rdata"))

## train
png(paste0(str_replace(Save_dir, "/Image", ""), "/DICE_1_Boxplot_train_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(dice_train_post_proc, 
        main = c(paste0("F1 score \nmedian = ", round(median(dice_train_post_proc, na.rm = T), 2))),
        xlab = "Train",
        ylab = "F1 score per image",
        ylim = c(0, 1))
dev.off()

## val
png(paste0(str_replace(Save_dir, "/Image", ""), "/DICE_2_Boxplot_val_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(dice_val_post_proc, 
        main = c(paste0("F1 score \nmedian = ", round(median(dice_val_post_proc, na.rm = T), 2))),
        xlab = "Validation",
        ylab = "F1 score per image",
        ylim = c(0, 1))
dev.off()

## test
png(paste0(str_replace(Save_dir, "/Image", ""), "/DICE_3_Boxplot_test_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(dice_test_post_proc, 
        main = c(paste0("F1 score \nmedian = ", round(median(dice_test_post_proc, na.rm = T), 2))),
        xlab = "Test",
        ylab = "F1 score per image",
        ylim = c(0, 1))
dev.off()
