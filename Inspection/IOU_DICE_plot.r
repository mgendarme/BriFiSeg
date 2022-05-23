
## prepare the data for metrics and display
X_train <- list2tensor(train_input$X)
Y_train <- list2tensor(train_input$Y)
X_val <- list2tensor(val_input$X)
Y_val <- list2tensor(val_input$Y)
X_test <- list2tensor(test_input$X)
Y_test <- list2tensor(test_input$Y)

X_train = X_train[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_train) = c(dim(X_train), 1)}
X_val = X_val[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_val) = c(dim(X_val), 1)}
X_test = X_test[,1:HEIGHT, 1:WIDTH,]
if(CHANNELS == 1){dim(X_test) = c(dim(X_test), 1)}

Y_train = Y_train[,1:HEIGHT, 1:WIDTH,]
if(length(dim(Y_train)) == 3){dim(Y_train) = c(dim(Y_train), 1)}
Y_val = Y_val[,1:HEIGHT, 1:WIDTH,]
if(length(dim(Y_val)) == 3){dim(Y_val) = c(dim(Y_val), 1)}
Y_test = Y_test[,1:HEIGHT, 1:WIDTH,]
if(length(dim(Y_test)) == 3){dim(Y_test) = c(dim(Y_test), 1)}
    
## Evaluate the fitted model
## Predict and evaluate on training images:
pred_batch_size = 1

Y_hat = predict(model,
                X_train,
                batch_size = pred_batch_size)
Y_hat_val = predict(model,
                    X_val,
                    batch_size = pred_batch_size)
Y_hat_test = predict(model,
                     X_test,
                     batch_size = pred_batch_size)

#######################################
###   Predictions post-processing   ###
###   e.g. keep only probs > 0.5    ###
#######################################

# if activation is softmax and class 1 keep only foreground 
if(ACTIVATION == "softmax"){
  Y_train = Y_train[,,,2:(dim(Y_train)[4])]
  if(length(dim(Y_train)) == 3){dim(Y_train) = c(dim(Y_train), 1)}
  Y_val = Y_val[,,,2:(dim(Y_val)[4])]
  if(length(dim(Y_val)) == 3){dim(Y_val) = c(dim(Y_val), 1)}
  Y_test = Y_test[,,,2:(dim(Y_test)[4])]
  if(length(dim(Y_test)) == 3){dim(Y_test) = c(dim(Y_test), 1)}
  
  Y_hat = Y_hat[,,,2:(dim(Y_hat)[4])]
  if(length(dim(Y_hat)) == 3){dim(Y_hat) = c(dim(Y_hat), 1)}
  Y_hat_val = Y_hat_val[,,,2:(dim(Y_hat_val)[4])]
  if(length(dim(Y_hat_val)) == 3){dim(Y_hat_val) = c(dim(Y_hat_val), 1)}
  Y_hat_test = Y_hat_test[,,,2:(dim(Y_hat_test)[4])]
  if(length(dim(Y_hat_test)) == 3){dim(Y_hat_test) = c(dim(Y_hat_test), 1)}
}

if(ACTIVATION == "softmax" & CLASS == 1){
  Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0)
  # Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)
  Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0)
  # Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)
  Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0)
  # Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)
} else if(ACTIVATION != "softmax" & CLASS == 1){
  Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0)
  # Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)
  Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0)
  # Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)
  Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0)
  # Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)
} else {
  Y_hat_train_ab_p5 = ifelse(Y_hat >= TRESH_PRED, Y_hat, 0)
  # Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)
  Y_hat_val_ab_p5 = ifelse(Y_hat_val >= TRESH_PRED, Y_hat_val, 0)
  # Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)
  Y_hat_test_ab_p5 = ifelse(Y_hat_test >= TRESH_PRED, Y_hat_test, 0)
  # Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)
}

# } else if(CLASS == 2){
#   Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0) -
#     ifelse(Y_hat[,,,2] >= TRESH_PRED, Y_hat[,,,2], 0)
#   Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)

#   Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0) -
#     ifelse(Y_hat_val[,,,2] >= TRESH_PRED, Y_hat_val[,,,2], 0)
#   Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)

#   Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0) -
#     ifelse(Y_hat_test[,,,2] >= TRESH_PRED, Y_hat_test[,,,2], 0)
#   Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)
# } else if(CLASS == 3){
#   Y_hat_train_ab_p5 = ifelse(Y_hat[,,,1] >= TRESH_PRED, Y_hat[,,,1], 0) -
#     ifelse(Y_hat[,,,2] >= TRESH_PRED, Y_hat[,,,2], 0) -
#     ifelse(Y_hat[,,,3] >= TRESH_PRED, Y_hat[,,,3], 0)
#   Y_hat_train_ab_p5 = ifelse(Y_hat_train_ab_p5 < 0, 0, Y_hat_train_ab_p5)

#   Y_hat_val_ab_p5 = ifelse(Y_hat_val[,,,1] >= TRESH_PRED, Y_hat_val[,,,1], 0) -
#     ifelse(Y_hat_val[,,,2] >= TRESH_PRED, Y_hat_val[,,,2], 0) -
#     ifelse(Y_hat_val[,,,3] >= TRESH_PRED, Y_hat_val[,,,3], 0)
#   Y_hat_val_ab_p5 = ifelse(Y_hat_val_ab_p5 < 0, 0, Y_hat_val_ab_p5)

#   Y_hat_test_ab_p5 = ifelse(Y_hat_test[,,,1] >= TRESH_PRED, Y_hat_test[,,,1], 0) -
#     ifelse(Y_hat_test[,,,2] >= TRESH_PRED, Y_hat_test[,,,2], 0) -
#     ifelse(Y_hat_test[,,,3] >= TRESH_PRED, Y_hat_test[,,,3], 0)
#   Y_hat_test_ab_p5 = ifelse(Y_hat_test_ab_p5 < 0, 0, Y_hat_test_ab_p5)
# }

###############
###   IOU   ###
###############

iou_thresh <- c(seq(0.01, 0.99, 0.01))

iou <- function(y_true, y_pred){
   # for testing
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
str(Y_train[,,,1])
str(Y_hat_train_ab_p5)
iou_train_post_proc = iou_metric(Y_train[,,,1], Y_hat_train_ab_p5)
iou_val_post_proc   = iou_metric(Y_val[,,,1], Y_hat_val_ab_p5)
iou_test_post_proc  = iou_metric(Y_test[,,,1], Y_hat_test_ab_p5)

history_iou = list(iou_train_post_proc = iou_train_post_proc,
                   iou_val_post_proc = iou_val_post_proc,
                   iou_test_post_proc = iou_test_post_proc)
                
save(history_iou, file = paste0(Save_plot, "/HTR_IOU_", Current_i, "_", loop_id, ".rdata"))

#######################
####   Boxplots    ####
#######################
## train
train_iou = format(round(median(iou_train_post_proc, na.rm = T), 3), nsmall = 3)
png(paste0(Save_plot, "/IOU_1_Boxplot_train_postproc_", loop_id, ".png"), width = 800, height = 1200, res = 300)
boxplot(iou_train_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", train_iou)),
        xlab = "Train",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

## val
val_iou = format(round(median(iou_val_post_proc, na.rm = T), 3), nsmall = 3)
png(paste0(Save_plot, "/IOU_2_Boxplot_val_postproc_", loop_id,".png"),
    width = 800, height = 1200, res = 300)
boxplot(iou_val_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", val_iou)),
        xlab = "Validation",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

## test
test_iou =  format(round(median(iou_test_post_proc, na.rm = T), 3), nsmall = 3)
png(paste0(Save_plot, "/IOU_3_Boxplot_test_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(iou_test_post_proc, 
        main = c(paste0("Jaccard index \nmedian = ", test_iou)),
        xlab = "Test",
        ylab = "Intersection over union per image",
        ylim = c(0, 1))
dev.off()

##############
#### DICE ####
##############

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
                
save(history_dice, file = paste0(Save_plot, "/HTR_DICE_", Current_i, "_", loop_id, ".rdata"))

#######################
####   Boxplots    ####
#######################

## train
train_dice = format(round(median(dice_train_post_proc, na.rm = T), 3), nsmall = 3)
png(paste0(Save_plot, "/DICE_1_Boxplot_train_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(dice_train_post_proc, 
        main = c(paste0("F1 score \nmedian = ", train_dice)),
        xlab = "Train",
        ylab = "F1 score per image",
        ylim = c(0, 1))
dev.off()

## val
val_dice = format(round(median(dice_val_post_proc, na.rm = T), 3), nsmall = 3)
png(paste0(Save_plot, "/DICE_2_Boxplot_val_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(dice_val_post_proc, 
        main = c(paste0("F1 score \nmedian = ", val_dice)),
        xlab = "Validation",
        ylab = "F1 score per image",
        ylim = c(0, 1))
dev.off()

## test
test_dice = format(round(median(dice_test_post_proc, na.rm = T), 3), nsmall = 3)
png(paste0(Save_plot, "/DICE_3_Boxplot_test_postproc_", loop_id,".png"), width = 800, height = 1200, res = 300)
boxplot(dice_test_post_proc, 
        main = c(paste0("F1 score \nmedian = ", test_dice)),
        xlab = "Test",
        ylab = "F1 score per image",
        ylim = c(0, 1))
dev.off()
