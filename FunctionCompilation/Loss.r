### Loss function

## DICE COEFICIENT
# Defining dice coefficient and its negative as loss function using keras-backend functions (k_*).
dice_coef <- function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  k_mean((2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)) # for use in combination with bce
}

dice <- function(y_true, y_pred){
  1 - dice_coef(y_true, y_pred)
}
attr(dice, "py_function_name") <- "dice"

dice_target <- function(y_true, y_pred, channel = 2){
  1 - dice_coef(y_true[,,,channel:(CLASS+1)], y_pred[,,,channel:(CLASS+1)])
}
attr(dice, "py_function_name") <- "dice_target"

dice_coef_loss_bce_1_class <- function(y_true, y_pred, dice_f = 0.5, bce_f = 0.5){
  k_binary_crossentropy(y_true, y_pred) * bce_f + dice(y_true, y_pred) * dice_f
}
attr(dice_coef_loss_bce_1_class, "py_function_name") <- "dice_coef_loss_bce_1_class"

dice_coef_loss_bce_1_class_softmax <- function(y_true, y_pred, class = CLASS,  dice_f = 0.5, bce_f = 0.5){
  if(class == 1){
    k_binary_crossentropy(y_true[,,,2], y_pred[,,,2]) * bce_f + dice(y_true[,,,2], y_pred[,,,2]) * dice_f
  } else {
    k_categorical_crossentropy(y_true, y_pred) * bce_f + dice(y_true, y_pred) * dice_f
  }
  
}
attr(dice_coef_loss_bce_1_class, "py_function_name") <- "dice_coef_loss_bce_1_class_softmax"

dice_coef_loss_bce_2_class <- function(y_true, y_pred, l_b_c = L_B_C, w_class_1 = W_CLASS_1, w_class_2 = W_CLASS_2){
  k_categorical_crossentropy(y_true, y_pred) * l_b_c +
    dice(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
    dice(y_true[,,,2], y_pred[,,,2]) * w_class_2
}
attr(dice_coef_loss_bce_2_class, "py_function_name") <- "dice_coef_loss_bce_2_class"

dice_coef_loss_bce_3_class <- function(y_true, y_pred,
                                        l_b_c = L_B_C,
                                        w_class_1 = W_CLASS_1,
                                        w_class_2 = W_CLASS_2,
                                        w_class_3 = W_CLASS_3){
  
  k_mean(k_binary_crossentropy(y_true, y_pred)) * l_b_c + ### try binary CE?
  # loss_categorical_crossentropy(y_true, y_pred) * l_b_c +
  # k_categorical_crossentropy(y_true, y_pred) * l_b_c +
  # k_mean(k_categorical_crossentropy(y_true, y_pred)) * l_b_c + # WITH OR WITHOUT k_mean()
    dice(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
    dice(y_true[,,,2], y_pred[,,,2]) * w_class_2 +
    dice(y_true[,,,3], y_pred[,,,3]) * w_class_3
}
attr(dice_coef_loss_bce_3_class, "py_function_name") <- "dice_coef_loss_bce_3_class"

# Custom Focal Loss
focalLoss <- function(y_true, y_pred, gamma=2., alpha=.25){
  
  ## for testing
  y_true = input$Y[[1]]
  y_pred = input$Y[[1]]
  
  #loss <- function(y_true,y_pred){
  pt_1 <- tf$where(tf$equal(y_true,1), y_pred,tf$ones_like(y_pred))
  pt_0 <- tf$where(tf$equal(y_true,0), y_pred,tf$ones_like(y_pred))
  
  #clip to prevent NaNs and Infs
  epsilon <- K$epsilon()
  
  pt_1 <- K$clip(pt_1,epsilon,1.-epsilon)
  pt_0 <- K$clip(pt_0,epsilon,1.-epsilon)
  
  return(
    -K$mean(alpha*K$pow(1.-pt_1,gamma)*K$log(pt_1))-K$mean((1-alpha)*K$pow(pt_0,gamma)*K$log(1.-pt_0))
    )

}

## source
## https://github.com/ANTsX/ANTsRNet/blob/master/R/customMetrics.R
# library(reticulate)
categorical_focal_loss = function(y_true, y_pred, gamma = 2.0, alpha = 0.25){

  categorical_focal_loss_fixed = function(y_true, y_pred){

    ## for testing
    # gamma = 2.0
    # alpha = 0.25

    K = keras::backend()

    y_pred = y_pred / K$sum(y_pred, axis = -1L, keepdims = TRUE)
    y_pred = K$clip(y_pred, K$epsilon(), 1.0 - K$epsilon())
    cross_entropy = k_binary_crossentropy(y_true, y_pred) 
    # cross_entropy = y_true * K$log(y_pred)
    gain = alpha * K$pow(1.0 - y_pred, gamma) * cross_entropy
    return(-K$sum(gain, axis = -1L))
    }

  return(categorical_focal_loss_fixed)

}

##############################################################################################################
########                                  Loss factory                                                  ######
##############################################################################################################

make_loss = function(loss_name){
                        if(loss_name == "dice_ce_3_class"){
                            return(
                                function(y_true, y_pred,
                                        l_b_c = .4,
                                        w_class_1 = .2,
                                        w_class_2 = .2,
                                        w_class_3 = .2){
                                            k_mean(k_binary_crossentropy(y_true, y_pred)) * l_b_c +
                                                dice(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
                                                dice(y_true[,,,2], y_pred[,,,2]) * w_class_2 +
                                                dice(y_true[,,,3], y_pred[,,,3]) * w_class_3
                            })
                        } else if(loss_name == "dice_ce_2_class") {
                            return(
                                function(y_true, y_pred,
                                        l_b_c = .6,
                                        w_class_1 = .2, 
                                        w_class_2 = .2){
                                    k_mean(k_binary_crossentropy(y_true, y_pred)) * l_b_c +
                                        dice(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
                                        dice(y_true[,,,2], y_pred[,,,2]) * w_class_2
                            })
                        } else if(loss_name == "dice_ce_1_class"){
                            return(
                                function(y_true, y_pred,
                                    dice_f = 0.5, bce_f = 0.5){
                                    k_binary_crossentropy(y_true, y_pred) * bce_f +
                                        dice(y_true, y_pred) * dice_f
                            })
                        } else if(loss_name == "dice_focal_1_class"){
                            return(
                                function(y_true, y_pred, dice = 0.5, focal = 0.5){
                                    categorical_focal_loss(y_true, y_pred, gamma=2., alpha=.25) * focal +
                                        dice(y_true, y_pred) * dice
                            })
                        } else if(loss_name == "dice_ce_1_class_softmax"){
                            return(
                                function(y_true, y_pred, dice_f = 0.5, ce_f = 0.5){
                                    k_categorical_crossentropy(y_true, y_pred) * ce_f +
                                        dice(y_true[,,,2], y_pred[,,,2]) * dice_f
                            })  
                        } else if(loss_name == "dice_ce_2_class_softmax"){
                             return(
                                function(y_true, y_pred, dice_f = 0.5, ce_f = 0.5){
                                    k_categorical_crossentropy(y_true, y_pred) * ce_f +
                                        dice(y_true[,,,2:3], y_pred[,,,2:3]) * dice_f
                            })  
                        }
                    }
