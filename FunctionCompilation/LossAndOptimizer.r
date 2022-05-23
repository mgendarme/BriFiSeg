### Loss function

## DICE COEFICIENT
# Defining dice coefficient and its negative as loss function using keras-backend functions (k_*).
dice_coef <- function(y_true, y_pred, smooth = 1) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  # (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth) 
  k_mean((2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth))
}

dice_coef_loss_for_bce <- function(y_true, y_pred){
  1 - dice_coef(y_true, y_pred)
}
attr(dice_coef_loss_for_bce, "py_function_name") <- "dice_coef_loss_for_bce"

dice_coef_loss_bce <- function(y_true, y_pred, dice = 0.5, bce = 0.5){
  k_binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss_for_bce(y_true, y_pred) * dice
}
attr(dice_coef_loss_bce, "py_function_name") <- "dice_coef_loss_bce"

dice_coef_loss_bce_2Classes <- function(y_true, y_pred, l_b_c = L_B_C, w_class_1 = W_CLASS_1, w_class_2 = W_CLASS_2){
  k_categorical_crossentropy(y_true, y_pred) * l_b_c +
  # k_mean(k_categorical_crossentropy(y_true, y_pred)) * l_b_c + # WITH OR WITHOUT k_mean()
    dice_coef_loss_for_bce(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
    dice_coef_loss_for_bce(y_true[,,,2], y_pred[,,,2]) * w_class_2
}
attr(dice_coef_loss_bce_2Classes, "py_function_name") <- "dice_coef_loss_bce_2Classes"

dice_coef_loss_bce_3Classes <- function(y_true, y_pred,
                                        l_b_c = L_B_C,
                                        w_class_1 = W_CLASS_1,
                                        w_class_2 = W_CLASS_2,
                                        w_class_3 = W_CLASS_3){
    
  k_mean(k_binary_crossentropy(y_true, y_pred)) * l_b_c +
    dice_coef_loss_for_bce(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
    dice_coef_loss_for_bce(y_true[,,,2], y_pred[,,,2]) * w_class_2 +
    dice_coef_loss_for_bce(y_true[,,,3], y_pred[,,,3]) * w_class_3

}
attr(dice_coef_loss_bce_3Classes, "py_function_name") <- "dice_coef_loss_bce_3Classes"

make_loss = function(loss_name){
    if(loss_name == ""){

    } else if(loss_name == "")
}
