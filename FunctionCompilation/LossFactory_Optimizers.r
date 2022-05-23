# loss factory

## to optimize for easyier se;ection
if(OPTIMIZER == "ADAM"){
  LossFactory <- function(CLASS){
    if(CLASS == 1){
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = "adam", # used instead of the classical stochastic gradient descent procedure
          # optimizer = optimizer_adam(lr = LR, decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } else if(CLASS == 2) {
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce_2Classes, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = "adam", # used instead of the classical stochastic gradient descent procedure
          # optimizer = optimizer_adam(lr = LR, decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } else if(CLASS == 3){
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce_3Classes, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          # optimizer = "adam", # used instead of the classical stochastic gradient descent procedure
          optimizer = optimizer_adam(lr = LR, decay = DC),#, clipnorm = 1.0), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } 
  }
} else if(OPTIMIZER == "NADAM"){
  LossFactory <- function(CLASS){
    if(CLASS == 1){
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = optimizer_nadam(lr = LR), # schedule_decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } else if(CLASS == 2) {
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce_2Classes, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = optimizer_nadam(lr = LR), # schedule_decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } else if(CLASS == 3){
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce_3Classes, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = optimizer_nadam(lr = LR), # schedule_decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } 
  }
} else if(OPTIMIZER == "RMSPROP"){
  LossFactory <- function(CLASS){
    if(CLASS == 1){
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = optimizer_rmsprop(lr = LR, decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } else if(CLASS == 2) {
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce_2Classes, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = optimizer_rmsprop(lr = LR, decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } else if(CLASS == 3){
      model <- model %>%
        compile(
          loss = dice_coef_loss_bce_3Classes, #dice_coef_loss_bce,  #dice_coef_loss_bce_multiClasses, #k_categorical_crossentropy,
          optimizer = optimizer_rmsprop(lr = LR, decay = DC), # used instead of the classical stochastic gradient descent procedure
          metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)#, # 'accuracy'
        )
    } 
  }
}
