## prepare the data for prediction, metrics and display
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
val_input$image_id
test_input$image_id
Y_hat = predict(model,
                X_train,
                batch_size = pred_batch_size)
Y_hat_val = predict(model,
                    X_val,
                    batch_size = pred_batch_size)
Y_hat_test = predict(model,
                     X_test,
                     batch_size = pred_batch_size)

Val_Img_dir = paste0(Save_loop, "/predict_val")
dir.create(Val_Img_dir, showWarnings = F)
Test_Img_dir = paste0(Save_loop, "/predict_test")
dir.create(Test_Img_dir, showWarnings = F)

for(i in 1:dim(Y_hat_val)[1]){   

    temp_im_name = paste0(Val_Img_dir, "/", CELL, "_", val_input$image_id[i])
    temp_im = Y_hat_val[i,,,]

    writeNIfTI(
        nim = temp_im,
        filename = temp_im_name,
        onefile = TRUE,
        gzipped = TRUE,
        verbose = FALSE,
        warn = -1,
        compression = 1
    )

    print(paste0("val sample #:" , i, " || Sample ID: ", val_input$image_id[i]))

}

for(i in 1:dim(Y_hat_test)[1]){   

    temp_im_name = paste0(Test_Img_dir, "/", CELL, "_", test_input$image_id[i])
    temp_im = Y_hat_test[i,,,]
    
    writeNIfTI(
        nim = temp_im,
        filename = temp_im_name,
        onefile = TRUE,
        gzipped = TRUE,
        verbose = FALSE,
        warn = -1,
        compression = 1
    )

    print(paste0("test sample #:" , i, " || Sample ID: ", test_input$image_id[i]))
}
