library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(glue)
library(EBImage)
options(EBImage.display = "raster")
options(EBImage.bg = "black")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti)
library(jsonlite)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = '/home/gendarme/Documents/UNet'
Unet_dir = paste0(RelPath, "/BF_Data")

# Load custom functions
list_scripts = list("PreprocessingAndTransformation.r", "Load_data_from_disk.r",
                    "Model_FPN.r", "Model_UNET_new.r", "Model_PSPNET.r", "Model_Deeplabv3plus_v3.r",
                    "Backbone_Zoo.r",
                    "Loss.r",
                    "CustomGenerator_CropFix.r",
                    "Inspection.r",
                    "Postprocessing.r")
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

# for(cell in c("A549", "HELA", "MCF7", "RPE1")){
    # cell="HELA"

    # print(cell)
    
    # if(cell == "HELA"){
    #     ds = paste0("Task010_", cell)
    # } else if(cell == "MCF7"){
    #     ds = paste0("Task011_", cell)
    # } else if(cell == "RPE1"){
    #     ds = paste0("Task015_", cell)
    # } else if(cell == "A549"){
    #     ds = paste0("Task001_", cell)
    # }
    cell = "A549"
    ds = paste0("Task001_", cell)
    
    image_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/", ds, "/imagesTs")
    label_folder_test = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/", ds, "/labelsTs")
    
    # Parameters for saving files
    Current_i = as.character(paste0("ImgBF512_1Class"))
    Save_dir = paste0(Unet_dir,"/",cell,"/","Prediction/", Current_i, "--",ifelse(cell == "A549", 100, 2))
    Save_image_semantic = paste0(Save_dir, "/Image_semantic2")
    dir.create(Save_image_semantic, showWarnings = F)
    source(paste0(Save_dir, "/Params_nnUNet_comparaison.r"))

    # generate weight model name
    weight_name = paste0("best_", "model_weights.hdf5")
    
    # build the models
    enc = ENCODER 
    redfct = 8L
    arc = ARCHITECTURE
    dropout = DROPOUT
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
    model1 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 1, "/", weight_name))
    model2 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 2, "/", weight_name))
    model3 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 3, "/", weight_name))
    model4 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 4, "/", weight_name))
    model5 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/unet--seresnext101--Epochs_200--Minibatches_50--FOLD_", 5, "/", weight_name))

    # model1 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/fpn--seresnext101--FOLD_", 1, "/", weight_name))
    # model2 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/fpn--seresnext101--FOLD_", 2, "/", weight_name))
    # model3 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/fpn--seresnext101--FOLD_", 3, "/", weight_name))
    # model4 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/fpn--seresnext101--FOLD_", 4, "/", weight_name))
    # model5 = model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/fpn--seresnext101--FOLD_", 5, "/", weight_name))

    test_input = load_data(image_folder_test, label_folder_test)
    
    X_test <- list2tensor(test_input$X)
    Y_test <- list2tensor(test_input$Y)
    
    X_test = X_test[,1:HEIGHT, 1:WIDTH,]
    if(CHANNELS == 1){dim(X_test) = c(dim(X_test), 1)}
    
    Y_test = Y_test[,1:HEIGHT, 1:WIDTH,]
    if(length(dim(Y_test)) == 3){dim(Y_test) = c(dim(Y_test), 1)}
    
    ## Evaluate the fitted model
    ## Predict and evaluate on training images:
    pred_batch_size = 1
    
    Y_hat_test1 = predict(model1, X_test, batch_size = pred_batch_size)
    Y_hat_test2 = predict(model2, X_test, batch_size = pred_batch_size) 
    Y_hat_test3 = predict(model3, X_test, batch_size = pred_batch_size) 
    Y_hat_test4 = predict(model4, X_test, batch_size = pred_batch_size) 
    Y_hat_test5 = predict(model5, X_test, batch_size = pred_batch_size)
    
    Y_hat_test_class_1 = abind(Y_hat_test1[,,,1], Y_hat_test2[,,,1], Y_hat_test3[,,,1], Y_hat_test4[,,,1], Y_hat_test5[,,,1], along = 4)
    Y_hat_test_class_1 = apply(Y_hat_test_class_1, c(1, 2, 3), mean)
    # str(Y_hat_test_class_1)
    Y_hat_test_class_2 = abind(Y_hat_test1[,,,2], Y_hat_test2[,,,2], Y_hat_test3[,,,2], Y_hat_test4[,,,2], Y_hat_test5[,,,2], along = 4)
    Y_hat_test_class_2 = apply(Y_hat_test_class_2, c(1, 2, 3), mean)
    # str(Y_hat_test_class_2)
    Y_hat_test = abind(Y_hat_test_class_1, Y_hat_test_class_2, along = 4)
    
    for(j in ifelse(ACTIVATION == "softmax", 2, 1):ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS)){
        for (i in 1:5) {
            vir_hat = img_to_viridis(Y_hat_test[i,,,j])
            # display(vir_hat_val)
            
            vir_gt = img_to_viridis(Y_test[i,,,ifelse(is.null(enc), 1, j)])
            # display(vir_gt_test)
            # writeImage(vir_gt_test, files = paste0(Save_image_semantic, "/", "vir_gt_test_class_",J,"_",i,".tif"),
            #            quality = 100, type = "tiff")

            gt_pred_bf_test = abind(combine_col(image_1 = normalize(X_test[i,,,ifelse(is.null(enc), 1, j)]),
                                                color_1 = "grey",
                                                dimension = c(HEIGHT, WIDTH)),
                                    vir_gt,
                                    vir_hat,
                                    along = 1)
            # display(gt_pred_bf_test)
            writeImage(gt_pred_bf_test, files = paste0(Save_image_semantic, "/", "gt_pred_bf_test_class_",j,"_",i,".tif"),
                       quality = 100, type = "tiff")
            
        }
    }

    ### Generate single instances and perform measurements #####################################################
    ## generate plots for IOU and DICE
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/IOU_DICE_plot_2_ensemble.r"))

    ## generate sample images
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images_no_dapi.r"))
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images_no_dapi_2.r"))

# }
