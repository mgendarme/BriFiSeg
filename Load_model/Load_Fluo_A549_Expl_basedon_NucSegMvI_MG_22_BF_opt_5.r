library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(EBImage)
options(EBImage.display = "raster")
library(rstudioapi)

# set seed for reproducibility
set.seed(11)

# set to run tensorflow on cpu if GPU is busy
tf$config$set_visible_devices(list(), 'GPU')

## Settings
# general parameters
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
# Load hyper parameters

# Load custom functions
list_scripts = list("Params.r", "PreprocessingAndTransformation.r", 
                    "FPN_Model.r", "UNET_Model.r", #"PSPNET_Model.r",
                    # "Loss2.r", #"LossFactory_Optimizers.r",
                    # "CustomGenerator_CropFix7.r",
                    "Inspection.r", "Postprocessing.r") %>%
    map( ~ paste0(RelPath, "/Scripts/FunctionCompilation/", .x))
for(l in 1:length(list_scripts)){
    source(list_scripts[[l]])
}

# for(cell in c("THP1", "A549")){

    # CELL = cell

    # Parameters for saving files
    Current_i = as.character(paste0("Img", IMAGE_SRC, as.character(HEIGHT), 
                                    "_", CLASS, "Class"                                
                                    ))
    # Save_dir = paste0(Unet_dir,"/",CELL,"/","Prediction/", Current_i, "--",1)

    # if(dir.exists(Save_dir) == T){
    #     list_dir = list.dirs(paste0(Unet_dir,"/",CELL,"/","Prediction"), full.names = T, recursive = F)
    #     my_dir = sub("--.*", "", Save_dir)
    #     my_dirs = which(grepl(my_dir, list_dir))
    #     Save_dir = paste0(sub("--.*", "", Save_dir), "--", (length(my_dirs) + 1))
    #     dir.create(Save_dir, showWarnings = F)
    # } else {
    #     dir.create(Save_dir, showWarnings = F)
    # }
    
    Save_dir = "/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgFLUO512_1Class--4"

    ## if one wants to work on an old folder
    # Save_dir = "/home/gendarme/Documents/UNet/BF_Data/THP1/Prediction/ImgBF512_1Class--1/--Arch_fpn--Enc_resnet101--Drop_5--Class_1--LR_5e-05--PT_8"

    ## Copy current script and associated parameters to training folder
    # file.copy(from = paste0(Unet_script_dir,"/",current_script), to = paste0(Save_dir, "/", current_script))
    # file.copy(from = paste0(Unet_script_dir,"/FunctionCompilation/Params.r"), to = paste0(Save_dir, "/Params.r"))
    # file.copy(from = paste0(Unet_script_dir,"/Data_postprocessing_full.r"), to = paste0(Save_dir, "/Data_postprocessing_full.r"))
    # file.copy(from = paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly.r"), to = paste0(Save_dir, "/DataAssembly.r"))

    ## load and pre-process data
    source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_repim_dapi.r"))

    # for(dp in 1:length(DROPOUT)){
        # for(arc in c("fpn", "unet")){
    #         print(paste0(dp, "_", arc))
    #     }
    # }
    
     # for(enc in c("null")){ #"null", "efficientnet_B1", "efficientnet_B2", "efficientnet_B3", "efficientnet_B4")){
     #     for(lr in c(2e-4)){#}, 1e-5)){ #, 5e-5)){
     #        for(fct in c(.8)){ #, 0.8)){
     #            for(pt in c(8)){ # 12
                    # for(redfct in c(8)){
                # for(cl in c(1)){#c(1, 2, 3)){
                    enc = "null"
                    lr = 2e-4
                    fct = .8
                    pt = 8
                    
                    # if(enc == "resnet101"){
                        # lr = 5e-5
                        callbacks_list = list(
                            callback_reduce_lr_on_plateau(monitor = "val_loss",
                                                        factor = fct,
                                                        patience = pt,
                                                        verbose = 1,
                                                        mode = "min", 
                                                        min_delta = 1e-4,
                                                        min_lr = 2e-06)
                                                        )
                    # } else {
                    # #     lr = 1e-4
                    #     callbacks_list = list()
                    # }
                    # DROPOUT = NULL
                    # EPOCHS = 2
                    # HEIGHT = 256
                    # WEIGHT = 256
                    # BATCH_SIZE = 16
                    # enc = "efficientnet_B0"
                    # dropout = ifelse(is.null(DROPOUT), 0, DROPOUT)
                    CLASS = 1 #if(exists("cl") == TRUE, cl, 1)
                    redfct = 8
                    arc = "unet"#ARCHITECTURE
                    # enc =  if(exists("enc") == TRUE, enc, "resnet101")
                    dropout = 0.5
                    # LR = 5e-05 #if(exists("lr") == TRUE, lr, 1e-4)
                    # lr = LR
                    LR = lr
                    # fct = 0.8

                    loop_id = paste0("--Arch_", arc,
                                    "--Enc_", enc,
                                    "--Drop_", dropout*10,
                                    "--Class_", CLASS,
                                    "--LR_", lr,
                                    "--PT_", pt
                                    #  "--FC_", fct#,
                                    # "--RedFct_", redfct
                                    )
                    
                    message(paste0("\n######################################################\n",
                                    loop_id,
                                    "\n######################################################"))

                    # build loop dir
                    # build plot dir and image dir
                    Save_loop = paste0(Save_dir, "/", loop_id) 
                    dir.create(Save_loop)
                    Save_plot = paste0(Save_loop, "/Plot")
                    dir.create(Save_plot)
                    Save_image = paste0(Save_loop, "/Image")
                    dir.create(Save_image)

                    ### LOAD MODEL
                    # if(ARCHITECTURE == "unet"){
                    if(arc == "unet"){
                        if(enc == "null"){
                            model = build_unet(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                                            backbone = NULL,
                                            nlevels = 5,
                                            upsample = 'upsampling', #c("upsampling", "transpose")
                                            output_activation = ACTIVATION,
                                            output_channels = CLASS,
                                            dec_filters = c(16, 32, 64, 128, 256, 512),
                                            dropout = c(dropout, 0, 0, 0, 0, 0)
                                            )
                        } else {
                            model = build_unet(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                                            backbone = enc,
                                            nlevels = NULL,
                                            upsample = 'upsampling', #c("upsampling", "transpose")
                                            output_activation = ACTIVATION,
                                            output_channels = CLASS,
                                            dec_filters = c(16, 32, 64, 128, 256, 512),
                                            dropout = c(dropout, 0, 0, 0, 0, 0)
                                            )
                        }
                        
                    # } else if(ARCHITECTURE == "fpn"){
                    } else if(arc == "fpn"){
                        model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                                            backbone = enc,
                                            nlevels = NULL,
                                            output_activation = ACTIVATION,
                                            output_channels = CLASS,
                                            decoder_skip = FALSE,
                                            dropout = dropout
                                            )
                    } else if(arc == "psp"){
                        model = PSPNet(backbone_name=enc,
                                    input_shape=c(HEIGHT, WIDTH, CHANNELS),
                                    classes=CLASS,
                                    activation=ACTIVATION,
                                    downsample_factor=redfct, # c(4, 8, 16)
                                    psp_dropout=0.5)
                    }

                    if(!(is.null(enc) & arc == "unet")){
                    # if(!(is.null(ENCODER) & ARCHITECTURE == "unet")){
                        model = freeze_weights(model, from = 1, to = 1)
                    }

                    train_input_test = train_input %>%
                        mutate(last_channel = CLASS,
                            Y = map2(Y, last_channel, ~ select_channels(.x, 1, .y)),
                            Y = map_if(Y, last_channel == 1, ~ add_dim(.x, 1))) %>%
                        select(-c(last_channel))
                    # str(train_input_test$Y, list.len = 2)
                    val_input_test = val_input %>%
                        mutate(last_channel = CLASS,
                            Y = map2(Y, last_channel, ~ select_channels(.x, 1, .y)),
                            Y = map_if(Y, last_channel == 1, ~ add_dim(.x, 1))) %>%
                        select(-c(last_channel))
                    # str(val_input_test$Y, list.len = 2)

                    ### COMPILE MODEL
                    # LossFactory(CLASS = CLASS)
                    make_loss = function(loss_name){
                        if(loss_name == "dice_coef_loss_bce_3_class"){
                            return(function(y_true, y_pred,
                                            l_b_c = .4,
                                            w_class_1 = .2,
                                            w_class_2 = .2,
                                            w_class_3 = .2){
                                k_mean(k_binary_crossentropy(y_true, y_pred)) * l_b_c +
                                    dice_coef_loss_for_bce(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
                                    dice_coef_loss_for_bce(y_true[,,,2], y_pred[,,,2]) * w_class_2 +
                                    dice_coef_loss_for_bce(y_true[,,,3], y_pred[,,,3]) * w_class_3
                            })
                        } else if(loss_name == "dice_coef_loss_bce_2_class") {
                            return(function(y_true, y_pred,
                                            l_b_c = .6,
                                            w_class_1 = .2, 
                                            w_class_2 = .2){
                                k_mean(k_binary_crossentropy(y_true, y_pred)) * l_b_c +
                                    dice_coef_loss_for_bce(y_true[,,,1], y_pred[,,,1]) * w_class_1 +
                                    dice_coef_loss_for_bce(y_true[,,,2], y_pred[,,,2]) * w_class_2
                                })
                        } else if(loss_name == "dice_coef_loss_bce_1_class"){
                            return(function(y_true, y_pred,
                                            dice = 0.5, bce = 0.5){
                                            k_binary_crossentropy(y_true, y_pred) * bce +
                                            dice_coef_loss_for_bce(y_true, y_pred) * dice
                                })
                        } else if(loss_name == "dice_coef_focal_1_class"){
                            return(function(y_true, y_pred,
                                            dice = 0.5, focal = 0.5){
                                            categorical_focal_loss(y_true, y_pred, gamma=2., alpha=.25) * focal +
                                            dice_coef_loss_for_bce(y_true, y_pred) * dice
                                            })
                        }
                    }

                    # loss_name = paste0("dice_coef_loss_bce_", CLASS, "_class")
                    loss_name = paste0("dice_coef_loss_bce_", CLASS, "_class")
                    loss = make_loss(loss_name = loss_name)

                    model <- model %>%
                        compile(
                            loss = loss,
                            optimizer = optimizer_adam(lr = LR, decay = DC),
                            metrics = custom_metric("dice_coef_loss_for_bce", dice_coef_loss_for_bce)
                    )

                    train_generator = custom_generator(data = train_input_test,
                                                    shuffle = TRUE,
                                                    scale = .2,
                                                    intensity_operation = FALSE,
                                                    batch_size = BATCH_SIZE)

                    val_generator = custom_generator(data = val_input_test,
                                                    shuffle = TRUE,
                                                    scale = .2,
                                                    intensity_operation = FALSE,
                                                    batch_size = BATCH_SIZE)

                    # callbacks_list <- list(
                    #     # callback_tensorboard(Save_dir),
                    #     ## callback_model_checkpoint(filepath = Save_dir,
                    #     ##                           monitor = "val_loss",
                    #     ##                          verbose = 1,
                    #     ##                           save_best_only = TRUE,
                    #     ##                           save_weights_only = TRUE,
                    #     ##                           mode = "min"
                    #     ##                           ),
                    #     # callback_reduce_lr_on_plateau(monitor = "val_loss",
                    #     #                               factor = fct,
                    #     #                               patience = pt,
                    #     #                               verbose = 1,
                    #     #                               mode = "min",
                    #     #                               min_delta = 1e-04,
                    #     #                               min_lr = 1e-06
                    #     #                               )#,
                    #     ## callback_early_stopping(patience = PATIENCE)
                    # )

                    history <- model %>% 
                        fit(
                            train_generator,
                            epochs = EPOCHS,
                            steps_per_epoch = STEPS_PER_EPOCHS,     # as.integer(nrow(train_input) / BATCH_SIZE),
                            validation_data = val_generator,
                            validation_steps = VALIDATION_STEPS,    # as.integer(nrow(val_input) / (BATCH_SIZE)),
                            verbose = 1L,
                            callbacks = callbacks_list
                            )

                    history_data = list(metrics = history$metrics,
                                        params = history$params)
                    save(history_data, file = paste0(Save_plot, "/HTR_data_", Current_i, "_", loop_id, ".rdata"))

                    HTR1 = plot(history) +
                        theme_light()
                    HTR2 = plot(history) +
                        scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
                        theme_light()
                    HTR3 = plot(history) +
                        scale_y_continuous(breaks = seq(0, 1, .05), minor_breaks =  seq(0, 1, .01)) +
                        theme_light()

                    ## Save the history of the training
                    ggsave(filename = paste0("HTR1_", Current_i, "_", loop_id, ".png"), plot = HTR1,
                            width = 6, height = 4, dpi = 1000, path = Save_plot)

                    # with extras #2
                    ggsave(filename = paste0("HTR2_", Current_i, "_", loop_id, ".png"), plot = HTR2,
                            width = 6, height = 4, dpi = 1000, path = Save_plot)

                    # with extras #3
                    ggsave(filename = paste0("HTR3_", Current_i, "_", loop_id, ".png"), plot = HTR3,
                            width = 6, height = 4, dpi = 1000, path = Save_plot)

                    # # # ## Save the model: 
                    if(!(enc %in% c("inception_resnet_v2", "seresnext101"))){
                        save_model_hdf5(model, filepath = paste0(Save_loop,"/unet_model_",loop_id,".hdf5"))
                        save_model_weights_hdf5(model, filepath = paste0(Save_loop,"/unet_model_weights_",loop_id,".hdf5"))
                    } else {
                        # model %>% export_savedmodel(paste0(Save_dir,"/savedmodel"), remove_learning_phase = TRUE)
                        save_model_weights_hdf5(model, filepath = paste0(Save_loop,"/unet_model_weights_",loop_id,".hdf5"))
                    }

                    ## generate plots for IOU and DICE
                    source(paste0(RelPath, "/Scripts/FunctionCompilation/IOU_DICE_plot.r"))

                    ## generate sample images
                    source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images.r"))

    #             }
    #         }
    #     }
    # }
# }
# mod_name = "unet_model_--Arch_fpn--Enc_resnet101--Drop_5--Class_1--LR_5e-05--PT_8.hdf5"
# # model = load_model_hdf5(filepath = paste0(Save_dir,"/unet_model_",Current_i, "_", loop_id,".hdf5"), compile = F)
# model = load_model_hdf5(filepath =  paste0(Save_dir, "/", mod_name), compile = FALSE)
# weight_name = "unet_model_weights_--Arch_fpn--Enc_resnet101--Drop_5--Class_1--LR_5e-05--PT_8.hdf5"
# # model %>% load_model_weights_hdf5(filepath = paste0(Save_dir,"/unet_model_weights_",Current_i, "_", loop_id, ".hdf5"))
# model %>% load_model_weights_hdf5(filepath = paste0(Save_dir, "/", weight_name))
# # ## dev set
# X_val = list2tensor(val_input$X)
# X_val = X_val[,1:HEIGHT, 1:WIDTH,]
# if(CHANNELS == 1){dim(X_val) = c(dim(X_val), 1)}
# Y_val = list2tensor(val_input$Y)
# Y_val = Y_val[,1:HEIGHT, 1:WIDTH,]

# pred_batch_size = 8
# # Y_hat <- predict(model, x = X_train)
# Y_hat_val = predict(model, 
#                     X_val,
#                     batch_size = pred_batch_size)
# # str(Y_hat_val)
# Nimg = 1
# display(abind(Y_hat_val[Nimg,,,1],
#               Y_val[Nimg,,],
#               along = 1))
# bw_label = Y_val[Nimg,,]
# bw_pred = Y_hat_val[Nimg,,,1]
# color_label = colorLabels(bwlabel(Y_val[Nimg,,]))
# hist(X_val[Nimg],,,1])
# bright_img = combine_col(image_1 = EBImage::normalize(X_val[Nimg,,,1], inputRange = c(-10, 50)),
# bright_img = combine_col(image_1 = EBImage::normalize(X_dapi_test[Nimg,,], inputRange = c(0.05, 0.12)),
#                          color_1 = "grey",
#                          dimension = c(dim(X_val[Nimg,,,1])))
# bright_img = UnSharpMasking(bright_img)
# display(bright_img)
# display(normalize(X_dapi_test[1,,]))
# hist(X_dapi_test[Nimg,,])
# dapi_img = combine_col(image_1 = EBImage::normalize(X_dapi_val[Nimg,,], inputRange = c(0.005, 0.07)),
# dapi_img = combine_col(image_1 = EBImage::normalize(X_test[Nimg,,,1], inputRange = c(-100, -0)),
#                        color_1 = "grey",
#                        dimension = c(dim(X_dapi_val[Nimg,,])))
# dapi_img = UnSharpMasking(dapi_img)
# display(dapi_img)
# display(normalize(X_test[Nimg,,,1]))
# hist(X_test[Nimg,,,1])
# dapi_blue_img = rgbImage(blue = dapi_img * 2)
# display(dapi_blue_img)
# display(abind(
#             #  bw_label,
#               color_label,
#               bright_img, along = 1))
# viridis_pred = img_to_viridis(Y_hat_val[Nimg,,,1])
# writeImage(bright_img, files = "/home/gendarme/Desktop/Img_presentation/bright_img.png")
# writeImage(bright_img, files = "/home/gendarme/Desktop/Img_presentation/bright_img_test.png")
# writeImage(dapi_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_img.png")
# writeImage(dapi_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_img_test.png")
# writeImage(dapi_blue_img, files = "/home/gendarme/Desktop/Img_presentation/dapi_blue_img.png")
# writeImage(color_label, files = "/home/gendarme/Desktop/Img_presentation/color_label.png")
# writeImage(bw_label, files = "/home/gendarme/Desktop/Img_presentation/bw_label.png")
# writeImage(viridis_pred, files = "/home/gendarme/Desktop/Img_presentation/viridis_pred.png")
# writeImage(bw_pred, files = "/home/gendarme/Desktop/Img_presentation/bw_pred.png")
