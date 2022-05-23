library(keras)
use_implementation("tensorflow")
tensorflow::tf_gpu_configured()
library(tidyverse)
library(glue)
library(EBImage)
options(EBImage.display = "raster")
options(EBImage.bg = "black")
library(rstudioapi) # necessary for locating scripts
library(oro.nifti) # we do work with niftis 
library(jsonlite)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/BFSeg', '~/Documents/BFSeg')

# Load custom functions
list_scripts = list("Params_nnUNet_comparaison.r",
                    "PreprocessingAndTransformation.r",
                    "Load_data_from_disk.r",
                    "Model_FPN.r",
                    "Model_UNET.r", 
                    "Model_PSPNET.r", 
                    "Model_Deeplabv3plus_v3.r",
                    "Backbone_Zoo.r",
                    "Loss.r",
                    "CustomGenerator_CropFix.r",
                    "Inspection.r",
                    "Postprocessing.r")
for(l in 1:length(list_scripts)){
    source(paste0(RelPath, "/Scripts/FunctionCompilation/", list_scripts[[l]]))
    print(paste0("Load script #", ifelse(l < 10, paste0("0", l), l), " : ", list_scripts[[l]]))
}

# Parameters for saving files
Current_i = as.character(paste0("Img", IMAGE_SRC, as.character(HEIGHT),
                                "_", CLASS, "Class" 
                                )
                        )
Save_dir = paste0(Unet_dir,"/",CELL,"/","Prediction/", Current_i, "--",1)

if(dir.exists(Save_dir) == T){
    list_dir = list.dirs(paste0(Unet_dir,"/",CELL,"/","Prediction"), full.names = T, recursive = F)
    my_dir = sub("--.*", "", Save_dir)
    dirs2keep = list_dir[which(grepl(my_dir, list_dir))]
    dirs_id = as.integer(str_remove(dirs2keep, paste0(my_dir, "--")))
    Save_dir = paste0(sub("--.*", "", Save_dir), "--", (max(dirs_id) + 1))
    dir.create(Save_dir)
} else {
    dir.create(Save_dir, showWarnings = F)
}

## Copy current script and associated parameters to training folder
file.copy(from = paste0(Unet_script_dir,"/",current_script), to = paste0(Save_dir, "/", current_script))
file.copy(from = paste0(Unet_script_dir,"/FunctionCompilation/Params_nnUNet_comparaison.r"), to = paste0(Save_dir, "/Params_nnUNet_comparaison.r"))
file.copy(from = paste0(Unet_script_dir,"/FunctionCompilation/Load_data_from_disk.r"), to = paste0(Save_dir, "/DataAssembly_nnUNet_comparaison.r"))

## load and pre-process data
# source(paste0(Unet_script_dir,"/FunctionCompilation/DataAssembly_nnUNet_comparaison.r"))
# set folders
task_dir = paste0("/home/gendarme/Documents/nnUNet/nnUNet_raw/nnUNet_raw_data/Task00", ifelse(CLASS == 1, 1, 2), "_A549")
image_folder = paste0(task_dir, "/imagesTr")
label_folder = paste0(task_dir, "/labelsTr")
image_folder_test = paste0(task_dir, "/imagesTs")
label_folder_test = paste0(task_dir, "/labelsTs")

# load data for random split in train and validation
input = load_data(image_folder, label_folder)
# load data for test at inference time
test_input = load_data(image_folder_test, label_folder_test)
# display one example for visual inspection
display(input$Y[[1]][,,1] + input$Y[[1]][,,2])

# loop for 5 fold cross validation
for(CV in 1:5){

    enc = ENCODER
    lr = LR
    fct = FACTOR
    pt = PATIENCE
    redfct = 8L
    arc = ARCHITECTURE
    dropout = DROPOUT

    loop_id = paste0(arc, "--", enc, "--Epochs_", EPOCHS, "--Minibatches_", STEPS_PER_EPOCHS, "--FOLD_", CV)

    message(paste0("\n######################################################\n",
                   loop_id,
                   "\n######################################################"))

    # build loop dir, plot dir and image dir
    Save_loop = paste0(Save_dir, "/", loop_id) 
    dir.create(Save_loop, showWarnings = F)
    Save_plot_semantic = paste0(Save_loop, "/Plot_semantic")
    dir.create(Save_plot_semantic, showWarnings = F)
    Save_image_semantic = paste0(Save_loop, "/Image_semantic")
    dir.create(Save_image_semantic, showWarnings = F)
    Save_plot_instance = paste0(Save_loop, "/Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
    Save_image_instance = paste0(Save_loop, "/Image_instance")
    dir.create(Save_image_instance, showWarnings = F)

    # set seed for comparability of first fold  if wished (not really recommended):
    # if(CV == 1){ set.seed(11) }

    # split the data in train and validation
    sampling_generator(input, val_split = VALIDATION_SPLIT)

    # write to json the index of train and validation data + parameters
    json_param = list(Cell = CELL,
                     Epoch = EPOCHS,
                     Minibatch = STEPS_PER_EPOCHS,
                     Batch_size = BATCH_SIZE,
                     Optimizer = OPTIMIZER,
                     Initial_lr = LR,
                     Architecture = ARCHITECTURE,
                     Encoder = ENCODER,
                     Train_index = train_input$image_id,
                     Val_index = val_input$image_id
                     )
    json_param = toJSON(json_param, auto_unbox = TRUE, pretty = TRUE)
    write_json_data_index(json_param, paste0(Save_loop, "/detail_param_", "fold_",  CV, ".json"))

    # display one image for sanity check
    display(paintObjects(
        Image(EBImage::normalize(train_input$Y[[1]][,,ifelse(ACTIVATION == "sigmoid", 1, 2)]), colormode = "Grayscale"),
        Image(transform_gray_to_rgb_rep(EBImage::normalize(train_input$X[[1]][,,1])), colormode = "Color"),
        col = "red",
        thick = TRUE
        ))

    ## polynomial learning rate decay scheduler
    schedule = function(epoch, lr){
        return(LR * (1 - epoch/EPOCHS)^0.9)
    }

    ## Callback for printing the LR at the end of each epoch.
    PrintLR = R6::R6Class("PrintLR",
        inherit = KerasCallback,
        
        public = list(    
            losses = NULL,
            on_epoch_end = function(epoch, logs = list()) {
            tf$print(glue('\nLearning rate for epoch {epoch} is {as.numeric(model$optimizer$lr)}\n'))
            }
    ))
    print_lr = PrintLR$new()
    
    callbacks_list = list(
        callback_learning_rate_scheduler(schedule),
        print_lr,
        callback_model_checkpoint(filepath = paste0(Save_loop, "/best_model_weights.hdf5"),
                                  monitor = "val_loss",
                                  save_weights_only = TRUE)
    )

    # build the model
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Model_Factory.r"))
    if(!(is.null(enc) & arc == "unet")){
        model = freeze_weights(model, from = 1, to = 1)
    }
    
    ### COMPILE MODEL ########################################################################################
    loss_name = paste0("dice_ce_", CLASS, "_class", ifelse(ACTIVATION == "softmax", "_softmax", ""))
    loss = make_loss(loss_name = loss_name)

    model = model %>%
        compile(
            loss = loss,
            optimizer = if(OPTIMIZER == "SGD"){
                            tf$keras$optimizers$SGD(learning_rate = LR,
                                                    momentum = 0.99,
                                                    nesterov = TRUE)
                        } else if(OPTIMIZER == "ADAM"){
                            optimizer_adam(learning_rate = LR, decay = DC)
                        },
            metrics = custom_metric(ifelse(ACTIVATION == "softmax", "dice_target", "dice"),
                                    ifelse(ACTIVATION == "softmax", dice_target, dice))
    )
        
    train_generator = custom_generator(data = train_input,
                                    shuffle = TRUE,
                                    scale = SCALE,
                                    intensity_operation = TRUE,
                                    batch_size = BATCH_SIZE)

    val_generator = custom_generator(data = val_input,
                                    shuffle = TRUE,
                                    scale = SCALE,
                                    intensity_operation = TRUE,
                                    batch_size = BATCH_SIZE)

    ### TRAIN #################################################################################################
    history = model %>% 
        fit(
            train_generator,
            epochs = EPOCHS,
            steps_per_epoch = STEPS_PER_EPOCHS,
            validation_data = val_generator,
            validation_steps = VALIDATION_STEPS,
            verbose = 1L,
            callbacks = callbacks_list
            )

    ### SAVE & PLOT ############################################################################################
    history_data = list(metrics = history$metrics,
                        params = history$params)
    save(history_data, file = paste0(Save_plot_semantic, "/HTR_data_", Current_i, "_", loop_id, ".rdata"))

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
            width = 6, height = 4, dpi = 1000, path = Save_plot_semantic)

    # with extras #2
    ggsave(filename = paste0("HTR2_", Current_i, "_", loop_id, ".png"), plot = HTR2,
            width = 6, height = 4, dpi = 1000, path = Save_plot_semantic)

    # with extras #3
    ggsave(filename = paste0("HTR3_", Current_i, "_", loop_id, ".png"), plot = HTR3,
            width = 6, height = 4, dpi = 1000, path = Save_plot_semantic)

    # training history a la Isensee
    source(paste0(RelPath, "/Scripts/FunctionCompilation/plot_history.r"))

    ## Save the model: 
    if(!(enc %in% c("inception_resnet_v2",          # models can't be saved
                    "seresnext50",  "seresnext101", # because of 
                    "resnext50", "resnext101",      # lambda layers in R
                    "senet154"
                    ))){
        save_model_hdf5(model, filepath = paste0(Save_loop,"/model.hdf5"))
        save_model_weights_hdf5(model, filepath = paste0(Save_loop,"/model_weights.hdf5"))
    } else {
        save_model_weights_hdf5(model, filepath = paste0(Save_loop,"/model_weights.hdf5"))
    }

    ### Generate single instances and perform measurements #####################################################
    ## generate plots for IOU and DICE
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Inspection/IOU_DICE_plot_2.r"))

    ## generate representative images
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Rep_Images_no_dapi.r"))
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Inspection/Rep_Images_no_dapi_2.r"))

    ## generate plots for single dice (boxplot dice instances, lin regression)
    ## generate rep. images for instance generation
    source(paste0(RelPath, "/Scripts/FunctionCompilation/Inspection/DICE_single.r"))

    ## generate very simple instance segmentation
    # source(paste0(RelPath, "/Scripts/FunctionCompilation/Inspection/Postprocessing_1c.r"))

}
