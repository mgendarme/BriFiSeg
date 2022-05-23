arraynum = function(image){
    return(array(as.numeric(image), dim = c(dim(image)[1:2])))
}

load_data = function(image_path, label_path){
    data = tibble(Image_Path = list.files(paste0(image_path), recursive = T),
                  Label_Path = list.files(paste0(label_path), recursive = T)) %>%
        filter(str_detect(Image_Path, "nii")) %>%
        mutate(# generate full path
                Image_Path = paste0(image_path, "/", Image_Path),
                Label_Path = paste0(label_path, "/", Label_Path),
                channels = CHANNELS,
                encoder = ifelse(is.null(ENCODER), "null", "pretrained-encoder"),
                image_id = sapply(str_split(sapply(str_split(Image_Path, paste0(CELL, "_")), "[", 2), "_"), "[", 1),
                
                # load images
                X = map(Image_Path, ~ readNIfTI(fname = .x)),
                X = map(X, ~ arraynum(.x)),
                X = map(X, ~ add_dim(.x, 1)),
                
                # preprocess the images according to the presence/absence of pretrained encoder
                X = map_if(X, channels == 3L, transform_gray_to_rgb_rep),
                X = map_if(X, encoder != "null", EBImage::normalize, separate = T, ft = c(0, 255)),
                X = map_if(X, encoder != "null", imagenet_preprocess_input, mode = "torch"),
                X = map_if(X, encoder == "null", ctr_std),
                
                # load labels
                Y = map(Label_Path, ~ readNIfTI(fname = .x)),
                Y = map(Y, ~ arraynum(.x)),
                Class = CLASS,
                Y = map_if(Y, Class == 1L, ~ add_dim(.x, 1)),

                # one hot encoding for background and foreground in case of softmax output activation
                ACT = ACTIVATION,
                Y = map_if(Y, ACT == "softmax", ~ to_categorical(.x, num_classes = CLASS + 1)) 
                ) %>% 
        select(X, Y, image_id)#, Image_Path)
    return(data)
}

sampling_generator = function(data, val_split = VALIDATION_SPLIT) {
        train_index = sample(1:nrow(data), as.integer(round(nrow(data) * (1 - val_split), 0)), replace = F)
        val_index = c(1:nrow(data))[!c(1:nrow(data) %in% train_index)]
        train_input <<- data[train_index,]
        val_input <<- data[val_index,]
}

write_json_data_index = function(data, filename){
    sink(filename)
    print(data)
    sink()
}

