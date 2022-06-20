## Prepare training images and metadata
TRAIN_PATH = paste0(Unet_dir,"/BATCH_",CELL,"/Train") 
TEST_PATH = paste0(Unet_dir,"/BATCH_",CELL,"/Test")

all_sample_out <- map_df(list.files(TRAIN_PATH,
                                    pattern = "_noBorder_shapedisc_thickness5_batch.csv",
                                    full.names = TRUE,
                                    recursive = TRUE),
                         read_csv, col_types = cols())#,

WELLS = unique(sapply(str_split(sapply(str_split(all_sample_out$ImageId, "Well"), "[", 2), "_"), "[", 1))[1:20]

# if(CELL == "A549"){
#     WELLS = unique(sapply(str_split(sapply(str_split(all_sample_out$ImageId, "Well"), "[", 2), "_"), "[", 1))[1:20]
# } else if(CELL == "THP1"){
#     WELLS = c("B02", "B03", "B04", "B10", "B05", "B06", "B09", "B11",
#     # c("B02", "B03", "B04", "B10", # "B05", "B06", "B09", "B11", "C02", "C03", "C04", "C06)
#     ### more data to double the training set
#     'B07', 'B08', 'B13', 'B14', 'B16', 'B21', 'B22', 'B23', 'C07', 'C08', 'C09', 'C10')
# } ifelse(CELL == "THP1", 3, 5))

data <- all_sample_out %>%
  mutate(ImageFile = file.path(paste0(Unet_dir, "/BATCH_",CELL,"/Image/", ImageId)),
         ImageShape =  list(c(512, 512)),  #map(ImageFile, .f = function(file) dim(readImage(file))[1:2]),
         Plate = 1,#as.numeric(str_sub(ImageId, 16, 18)),
         ID = sapply(str_split(sapply(str_split(ImageId, "Well"), "[", 2), "_"), "[", 1),
        #  Position = as.numeric(sapply(str_split(sapply(str_split(ImageId, "_"), "[", 3), "_Channel"), "[", 1)),
         Position = as.numeric(sapply(str_split(ImageId, "_"), "[", 3)),
         Channel = sapply(str_split(sapply(str_split(ImageId, "Channel"), "[", 2), "_"), "[", 1),
         Crop = map(str_split(Crop, " "), as.numeric)
  ) %>% select(-c("Channel")) %>%
  filter(ID %in% c(WELLS)) %>% 
  filter(!is.na(Interface3)) %>%
  glimpse

image_data <- tibble(ImageId_BF = list.files(paste0(Unet_dir, "/BATCH_",CELL,"/Image"), recursive = T)) %>% 
  mutate(ImageFile_BF = file.path(paste0(Unet_dir, "/BATCH_",CELL,"/Image/", ImageId_BF)),
         Plate = 1,#as.numeric(str_sub(ImageId, 16, 18)),
         ID = sapply(str_split(sapply(str_split(ImageId_BF, "Well"), "[", 2), "_"), "[", 1),
         Position = as.numeric(sapply(str_split(ImageId_BF, "_"), "[", 3)),
         Channel = sapply(str_split(sapply(str_split(ImageId_BF, "Channel"), "[", 2), "_"), "[", 1), 
         ZStack = as.numeric(sapply(str_split(sapply(str_split(ImageId_BF, "ZStack"), "[", 2), "_"), "[", 1))) %>%
  filter(ID %in% c(WELLS))#, WELLS_TEST))

CHANNEL = ifelse(IMAGE_SRC == "BF", "DIA", "DAPI")
if(length(ZSTACK) == 1){
  if(CHANNEL == "DIA"){
    image_data_1c <- image_data %>%
      filter(ZStack == ZSTACK[1] & Channel == CHANNEL) %>% #glimpse
      mutate(ImageFile_ = ImageFile_BF,
             ImageFile_Dapi = filter(image_data, Channel == "DAPI") %>% select(c("ImageFile_BF")) %>% pull)
    data <- data %>%
      left_join(filter(image_data_1c, ZStack == ZSTACK[1]), by = c("Plate", "ID", "Position"))  %>%
      glimpse()
  } else if(CHANNEL == "DAPI"){ 
    image_data_1c <- image_data %>%
      filter(Channel == CHANNEL) %>%
      mutate(ImageFile_ = filter(image_data, Channel == "DAPI") %>% select(c("ImageFile_BF")) %>% pull,
             ImageFile_Dapi = filter(image_data, ZStack == ZSTACK[1] & Channel == "DIA") %>% select(c("ImageFile_BF")) %>% pull)
    data <- data %>%
      left_join(image_data_1c, by = c("Plate", "ID", "Position"))  %>%
      glimpse()
  }
  
} else if(length(ZSTACK) == 3){
  image_data_3c <- image_data %>%
    filter(ZStack == ZSTACK[1] & Channel == CHANNEL) %>%
    mutate(img1 = filter(image_data, ZStack == ZSTACK[1] & Channel == CHANNEL) %>% select(c("ImageFile_BF")) %>% pull,
           img2 = filter(image_data, ZStack == ZSTACK[2] & Channel == CHANNEL) %>% select(c("ImageFile_BF")) %>% pull,
           img3 = filter(image_data, ZStack == ZSTACK[3] & Channel == CHANNEL) %>% select(c("ImageFile_BF")) %>% pull,
           ImageFile_ = pmap(list(img1, img2, img3), list),
           ImageFile_Dapi = filter(image_data, Channel == "DAPI") %>% select(c("ImageFile_BF")) %>% pull
           )
  data <- data %>% 
    left_join(filter(image_data_3c), by = c("Plate", "ID", "Position")) %>%
      glimpse()
}

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

# a = readImage(data$ImageFile_BF[1], type = "tiff")
# preprocess_image_TL(data$ImageFile_[1], shape = c(1000, 1000), cropping = unlist(data$Crop[1]))

# for(inp in c("preproc", "nopreproc")){
# generate input data:
if(length(ZSTACK) == 1){
  input <- data %>% 
    mutate(X = map2(ImageFile_, Crop, ~ preprocess_image_TL(.x, shape = c(1000, 1000), cropping = .y)),
           Chan = CHANNELS,
           # Mod = ARCHITECTURE[2],
           # ZStack = ZSTACK,
           # X = map_if(X, Chan == 3L, transform_gray_to_rgb_rep),
           Y = map(Mask, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
           B = map(Border3, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
           ## error handling in case there aren't any interfaces encoded
           I = map_if(Interface3, Interface3 != "1 1000000", rle2masks, c(1000, 1000)), #cropping = HEIGHT),)
           I = map_if(I, Interface3 == "1 1000000", ~ array(data = 0, dim = c(1000, 1000)))) %>%
        select(X, Y, B, I, ID)
} else if(length(ZSTACK) == 3){
  input <- data %>% #sample_n(train_data, nrow(train_data)) %>%
    mutate(Y = map(Mask, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
          B = map(Border3, rle2masks, c(1000, 1000)),#, cropping = HEIGHT),
          #  I = map(InterfaceEncodedPixels, ~ ifelse(InterfaceEncodedPixels == "1 1000000",
          #                                            ~ array(data = 0, dim = c(1000, 1000)),
          #                                            ~ rle2masks(.x , c(1000, 1000)))), # cropping = HEIGHT),)
          I = map_if(Interface3, Interface3 != "1 1000000", rle2masks, c(1000, 1000)), #cropping = HEIGHT),)
          I = map_if(I, Interface3 == "1 1000000", ~ array(data = 0, dim = c(1000, 1000))),
          # X = map2(ImageFile_BF, Crop, ~ preprocess_image_TL(.x, shape = c(WIDTH, HEIGHT), cropping = HEIGHT, .y)),
          # X = map2(ImageFile_BF, Crop, ~ preprocess_image_TL_3c(.x, shape = c(1000, 1000), cropping = .y)))#,
          X1 = map2(img1, Crop, ~ preprocess_image_TL(.x, shape = c(1000, 1000), cropping = .y)),
          X2 = map2(img2, Crop, ~ preprocess_image_TL(.x, shape = c(1000, 1000), cropping = .y)),
          X3 = map2(img3, Crop, ~ preprocess_image_TL(.x, shape = c(1000, 1000), cropping = .y)),
          Chan = CHANNELS,
          X = map_if(X1, Chan == 3L, transform_gray_to_rgb_rep),
          X = map2(X, X2, ~ swap_channel(.x, .y, 2)),
          X = map2(X, X3, ~ swap_channel(.x, .y, 3))) %>%
    select(X, Y, B, I, ID)
}
# str(input, list.len = 3)
input = input %>%
  mutate(Img_Src = IMAGE_SRC,
         #  Inp =  inp,
         #  X = map_if(X, Inp == "nopreproc", EBImage::normalize),
         #  X = map_if(X, Inp == "preproc", EBImage::normalize, separate = T, ft = c(0, 255)),
         #  X = map_if(X, Inp == "preproc", imagenet_preprocess_input, mode = "caffe"), # tf, caffe, torch
         # X = map(X, transform_gray_to_rgb_rep),
         # X = map(X, EBImage::normalize, separate = T, ft = c(0, 255)),
         # X = map(X, imagenet_preprocess_input, mode = "torch"),
         # X = map(X, imagenet_preprocess_input, mode = "caffe"), # tf, caffe, torch
         #  X = map_if(X, Img_Src == "BF", UnSharpMasking, 4), 
         #  X = map_if(X, Img_Src == "BF", filter2, sharp)
  ) %>%
  select(X, Y, B, I, ID) %>%
  mutate(Activation = ACTIVATION,
         Class = CLASS,
         # Y = map_if(Y, Activation == "softmax" & CLASS == 3, softmax_transf_channel, 3),
         # Y = map_if(Y, Activation == "softmax" & CLASS == 2, softmax_transf_channel, 2),
         # Y = map(Y, resize_masks, shape = SHAPE_MASK),
         B = map(B, ~ ifelse( .x == 1, 2, .x)),
         I = map(I, ~ ifelse( .x == 1, 3, .x)),
         Y = map2(Y, B, ~ .x - .y),
         Y = map2(Y, I, ~ .x - .y),
         Y = map(Y, ~ ifelse( .x < 0, 0, .x)),
         Y = map2(Y, B, ~ .x + .y),
         Y = map2(Y, I, ~ .x + .y),
         Y = map(Y, ~ ifelse( .x > 2, 3, .x)),
         # hist(input$Y[[1]])
         # display(input$Y[[1]] == 5)
         Y = map(Y, ~ keras::to_categorical( .x, 4)),
         Y = map(Y, sum_channels, 3, 3, 4),
         Y = map(Y, diff_channels, 2, 2, 4),
         Y = map(Y, ~ ifelse( .x > 0, 1, 0)),
         Y = map_if(Y, Class == 1, ~ select_channels(.x, 2, 2)),
         Y = map_if(Y, Class == 1, ~ add_dim(.x, 1)),
         Y = map_if(Y, Class == 2, ~ keep_channels(.x, 2, ifelse(SND_CLASS == "border", 3, 4))),
         Y = map_if(Y, Class == 2, ~ select_channels(.x , 1, 2)),
         Y = map_if(Y, Class == 3, ~ select_channels(.x , 2, 4))#,
         # Y = map(Y, transform_gray_to_rgb_rep)
  ) %>% 
  select(X, Y, ID) # %>%
 # str(input2, list.len = 4)
# unique(as.numeric(input2$[[12]]))


## display to check inputs
# display(abind(
#     # abind(
#     # EBImage::normalize(readImage(data$img1[[1]])[1:256, 1:256]),
#     # EBImage::normalize(readImage(data$img2[[1]])[1:256, 1:256]),
#     # EBImage::normalize(readImage(data$img3[[1]])[1:256, 1:256]),
#     # along = 1),
#     abind(
#     normalize(input$X[[1]])[1:256, 1:256, 1],
#     normalize(input$X[[1]])[1:256, 1:256, 2],
#     normalize(input$X[[1]])[1:256, 1:256, 3],
#     along = 1)
#     ), bg = "black")

## Randomise samples and split into train and test sets:
TRAIN_INDEX <- sample(1:nrow(input),
                      as.integer(round(nrow(input) * (1 - VALIDATION_SPLIT), 0)),
                      replace = F)
NOT_TRAIN_INDEX <- c(1:nrow(input))[!c(1:nrow(input) %in% TRAIN_INDEX)]
VAL_TEST_INDEX <- sample(NOT_TRAIN_INDEX,
                         as.integer(round(length(NOT_TRAIN_INDEX) * 0.5, 0)),
                         replace = F)

sampling_generator <- function(data, train_index = TRAIN_INDEX, val_index = VAL_TEST_INDEX) {
  train_input <<- data[train_index,]
  val_input <<- data[val_index,]
  test_input <<- data[-c(val_index, train_index),]
}
sampling_generator(input)

# str(input, list.len = 3)
# display(
#     abind(
#       normalize(train_input$X[[1]][,,1]),
#       train_input$Y[[1]][,,1],
#       normalize(val_input$X[[1]][,,1]),
#       val_input$Y[[1]][,,1],
#       along = 1
#     )
# )
# 
# display(
#   abind(
#     normalize(input2$X[[1]][,,1]),
#     input2$Y[[1]][,,1],
#     # input$B[[1]][,],
#     along = 1
#   )
# )
