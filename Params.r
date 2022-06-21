### list of param to load for model training
CELL = "A549"                           # c("A549", "RPE1", "HELA", "MCF7")
### set path to data down here
dataset = "Task001_A549"
Unet_dir = ifelse(grepl("Windows", sessionInfo()$running),
                  paste0('~/BF_Data/', dataset),
                  paste0('~/Documents/BF_Data/', dataset))
### set path to scripts down here if not already done in the train or inference script
Unet_script_dir = paste0(RelPath)
# current_script = sub(".*/", "", rstudioapi::getActiveDocumentContext()$path)

# model type
ARCHITECTURE = c("unet")                 # c("unet", "fpn", "psp", "deeplab")
ENCODER = c("seresnext101")              # see at tthe bottom of to find out which encoders are possible

# image/mask settings
HEIGHT = 512L
WIDTH = 512L                          
CHANNELS = ifelse(ARCHITECTURE == "unet" & is.null(ENCODER), 1L, 3L)
SHAPE = as.integer(c(WIDTH, HEIGHT, CHANNELS))
CLASS = 1L
SHAPE_MASK = as.integer(c(WIDTH, HEIGHT, CLASS))
# UnSharpMasking_Power = 5
IMAGE_SRC = "BF"                         # c("FLUO", "BF") => c("FLUO_NUC", "FLUO_CYTO", "BF")
SCALE = .4

# model settings
ACTIVATION = "softmax"                   # c("sigmoid", "softmax")
# learning rate decay
DC = 1e-4 
FACTOR = c(.8)
OPTIMIZER = "ADAM"                       # c("SGD", "ADAM")
# learning rate
LR = ifelse(OPTIMIZER == "ADAM", 1e-3, 1e-2)
PATIENCE = 8L
DECODER_SKIP = FALSE                     # if using residual connection in decoder of FPN is required
# FILTERS = 64 # For "unet" only
DROPOUT = 0

# learning settings
VALIDATION_SPLIT = 0.2                   # .15 for val, .15 for test
EPOCHS = 200L                            # 1 EPOCH = 1 Forward pass + 1 Backward pass for ALL training samples
if(!is.null(ENCODER)){
  if(ENCODER == "senet154"){
    STEPS_PER_EPOCHS = 100L              # as.integer(nrow(train_input) / BATCH_SIZE)  
    BATCH_SIZE = 4L                      # BATCH_SIZE = Number of training samples in (1 Forward / 1 Backward) pass
  } else {
  STEPS_PER_EPOCHS = 50L
  BATCH_SIZE = 8L
  }
} else {
  STEPS_PER_EPOCHS = 50L
  BATCH_SIZE = 8L 
}  
# theroretical minibatches to compare to 250 minibatches annd batch size 11
VALIDATION_STEPS = as.integer(STEPS_PER_EPOCHS / 5L)

# Weighting pararameters:
# second possible choice for weights .4, .2, .2, .2 instead of .7, .1, .1, .1
# would lead to less weight on categorical crossentropy
L_B_C = ifelse(CLASS == 3L, 0.4, 0.6) # loss binary / categorical crossentropy weight
W_CLASS_1 = 0.2                       # object weight
W_CLASS_2 = 0.2                       # border or interface weight if only 2 classes (check parameters for keep_channels() and select_channels() )
W_CLASS_3 = 0.2                       # interfeace weight if 3 classes are getting classifyied
TypeOfObjects = "nuc"                 # c("nuc", "cyto")
SND_CLASS = "interface"               # c("border", "interface")

## inspection 
TRESH_PRED = 0.5

# possible_values_for_encoder = c(
#   NULL, # only for unet architecture
#   "resnet50", "resnet101", "resnet152,
#   "resnet50v2", "resnet101v2", "resnet152v2",
#   "resnext50", "resnext101",
#   "seresnext50", "seresnext101",
#   "senet154",
#   "xception",
#   "inception_resnet_v2",
#   "efficientnet_B0", "efficientnet_B1", "efficientnet_B2",
#   "efficientnet_B3", "efficientnet_B4",
#   past efficientnet_B4 model gets to big with input images of size c(512, 512, 3) & 24 GB of GPU memory & batch size 8
#   "efficientnet_B5", "efficientnet_B6", "efficientnet_B7",
#   "nasnet_mobile", "nasnet_large",
#   "densenet121", "densenet169", "densenet201"
#   )
