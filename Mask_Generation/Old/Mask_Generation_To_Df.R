library(EBImage)
library(tidyverse)

# Settings#############################################################################################################################################
UnetTrain_dir <- "/home/gendarme/Documents/U-Net/PMI_Data/Original_IF012/Train/"
base_dir <- "/home/gendarme/Documents/U-Net/PMI_Data/Original_IF012"
TRAIN_PATH = paste0(base_dir,"/Train--Normalize-Range_1-2") 
HEIGHT = 1000
WIDTH = 1000
CHANNELS = 1          # only grayscale
SHAPE = c(WIDTH, HEIGHT, CHANNELS)

## Prepare training images and metadata
ImageMaskTrain <- list.files(base_dir, pattern = ".tif", recursive = T) %>% str_replace(".tif", "") %>%  as.data.frame()
colnames(ImageMaskTrain)[1] <- "ImageId"
ImageTrain <- filter(ImageMaskTrain, str_detect(ImageId, "mCherry|DAPI"))#|NImg405|Img568|Img405"))
ImageIdSimp <- str_split(ImageTrain$ImageId, "/", simplify = T)[,3]

#MaskTrain <- filter(ImageMaskTrain, str_detect(ImageId, "cytoMask"))
train_data <- data.frame(ImageId = ImageTrain$ImageId) %>%
  #mutate(MaskId = MaskTrain$ImageId) %>%
  mutate(ImageFile = file.path(base_dir, paste0(ImageId, ".tif")),
         ImageIdSimp = ImageIdSimp,
         NucEncodedPixels = NA,
         CytoEncodedPixels = NA,
         ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]),
         Plate = as.numeric(str_sub(ImageId, 16, 18)),
         ID = sub("--W.*", "", ImageIdSimp),
         Position = as.numeric(sapply(str_split(sapply(str_split(ImageIdSimp, "--P"), "[", 2), "--"), "[", 1)),
         Channel = sub(".*--", "", ImageId)
         )
IdToMap <- function(FileList){
  rN = c(1:16)
  cN = c(1:24)
  cID = c(1:24)
  rID = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P')
  Row <- data.frame(rID, rN)
  ID = data.frame(cID, cN)
  ID$rID <- NA
  for(i in rID){
    IDtemp <- ID[1:24,]
    IDtemp$rID <- as.character(i) 
    ID <- rbind(ID, IDtemp)
  }
  ID <- merge(ID[25:nrow(ID),], Row, by = "rID", all.x = T)
  ID$ID <- paste(ID$rID, ID$cID, sep = "")
  FileList <- merge(x = FileList, y = ID, by = c("ID"), all.x = T)
}
train_data <- IdToMap(train_data)
train_data <- filter(train_data, cN == 1)
enc_list <- list()
#n_sample <- as.tibble(train_data[FALSE,])
#n_sample_out <- as.tibble(train_data[FALSE,])
#################################### Encoding functions ######################################################################
# Encoding of mask in data frame for CNN: ####
## image2rle encodes the image pixels values
image2rle <- function(image){
  
  labels <- 1:max(image) ## assuming background  == 0
  
  x <- as.vector(t(image))
  
  encoding <- rle(x)
  
  ## Adding start positions
  encoding$positions <- 1 + c(0, cumsum(encoding$lengths[-length(encoding$lengths)]))
  
  ## encodes every individual mask bsaed on label
  mask2rle <- function(label, enc) {
    indices <- enc$values == label
    list(position = enc$positions[indices][1],
         encoding = paste(enc$positions[indices], enc$lengths[indices], collapse=" "))
  }
  
  ##return encodings with increasing positions
  map_df(labels, mask2rle, encoding) %>%
    arrange(position) %>%
    pull(encoding)
}

## postprocess_image resize the masks images and runs image2rle, eventually adds labels if not existing
postprocess_image <- function(image, shape){
  image <- resize(image[,,1], w = shape[1], h = shape[2])
  #image <- bwlabel(image > .5) ##binarize and label if no labels to start with
  image2rle(image)
}

## preprocess_image performs operations on images e.g. clahe, normalize, resize ...
preprocess_image <- function(file, shape){
  image <- readImage(file, type = "tiff")
  #image <- to_gray_scale(image)                       ## convert to gray scale  
  image <- resize(image, w = shape[1], h = shape[2])  ## make all images of dimensions
  image <- clahe(image)                               ## local adaptive contrast enhancement
  image <- normalize(image)                           ## standardize between [0, 1]
  #image <- invert(image)                              ## invert brightfield
  dim(image) <- c(dim(image), 1)
  imageData(image)                                    ## return as array
} ### Not necessary here

# Read images and extract features#############################################################################################
#ImageProcessingAndEncoding <- function(Image, Plate, Well, Position){
  ##For testing:
  # # ##Image = train_data
  Plate = 1
  Well = "D1"
  Position = 1
  Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position
  # # #---------------Rescale pixel intensity----------
  Rescale <- function(x){
    Bottom = 2^15/(2^16-1) 
    Top = (2^15+4095)/(2^16-1)
    (x - Bottom) / (Top - Bottom)   
  }
  
  #---------------Load Images------------------------
  #Position = "00001" #for testing
  
  Img405 = readImage(train_data[train_data$Channel == "DAPI" & Boolean, "ImageFile"])
  Img405 = Rescale(Img405)
  NImg405 = normalize(Img405, inputRange = c(range(Img405)[1], range(Img405)[2]))
  #display(NImg405)
  #writeImage(Img405, "Img405.tif")
  # Img488 = readImage(train_data[train_data$Channel == "GFP" & Boolean, "ImageFile"])
  # Img488 = Rescale(Img488)
  # NImg488 = normalize(Img488)
  #display(NImg488)
  #writeImage(Img488, "Img488.tif")
  Img568 = readImage(train_data[train_data$Channel == "mCherry" & Boolean, "ImageFile"])
  Img568 = Rescale(Img568)
  NImg568 = normalize(Img568, inputRange = c(range(Img568)[1], range(Img568)[2]))
  #display(NImg568)
  #writeImage(Img568, "Img568.tif")
  # Img647 = readImage(train_data[train_data$Channel == "647" & Boolean, "FileNames"])
  # Img647 = Rescale(Img647)
  # NImg647 = normalize(Img647, inputRange = c(range(Img647)[1], range(Img647)[2]))
  #display(NImg647)
  #writeImage(Img647, "Img647.tif")
  #---------------smooth and threshold nuleus------------------------------
  FilterNuc = makeBrush(size = 51, shape = "gaussian", sigma = 2)
  Img405smooth = filter2(Img405, filter = FilterNuc)
  # sharpen <- function(img){
  #   FilterHighPass = matrix(-1/15, nrow = 3, ncol = 3)
  #   FilterHighPass[2, 2] = 1
  #   img = filter2(img, FilterHighPass)
  # }
  # Img405sharp = sharpen(Img405)
  # display(normalize(Img405sharp))
  # Img405smoothSharp = filter2(Img405sharp, filter = FilterNuc)
  # Img405smoothSharp = Img405sharp
  # nucthrotsu <- Img405sharp > otsu(Img405sharp)
  # display(nucthrotsu)
  # 
  # # AO 
  # disc = makeBrush(51, "disc")
  # disc = disc / sum(disc)
  # offset = 0.005
  # nuc_bg = filter2( Img405sharp, disc )
  # nuc_th = Img405sharp > nuc_bg# + offset
  # display(nuc_th, all=TRUE)
  
  # Img405sharpSmooth = filter2(Img405sharp, filter = FilterNuc)
  # display(Img405sharpSmooth)
  # nucThrManual2 = thresh(Img405sharp, w = 100, h = 100, offset = 0.04)
  # display(nucThrManual2)
  #display(normalize(Img405smooth))
  #nucThrManual = thresh(Img405smooth, w = 100, h = 100, offset = 0.001)
  nucThrManual = thresh(Img405smooth, w = 100, h = 100, offset = 0.001)
  #display(nucThrManual)
  nucOpening = nucThrManual
  #nucOpening = opening(nucThrManual, kern = makeBrush(15, shape="disc"))
  #nucClosing = erode(nucRegions, makeBrush(2, shape = "disc"))
  #display(nucOpening)
  nucSeed = bwlabel(nucOpening)
  #display(colorLabels(nucSeed))
  nucFill = fillHull(thresh(Img405smooth, w = 20, h = 20, offset = 0.005))
  #display(nucFill)
  nucRegions = propagate(Img405smooth, nucSeed, mask = nucFill)
  #display(nucRegions)
  nucMask = watershed(distmap(nucRegions), tolerance = 1, ext = 1)
  #display(colorLabels(nucMask))
  NImgCol405 = rgbImage(blue = NImg405*3)
  #display(NImgCol405)
  nucSegm = paintObjects(nucMask, NImgCol405, col = 'red')
  #display(nucSegm, all = T)
  
  #------------------Generate voronoi-------------------------------------
  VoR = propagate(nucMask, nucMask, lambda = 100000)
  # display(colorLabels(VoR))
  # #------------------Generate donut and bubble for proxy Cytoplasm -------
  # Bubble = dilate(nucRegions, kern = makeBrush(15, shape = 'disc'))
  # #BubbleVoR = selfComplementaryTopHat(Bubble, VoR)
  # #display(colorLabels(BubbleVoR))
  # BubbleBound = propagate(Bubble, nucMask, Bubble, lambda = 1000)
  # # display(colorLabels(BubbleBound))
  # Donut = BubbleBound - nucMask
  # # display(colorLabels(Donut))
  # # DonutSegm = paintObjects(Donut, NImgCol405, col = 'red')
  # # display(DonutSegm, all = T)

  #---------------smooth and threshold Cytoplasm------------------------------
  FilterCyto = makeBrush(size = 201, shape = "gaussian", sigma = 5)
  Img568smooth = filter2(Img568, filter = FilterCyto)
  sharpen <- function(img){
    FilterHighPass = matrix(-1/40, nrow = 3, ncol = 3)
    FilterHighPass[2, 2] = 1
    img = filter2(img, FilterHighPass)
  }
  Img568sharpen = sharpen(Img568)
  #display(normalize(Img568sharpen))
  Img568sharpenSmooth = filter2(Img568sharpen, filter = makeBrush(size = 201, shape = "gaussian", sigma = 2))
  Img568sharpen = Img568sharpenSmooth
  # display(normalize(Img568smooth))
  cytoThr = thresh(Img568smooth, w = 400, h = 400, offset = 1e-6)
  #display(cytoThr)
  #cytoOpening = opening(cytoThr, kern = makeBrush(1, shape = "disc"))
  #display(cytoOpening)
  ctmask = opening(Img568sharpen > 0.115, makeBrush(5, shape='disc'))
  cytoMask = propagate(Img568, seeds = nucMask, mask = ctmask)
  # cytocol1 = paintObjects(cytoMask, NImgCol568, col = 'orange')
  # cytoMask2 = propagate(Img568, seeds = nucMask, mask = ctmask, lambda = 1e-10)
  # cytocol2 = paintObjects(cytoMask2, NImgCol568, col = 'orange')
  # display(combine(cytocol1, cytocol2), all = T)
  #cytoOpeningNoNuc = cytoOpening - nucOpening
  #CytoMaskNoNuc = propagate(Img568, nucMask, cytoOpeningNoNuc)
  # display(colorLabels(cytoMaskNoNuc))
  # display(colorLabels(cytoMaskNoNuc))
  NImgCol568 = rgbImage(red = NImg568 * 2)
  #display(NImgCol568)
  #NImgCol405_568 = rgbImage(blue = NImg405 * 3, red = NImg568 * 1.5)
  #display(NImgCol405_568)
  #display(Segm, all = T)
  
  Empty = Image(data = array(0, dim = c(1000 ,1000)))
  #display(Empty)
  
  #test with stackObjects & tiling
  # maskTile <- tile(stackObjects(cytoMask, normalize(cytoMask)), nx = 15, lwd = 2)
  # display(maskTile, all = T)
  # maskuntile <- untile(maskTile, nim = c(1,1))
  # display(maskuntile)#, all = T)
  # str(maskTile)
  # maskTile[,,1]
  # for (i in 1:dim(maskTile)[3]) {
  #   maskTile[,,i] <- erode(maskTile[,,i], kern = makeBrush(2, shape = "diamond"))
  # }
  # display(normalize(maskTile[,,1]))
  # display(normalize(erode(maskTile[,,1], kern = makeBrush(2, shape = "diamond"))))
  for (i in 1:(max(cytoMask) + 1)) {
    cytoMaskTemp <- erode(cytoMask == i, kern = makeBrush(2, shape = "diamond"))
    Empty = Empty + cytoMaskTemp
  }
  #display(Empty)
  EmptyInv = 1 - Empty
  #display(EmptyInv)
  cytoMaskBw = ifelse(cytoMask == 0 , 1, 0)
  #display(cytoMaskBw)
  Edges = EmptyInv - cytoMaskBw 
  display(Edges)
  #display(colorLabels(cytoMask))
  ctmaskNoEdges = ctmask - Edges
  display(ctmaskNoEdges)
  cytoMaskNoEdges <- bwlabel(ctmaskNoEdges)
  #display(colorLabels(cytoMaskNoEdges))
  
  F_cytoMaskNoEdges <- as.data.frame(computeFeatures.shape(cytoMaskNoEdges)) %>% rownames_to_column
  ids <- filter(as.data.frame(F_cytoMaskNoEdges), s.area < 50)
 
  # filter out objects with s.area < 20
  cytoMaskNoEdgesFilter <- rmObjects(cytoMaskNoEdges, index = ids)
  # display(colorLabels(cytoMaskNoEdgesFilter))
  # SegmFilter = paintObjects(cytoMaskNoEdgesFilter, NImgCol568, col = 'orange')
  # SegmFilter = paintObjects(nucMask, SegmFilter, col = 'blue')
  # display(SegmFilter, all = T)
  cytoMask <- cytoMaskNoEdgesFilter
  display(Edges)
  
  #python code for watershed postprocessing:
  # def label_mask(mask_img, border_img, seed_ths, threshold, seed_size=8, obj_size=10):
  #   img_copy = np.copy(mask_img)
  # m = img_copy * (1 - border_img)
  # img_copy[m <= seed_ths] = 0
  # img_copy[m > seed_ths] = 1
  # img_copy = img_copy.astype(np.bool)
  # img_copy = remove_small_objects(img_copy, seed_size).astype(np.uint8)
  # mask_img[mask_img <= threshold] = 0
  # mask_img[mask_img > threshold] = 1
  # mask_img = mask_img.astype(np.bool)
  # mask_img = remove_small_objects(mask_img, obj_size).astype(np.uint8)
  # markers = ndimage.label(img_copy, output=np.uint32)[0]
  # labels = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
  # return labels
  
  ## Generate masks for border and boudaries between cells/nuclei: 
  cytoDilate = dilate(cytoMask, kern = makeBrush(11, shape = "disc"))
  #display(combine(colorLabels(cytoMask), colorLabels(cytoDilate)), all = T)
  cytoBorder = cytoDilate - cytoMask
  display(colorLabels(cytoBorder))
  InterfaceBorderCyto = ifelse(cytoMask > 0, 1, 0) - ifelse(cytoBorder > 0, 2, 0)
  display(normalize(ifelse(InterfaceBorderCyto < 0, 1, 0)))
  display(colorLabels(InterfaceBorderCyto))
  #Mask to compute IoU on:
  display(colorLabels(bwlabel(ifelse(InterfaceBorderCyto == 1, 1, 0))))
  
  #hist(InterfaceBorderCyto)
  
  nucDilate = dilate(nucMask, kern = makeBrush(11, shape = "disc"))
  #display(combine(colorLabels(nucMask), colorLabels(nucDilate)), all = T)
  nucBorder = nucDilate - nucMask
  #display(colorLabels(nucBorder))
  InterfaceBorderNuc = ifelse(nucMask > 0, 1, 0) - ifelse(nucBorder > 0, 2, 0)
  display(normalize(InterfaceBorderNuc))
  #display(normalize(ifelse(InterfaceBorderNuc == -1, 1, 0)))
  #hist(InterfaceBorderNuc)
  
  ## Remove small objects
  ## Encode masks genrated without border or boundary pixels
  ## Encode pixels from mask, border, boudary
  
  
  ##Relabel masks:
  #cytoBorder_mask = ifelse(InterfaceBorderCyto < 0, 1, 0)
  #cytoBorder_label = bwlabel(cytoBorderRlb)
  #display(colorLabels(cytoBorder_label))
  
  ##For testing relabeling Borders to individual cells (probably not necessary)
  # cytoDilate2 = cytoDilate
  # cytoDilate2[cytoDilate2] = VoR[cytoDilate2]
  # display(combine(colorLabels(cytoDilate2), colorLabels(VoR)), all = T)
  # 
  # cytoBorder2 = cytoBorder_label
  # cytoBorder2[cytoBorder2] = cytoDilate[cytoBorder2]
  # display(colorLabels(cytoBorder2))
  # 
  # border_label_new <- cytoDilate[cytoBorder_mask]
  # length(border_label_new)
  # border_label_old <- cytoBorder[cytoBorder_mask]
  # length(border_label_old)
  # split(border_label_new, border_label_old) <- tapply(border_label_new, border_label_old, function(x) {
  #   u <- unique(x)
  #   if ( length(u) == 1L ) {
  #     x
  #   }
  #   else {
  #     s <- sapply(u, function(i) sum(x == i))
  #     rep(u[which.max(s)], length(x))
  #   }
  # })
  # cytoBorder_test = cytoBorder
  # cytoBorder_test[cytoBorder_mask] <- border_label_new
  # display(colorLabels(cytoBorder_test))
  # 
  # Dilate_in_Border <- cytoDilate[cytoBorder]
  # Border_in_Border <- cytoBorder
  # display(colorLabels(Border_in_Border))
  # Border_in_Border[cytoBorder] <- Dilate_in_Border
  # display(colorLabels(Border_in_Border))
  # 
  # length(cytoBorderRlb[cytoBorderRlb])
  # length(VoR[cytoBorderRlb])
  # display(colorLabels(cytoBorderNew))
  # display(combine(ifelse(cytoBorderNew == 6, 1, 0), ifelse(cytoDilate == 6, 1, 0), ifelse(VoR == 6, 1, 0)),  all = T)
  # max(BWcytoBorderRlb)
  # display(colorLabels(cytoBorderNew))
  # cytoBorderRlb[cytoDilate]
  # display(colorLabels(cytoBorderRlb))
  # max(cytoBorderRlb)
  # display(combine(ifelse(cytoBorderNew == 6, 1, 0), ifelse(cytoDilate == 6, 1, 0), ifelse(VoR == 6, 1, 0), ifelse(VoR == 3, 1, 0)),  all = T)
  # display(ifelse(cytoDilate == 23, 1, 0))
  # display(colorLabels(cytoDilate))
  
  Segm = paintObjects(cytoMask, NImgCol568, col = 'orange')
  Segm = paintObjects(nucMask, Segm, col = 'blue')
  #display(Segm)
  # #---------------Remove cells at the edge-------------------------------------------------------------------------------
  # #subset for boundary pixels
  # dims = dim(nucMask) 
  # #display(colorLabels(nucMask))
  # border = c(nucMask[1:dims[1],1],
  #            nucMask[1:dims[1],dims[2]],
  #            nucMask[1,1:dims[2]],
  #            nucMask[dims[1],1:dims[2]]
  #            )
  # #extract object identifiers at the boundary
  # ids = unique(border[which(border != 0)])
  # #Filter our ids from Fc or add info to new col.
  # # Fc$Border <- ifelse(Fc[,"rowname"] %in% ids, "Border", "Inner") #if the border objects are of interest and need to be kept
  # Fc = filter(Fc, !rowname %in% ids)
  # 
  # if (nrow(Fc) == 0) {
  #   Fc = data.frame(0, nrow = 0, ncol = 141*12+1)
  # }
  # 
  # write.csv(Fc, paste0("IF005","_p--",Plate,"_w--",Well,"_pos--",Position,".csv"))
  
  # #--------------- Save images and masks ---------------------------------------------------------------------------------
  # writeImage(nucMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--nucMask.tif"))
  # writeImage(cytoMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMask.tif"))
  # #writeImage(cytoMaskNorm, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMaskNorm.tif"))
  # #writeImage(NImg405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg405.tif"))
  # writeImage(Img405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Img405.tif"))
  # #writeImage(NImg568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg568.tif"))
  # writeImage(Img568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Img568.tif"))
  writeImage(Segm, paste0(base_dir, "/Segmentation/", "p_",Plate,"--W_",Well,"--P_",Position,"--Segm.tif"), type = "tiff")
  
  #Encode2df <- function(nucMask, cytoMask){
  #add encoding to df:
  dim(nucMask) <- c(dim(nucMask), 1)
  dim(cytoMask) <- c(dim(cytoMask), 1)
  dim(InterfaceBorderCyto) <- c(dim(InterfaceBorderCyto), 1)
  dim(InterfaceBorderCyto) <- c(dim(InterfaceBorderCyto ), 1)
  #EncodedNuc <- postprocess_image(nucMask, shape = SHAPE)
  #EncodedCyto <- postprocess_image(cytoMask, shape = SHAPE)
  
  n_sample_InterfaceBorderCyto <- as.tibble(train_data[Boolean & train_data$Channel == "mCherry",]) %>%
    add_column(Masks = list(as.array(InterfaceBorderCyto))) %>% # remove this 2 columns in 
    mutate(NucEncodedPixels = map2(Masks, ImageShape, postprocess_image)) %>% 
    select(-Masks, -ImageShape)
  
  n_sample_nuc <- as.tibble(train_data[Boolean & train_data$Channel == "DAPI",]) %>% 
    add_column(nucMasks = list(as.array(nucMask))) %>% # remove this 2 columns in 
    mutate(NucEncodedPixels = map2(nucMasks, ImageShape, postprocess_image)) %>% 
    select(-nucMasks, -ImageShape)
  
  n_sample_cyto <- as.tibble(train_data[Boolean & train_data$Channel == "mCherry",]) %>% 
    add_column(cytoMasks = list(as.array(cytoMask))) %>% # remove those 2 columns in 
    mutate(CytoEncodedPixels = map2(cytoMasks, ImageShape, postprocess_image)) %>% 
    select(-cytoMasks, -ImageShape)
  
  n_sample_out_cyto <- n_sample_cyto %>% 
    unnest(CytoEncodedPixels) %>% 
    mutate(CytoEncodedPixels = as.character(CytoEncodedPixels)) %>% 
    select(ImageId, CytoEncodedPixels)
  
  n_sample_out_nuc <- n_sample_nuc %>% 
    unnest(NucEncodedPixels) %>% 
    mutate(NucEncodedPixels = as.character(NucEncodedPixels)) %>% 
    select(ImageId, NucEncodedPixels)
  
  write_csv(n_sample_out_nuc, paste0(UnetTrain_dir,"Plate_", Plate, "--Well_", Well, "--Position_", Position,"--n_sample_out_nuc_noBoundaries.csv"))
  write_csv(n_sample_out_cyto, paste0(UnetTrain_dir,"Plate_", Plate, "--Well_", Well, "--Position_", Position,"--n_sample_out_cyto_noBoundaries.csv"))
  
  # add encoded information to list
  # enc_list[[paste0("p--",Plate,"--",Well,"--P",Position)]] <- n_sample_out
  
  #}
  #enc_list[[paste0("p--",Plate,"--",Well,"--P",Position)]] <- Encode2df(nucMask, cytoMask)
  
  # #Check if cytomask encodes for labels
  # cytoMaskNorm <- normalize(cytoMask)
  # ftsCyto = computeFeatures.moment(cytoDilate)
  # display(colorLabels(cytoDilate), method="raster")
  # text((ftsCyto[,"m.cx"]), (ftsCyto[,"m.cy"]), labels = rownames(ftsCyto), col="Green", cex=.8)
  # # range(cytoMaskNorm)
  # # hist(cytoMaskNorm)
  # ftsBorder = computeFeatures.moment(cytoBorder)
  # display(colorLabels(cytoBorder), method="raster")
  # text((ftsBorder[,"m.cx"]), (ftsBorder[,"m.cy"]), labels = rownames(ftsBorder), col="Green", cex=.8)
  
}

#----------------Run ImageProcessing() to everyfield---------------------------------------------------------------------------------------------------

# run full script
for (Plate in unique(train_data$Plate)) {
  for(Well in unique(train_data$ID)) {
    for(Position in unique(train_data$Position)) {
      Boolean <- train_data$Plate == Plate & train_data$ID == Well & train_data$Position == Position
      # run image proc and encoding for 1 field of view
      ImageProcessingAndEncoding(Image = train_data, Plate = Plate, Well = Well, Position = Position)
      
      # check iteration
      print(paste0("Plate: ", Plate, "; Well: ", Well, "; Position:", Position, "; Time: ", Sys.time()))
    }
  }
}

# transform then write output if append to list works: enc_list[[paste0("p--",Plate,"--",Well,"--P",Position)]] <- n_sample_out 
# enc_df = do.call(rbind, enc_list)
# str(enc_df)

# read the individual csvs generated and merge them for checking integrity of the output
# list_sample_out <- list.files(UnetTrain_dir, pattern = ".csv")
all_sample_out <- map_df(list.files(UnetTrain_dir, pattern = ".csv", full.names = TRUE), read_csv)#, col_names = FALSE)
str(all_sample_out)
# glimpse(all_sample_out)
