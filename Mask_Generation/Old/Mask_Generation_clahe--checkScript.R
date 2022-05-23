WD <- as.character('V:/Team_LabNotebooks/PMI/Gendarme/MG012_IF012/IF012')
setwd(WD)
library(EBImage)
library(tidyverse)

# Settings#############################################################################################################################################
UnetTrain_dir <- 'V:/Team_LabNotebooks/PMI/Gendarme/U-Net/PMI_Data/UnetTrain/Normalize_Clahe/'

# List of files########################################################################################################################################
AllImages <- as.data.frame(list.files(getwd(), recursive = T), col.names = 'FileNames')
colnames(AllImages)[1] <- 'FileNames'
AllImages <- as.data.frame(grep('.tif', AllImages$FileNames, value=TRUE))
colnames(AllImages)[1] <- 'FileNames'

# Extract Metainformation------------------------------------------------------------------------------------------------------------------------------
MetaInformation <- function(Files, Path){
  Files$Directory <- Path
  Files$Plate <- substring(Files$FileNames,12,14) ## Plate
  Files$Plate <- as.numeric(Files$Plate)
  Files$Stack <- sub(".*data/", "", Files$FileNames) ## Stack
  Files$DataStack <- substring(Files$Stack,0,35) ## DataStack
  Files$ID <- sub("--W.*", "", Files$Stack) ## ID
  Files$Well <- sub(".*--W", "", Files$Stack) ## W #
  Files$Well <- substring(Files$Well,0,5) ## W #
  Files$Position <- sub(".*--P", "", Files$Stack) ## Position
  Files$Position <- substring(Files$Position,0,5) ## Position
  Files$Time <- sub(".*--T", "", Files$Stack) ## Time
  Files$Time <- substring(Files$Time,0,5) ## Time
  Files$Channel <- sub(".*--", "", Files$Stack) ## Channel
  Files$Channel <- sub(".tif.*", "", Files$Channel) ## Channel
  return(as.data.frame(Files))
}
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
# Add metainformation (if necessary, more revelant for data analysis)----------------------------------------------------------------------------------
# library(xlsx)
# MetaInfo <- read.xlsx("IF005_Plate_Setup.xlsx", sheetIndex = 1)
# AllImages <- merge(x = AllImages, y = MetaInfo, by = c("ID","Plate"), all.x = TRUE)


# Read images and extract features#####################################################################################################################
ImageProcessing <- function(Image, Plate, Well, Position){
  # ##For testing:
  # Image = AllImages
  # Plate = 1
  # Well = "C1"
  # Position = "00001"
  #---------------Rescale pixel intensity----------
  Bottom = 2^15/(2^16-1) 
  Top = (2^15+4095)/(2^16-1)
  Rescale <- function(x){
    (x - Bottom) / (Top - Bottom)   
  }
  
  #---------------Load Images------------------------
  #Position = "00001" #for testing
  Boolean <- AllImages$Plate == Plate & AllImages$ID == Well & AllImages$Position == Position
  Img405 = readImage(paste0(AllImages[AllImages$Channel == "DAPI" & Boolean, "Directory"],
                            "/",
                            AllImages[AllImages$Channel == "DAPI" & Boolean, "FileNames"]))
  Img405 = Rescale(Img405)
  NImg405 = normalize(clahe(Img405))#, inputRange = c(range(Img405)[1], range(Img405)[2]))
  #display(NImg405)
  #writeImage(Img405, "Img405.tif")
  Img488 = readImage(paste0(AllImages[AllImages$Channel == "GFP" & Boolean, "Directory"],
                            "/",
                            AllImages[AllImages$Channel == "GFP" & Boolean, "FileNames"]))
  Img488 = Rescale(Img488)
  NImg488 = normalize(clahe(Img488))
  #display(NImg488)
  #writeImage(Img488, "Img488.tif")
  Img568 = readImage(paste0(AllImages[AllImages$Channel == "mCherry" & Boolean, "Directory"], 
                            "/",
                            AllImages[AllImages$Channel == "mCherry" & Boolean, "FileNames"]))
  Img568 = Rescale(Img568)
  NImg568 = normalize(clahe(Img568))#, inputRange = c(range(Img568)[1], range(Img568)[2]))
  #display(NImg568)
  #writeImage(Img568, "Img568.tif")
  Img647 = readImage(paste0(AllImages[AllImages$Channel == "647" & Boolean, "Directory"], 
                            "/",
                            AllImages[AllImages$Channel == "647" & Boolean, "FileNames"]))
  Img647 = Rescale(Img647)
  NImg647 = normalize(clahe(Img647))#, inputRange = c(range(Img647)[1], range(Img647)[2]))
  #display(NImg647)
  #writeImage(Img647, "Img647.tif")
  #---------------smooth and threshold nuleus------------------------------
  FilterNuc = makeBrush(size = 51, shape = "gaussian", sigma = 2)
  #Img405clahe = clahe(Img405)
  #display(normalize(Img405clahe))
  Img405smooth = filter2(Img405, filter = FilterNuc)
  #display(normalize(Img405smooth))
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
  #display(colorLabels(nucMask), all=TRUE)
  NImgCol405 = rgbImage(blue = NImg405*3)
  #display(NImgCol405)
  nucSegm = paintObjects(nucMask, NImgCol405, col = 'red')
  display(nucSegm, all = T)

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
  Img568clahe = clahe(Img568)
  #display(normalize(Img568clahe))
  #Img568sharpen = Img568clahe
  Img568sharpenSmooth = filter2(Img568sharpen, filter = makeBrush(size = 201, shape = "gaussian", sigma = 3))
  Img568sharpen = Img568sharpenSmooth
  # display(normalize(Img568smooth))
  cytoThr = thresh(Img568smooth, w = 400, h = 400, offset = 0.0000001)
  #display(cytoThr)
  #cytoOpening = opening(cytoThr, kern = makeBrush(1, shape = "disc"))
  #display(cytoOpening)
  ctmask = opening(Img568sharpen > 0.125, makeBrush(5, shape='disc'))
  #display(ctmask)
  cytoMask = propagate(Img568, seeds = nucMask, mask = ctmask)
  #display(colorLabels(cytoMask))
  #cytoOpeningNoNuc = cytoOpening - nucOpening
  #CytoMaskNoNuc = propagate(Img568, nucMask, cytoOpeningNoNuc)
  # display(colorLabels(cytoMaskNoNuc))
  # display(colorLabels(cytoMaskNoNuc))
  #NImgCol568 = rgbImage(red = NImg568)
  #display(NImgCol568)
  NImgCol405_568 = rgbImage(blue = NImg405 * 3, red = normalize(Img568clahe) * 1.5)
  #display(NImgCol405_568)
  Segm = paintObjects(cytoMask, NImgCol405_568, col = 'orange')
  Segm = paintObjects(nucMask, Segm, col = 'red')
  display(Segm, all = T)

  # #---------------smooth and threshold Specs----------------------------------
  ascThr = Img488 > sum(range(Img488))/6
  #ascThr = thresh(Img488, w = 200, h = 200, offset = 0.05)
  ascOpening = opening(ascThr, makeBrush(5, shape = 'disc'))
  ascLbl = bwlabel(ascOpening)
  display(colorLabels(ascLbl))
  FilterSpec = makeBrush(size = 51, shape = "gaussian", sigma = 2)
  Img488smooth = filter2(Img488, filter = FilterSpec)
  ascThrSmooth = Img488smooth > sum(range(Img488smooth))/6
  ascOpening = opening(ascThrSmooth, makeBrush(5, shape = 'disc'))
  ascLbl = bwlabel(ascOpening)
  NImgCol488 = rgbImage(green = NImg488)
  NImgCol488 = rgbImage(green = Img488)
  # display(NImgCol488)
  NImgCol405_488 = rgbImage(green = NImg488 * 1.5, red = NImg568 * 1.2, blue = NImg405 * 2)
  SegmSpec = paintObjects(ascLbl, NImgCol405_488, col = 'yellow', thick = T)
  SegmSpec = paintObjects(nucMask, SegmSpec, col = 'orange', thick = T)
  SegmSpec = paintObjects(cytoMask, SegmSpec, col = 'grey', thick = T)
  display(SegmSpec) #, method = "raster")
  text(fts[,"m.cx"], fts[,"m.cy"], labels = seq_len(nrow(fts)), col="white", cex=.8)
  writeImage(SegmSpec, files = "SegmentationComplete_Example.tiff", quality = 100, type = "tiff")

  # #------------------specs need to be mapped back to their belonging cell--------
  # v_labels <- VoR[ascOpening]
  # s_labels <- ascLbl[ascOpening]
  # 
  # split(v_labels, s_labels) <- tapply(v_labels, s_labels, function(x) {
  #   u <- unique(x)
  #   if ( length(u) == 1L ) {
  #     x
  #   }
  #   else {
  #     s <- sapply(u, function(i) sum(x == i))
  #     rep(u[which.max(s)], length(x))
  #   }
  # })
  # 
  # LabelDict <- setNames(data.frame(matrix(ncol = 2, nrow = length(v_labels))), c("VoR", "Speck"))
  # LabelDict$VoR <- v_labels
  # LabelDict$Speck <- s_labels 
  # LabelDict <- LabelDict %>% group_by(VoR, Speck) %>% summarize(SpeckCount = n_distinct(Speck)) %>% arrange(Speck)
  # #LabelDict

  # ftsSpec = computeFeatures.moment(ascLbl)
  # ftsNuc = computeFeatures.moment(VoR)
  # NImgCol405_488 = rgbImage(green = NImg488 * 2, blue = NImg405 * 1.5)
  # display(NImgCol405_488, method="raster")
  # text((ftsSpec[,"m.cx"]), (ftsSpec[,"m.cy"]), labels = rownames(ftsSpec), col="Green", cex=.8)
  # text(ftsNuc[,"m.cx"], ftsNuc[,"m.cy"], labels = rownames(ftsNuc), col="LightBlue", cex=.8)

  # NImgCol405_488 = rgbImage(green = NImg488 * 4)#, blue = NImg405 * 1.5) # red = NImg568 * 1.2, 
  # SegmSpec = paintObjects(ascLblNew, NImgCol405_488, col = 'red', thick = T)
  # SegmSpec = paintObjects(nucMask, SegmSpec, col = 'grey')
  # SegmSpec = paintObjects(VoR, SegmSpec, col = 'purple')
  # display(SegmSpec)
  # writeImage(SegmSpec, "SegmSpec_ATP5_LPS10_1h_2.tif", quality = 100)

  # #---------------Features extraction-------------------------------------------------------------------------
  # F405_Nuc = computeFeatures(nucMask, Img405, xname = "Nuc", refnames = "405", haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # 
  # F488_Nuc = computeFeatures(nucMask, Img488, xname = "Nuc", refnames = "488", haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F488_Donut = computeFeatures(Donut, Img488, xname = "Donut", refnames = "488", haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F488_BubbleBound = computeFeatures(BubbleBound, Img488, xname = "Bubble", refnames = "488", haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F488_CytoNoNuc = computeFeatures(CytoMaskNoNuc, Img488, xname = "CytoNoNuc", refnames = "488",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F488_Cyto = computeFeatures(cytoMask, Img488, xname = "Cyto", refnames = "488",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # 
  # F568_Nuc = computeFeatures(nucMask, Img568, xname = "Nuc", refnames = "568",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F568_Donut = computeFeatures(Donut, Img568, xname = "Donut", refnames = "568",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F568_Bubble = computeFeatures(BubbleBound, Img568, xname = "Donut", refnames = "568",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F568_CytoNoNuc = computeFeatures(CytoMaskNoNuc, Img568, xname = "CytoNoNuc", refnames = "568",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # F568_Cyto = computeFeatures(cytoMask, Img568, xname = "Cyto", refnames = "568",haralick.nbins = 32, haralick.scales = c(1,2,4,8))
  # 
  # F488_Spec = computeFeatures(ascLbl, Img488, xname = "ASC", refnames = "488", haralick.nbins = 32, haralick.scales = c(1,2,4,8))# %>%
  # F488_Spec = cbind(F488_Spec, LabelDict$VoR) %>% as.data.frame(F488_Spec) 
  # colnames(F488_Spec)[length(colnames(F488_Spec))] = "VoRLabel"
  # F488_Spec$VoRLabel = as.character(F488_Spec$VoRLabel)
  # 
  # Fc = cbind(F405_Nuc, 
  #            #F488_Spec, 
  #            F488_Nuc,
  #            F488_Donut, 
  #            F488_BubbleBound, 
  #            F488_CytoNoNuc,
  #            F488_Cyto,
  #            F568_Nuc, 
  #            F568_Donut, 
  #            F568_Bubble, 
  #            F568_CytoNoNuc, 
  #            F568_Cyto) 
  # Fc = as.data.frame(Fc) %>% rownames_to_column 
  # Fc = full_join(Fc, F488_Spec, by = c("rowname" = "VoRLabel"))

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
  
  #---------------Save images and masks
  writeImage(nucMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--nucMask.tif"))
  writeImage(cytoMask, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--cytoMask.tif"))
  writeImage(NImg405, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg405.tif"))
  writeImage(NImg568, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--NImg568.tif"))
  writeImage(Segm, paste0(UnetTrain_dir, "p--",Plate,"--",Well,"--P",Position,"--Segm.tif"))
}
# add a rule remove very small nuclei detected

#----------------Run ImageProcessing() to everyfield---------------------------------------------------------------------------------------------------
AllImages <- MetaInformation(Files = AllImages, Path = WD)
AllImages <- IdToMap(AllImages)

## Small sample size for testing
AllImagesSmall <- AllImages[AllImages$cN == 1,] 
AllImages <- AllImagesSmall # Small sample size for testing

for (Plate in unique(AllImages$Plate)) {
  for(Well in unique(AllImages$ID)) {
    for(Position in unique(AllImages$Position)) {
      ImageProcessing(Image = AllImages, Plate = Plate, Well = Well, Position = Position)
    }
  }
}


