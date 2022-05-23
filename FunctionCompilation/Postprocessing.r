## Post-processing

# define threschold to apply on probability masks
simple_postprocessing <- function(x, y, y_hat, thresh, tresh_prop = thresh, size){
  #for testing
  # x = X
  # y = Y
  # y_hat = Y_hat
  # thresh = 0.5
  # size = 200
  # i = 5
  # display(y_hat[1,,,3])
  # SIZE = size
  # THRESH = thresh
  # 
  objectpred <- y_hat[i,,,1] - y_hat[i,,,2] - y_hat[i,,,3]
  objectpred_thr <- objectpred > thresh
  objectpred_op <- opening(objectpred_thr, kern = makeBrush(7, shape = "disc"))
  objectpred_mask <- bwlabel(objectpred_op)
  # display(objectpred_mask)
  display(paintObjects(y[i,,,1], toRGB(objectpred_mask), col = "red", thick = T))
  # remove small objects  
  # for (j in 1:(max(objectpred_mask) + 1)) {
  #   ifelse(length(objectpred_mask == j) < SIZE, 0, j)
  # }
  
  objectpred_mask_WS <- watershed(distmap(objectpred_mask), tolerance = 1, ext = 1)
  objectpred_mask_WS_prop <- propagate(y_hat[i,,,1], objectpred_mask_WS, mask = y_hat[i,,,1] > tresh_prop)
  
  # segm1 <- paintObjects(objectpred_mask_WS_prop,
  #                       rgbImage(green = x[i,,,]),
  #                       col = "yellow")
  # 
  # segm2 <- rgbImage(green = x[i,,,],
  #                   red = y[i,,,2] + y[i,,,3])
  # 
  # segm3 <- paintObjects(objectpred_mask_WS_prop,
  #                       rgbImage(green = x[i,,,],
  #                                red = y[i,,,2] + y[i,,,3]),
  #                       col = "yellow",
  #                       thick = T)
  
  segm4 <- paintObjects(y[i,,,1], colorLabels(objectpred_mask_WS), col = "red", thick = T)
  # display(segm4)
  # display(colorLabels(objectpred_mask_WS))
  # display(colorLabels(objectpred_mask_WS_prop))
  # display(segm3)
  # display(abind(segm1, segm2, rgbImage(blue =  Y_hat[i,,,2] + Y_hat[i,,,3]), along = 1))
  
  montage_segm <- abind(normalize(toRGB(x[i,,,])),
                        segm4,
                        along = 1)
    # display(montage_segm)
    
  montage_segm_yhat <- abind(rgbImage(blue = y_hat[i,,,1],
                                 green = y_hat[i,,,2],
                                 red = y_hat[i,,,3]),
                          segm4,
                          along = 1)
    # display(montage_segm)  
  
  writeImage(montage_segm, files = paste0(Save_dir_rep, "/", "montage_segm_postproc",i,".tif"),
             quality = 100, type = "tiff")
  writeImage(montage_segm_yhat, files = paste0(Save_dir_rep, "/", "montage_segm_yhat_postproc",i,".tif"),
             quality = 100, type = "tiff")
  writeImage(segm4, files = paste0(Save_dir_rep, "/", "merge_Y_segm_postproc_overlay",i,".tif"),
             quality = 100, type = "tiff")
}