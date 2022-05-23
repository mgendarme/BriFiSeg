## Post-processing

# define threschold to apply on probability masks
simple_postprocessing <- function(x, y, y_hat, thresh, tresh_prop, size, class){
  #for testing
  x = X_test
  y = Y_test
  y_hat = Y_hat_test
  thresh = 0.5
  tresh_prop = 0.5
  class = 1
  # size = 200
  # i = 1
  # display(y_hat[1,,,3])
  # SIZE = size
  # THRESH = thresh
  # 
  # objectpred <- y_hat[i,,,1] - y_hat[i,,,2] - y_hat[i,,,3]
  if(class == 1){
    objectpred_thr <- y_hat[i,,,1] > thresh
  } else if(class == 2){
    objectpred_thr <- (y_hat[i,,,1] > thresh) - (y_hat[i,,,2] > thresh)
  } else {
    objectpred_thr <- (y_hat[i,,,1] > thresh) - (y_hat[i,,,2] > thresh) - (y_hat[i,,,3] > thresh)
  }

  objectpred_thr = ifelse(objectpred_thr < 0, 0, objectpred_thr)
  
  objectpred_op <- opening(objectpred_thr, kern = makeBrush(7, shape = "disc"))
  # display(objectpred_op)
  # display(y_hat[i,,,1])
  # display(paintObjects(y[i,,,1], toRGB(objectpred_mask), col = "red", thick = T))
  # remove small objects  
  # for (j in 1:(max(objectpred_mask) + 1)) {
  #   ifelse(length(objectpred_mask == j) < SIZE, 0, j)
  # }
  
  
  # display(colorLabels(objectpred_mask_WS))
  objectpred_mask_er = erode(objectpred_op, kern = makeBrush(13, shape = "disc"))
  # display(objectpred_mask_er)
  # writeImage(objectpred_mask_er, files = "/home/gendarme/Desktop/Img_presentation/objectpred_mask_er.png")
  objectpred_mask_er = bwlabel(objectpred_mask_er)
  objectpred_mask_er_WS = watershed(distmap(objectpred_mask_er), tolerance = 1, ext = 1)
  # writeImage(normalize(distmap(objectpred_mask_er)), files = "/home/gendarme/Desktop/Img_presentation/objectpred_mask_er_distmap.png")
  # writeImage(objectpred_thr, files = "/home/gendarme/Desktop/Img_presentation/objectpred_mask_thr.png")
  # writeImage(colorLabels(objectpred_mask_er_WS), files = "/home/gendarme/Desktop/Img_presentation/objectpred_mask_er_WS.png")
  # display(normalize(distmap(objectpred_mask_er)))
  # str(y_hat[1,,,1])
  # str(EBImage::as.Image(objectpred_thr))
  # maskcol = colorLabels(objectpred_mask_er_WS)
  # probcol = toRGB(y_hat[1,,,1] > 0.5)
  # str(probcol)
  # str(maskcol)
  # hist(maskcol)
  # str(probcol)
  # probcoldonut = probcol - ifelse(maskcol > 0, 1, 0)
  # sumcol = maskcol
  # sumcol[,,1] = sumcol[,,1] + probcoldonut[,,1]
  # sumcol[,,2] = sumcol[,,2] + probcoldonut[,,2]
  # sumcol[,,3] = sumcol[,,3] + probcoldonut[,,3]
  # display(sumcol)
  # writeImage(sumcol, files = "/home/gendarme/Desktop/Img_presentation/objectpred_mask_er_WS_propagate.png")
  # writeImage(colorLabels(objectpred_mask_er_WS_prop),
  #  files = "/home/gendarme/Desktop/Img_presentation/objectpred_mask_instance.png")
  # # display(colorLabels(objectpred_mask_er_WS))
  # display(normalize(distmap(objectpred_mask_er)))
  objectpred_mask_er_WS_prop = propagate(y_hat[i,,,1], objectpred_mask_er_WS, mask = (y_hat[i,,,1] > tresh_prop))
  # display(colorLabels(objectpred_mask_er_WS_prop))
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
  
  # segm4 <- paintObjects(y[i,,,1], colorLabels(objectpred_mask_WS), col = "red", thick = T)
  # display(segm4)
  # display(colorLabels(objectpred_mask_WS))
  # display(colorLabels(objectpred_mask_WS_prop))
  # display(segm3)
  # display(abind(segm1, segm2, rgbImage(blue =  Y_hat[i,,,2] + Y_hat[i,,,3]), along = 1))
  
  # montage_segm <- abind(normalize(toRGB(x[i,,,1])),
  #                       segm4,
  #                       along = 1)
  #    display(montage_segm)
    
  # montage_segm_yhat <- abind(rgbImage(blue = y_hat[i,,,1]#,
  #                               #  green = y_hat[i,,,2],
  #                               #  red = y_hat[i,,,3]
  #                               ),
  #                         segm4,
  #                         along = 1)
  #    display(montage_segm)  
  
  # max(bwlabel(y[i,,,1]))
  # max(bwlabel(objectpred_mask_WS))
  # for(i in 1:max(bwlabel(objectpred_mask_WS))){
    # imgnum = 4
    # display(abind(bwlabel(y[i,,,1]) == imgnum,
    #              objectpred_mask_WS == imgnum,
    #               along = 1
    # ))
  # }
  # display(colorLabels(objectpred_mask_WS))
  # max(objectpred_mask_WS)
  
  montage_y_yhat = abind(
    paintObjects(y[i,,,1], colorLabels(bwlabel(y[i,,,1])), col = "red", thick = T),
    paintObjects(y[i,,,1], colorLabels(objectpred_mask_er_WS_prop), col = "red", thick = T),
    along = 1
  )
  #  display(montage_y_yhat)

  writeImage(montage_y_yhat, files = paste0(Save_image, "/", "montage_y_yhat_label",i,".png"), quality = 100)
  # writeImage(montage_y_yhat, files = paste0("/home/gendarme/Desktop", "/", "montage_y_yhat_label",i,".png"), quality = 100)
  # writeImage(montage_segm_yhat, files = paste0(Save_image, "/", "montage_segm_yhat_postproc",i,".tif"),
  #            quality = 100, type = "tiff")
  # writeImage(segm4, files = paste0(Save_image, "/", "merge_Y_segm_postproc_overlay",i,".tif"),
  #            quality = 100, type = "tiff")
}

for(i in 1:5){
  simple_postprocessing(X_test, Y_test, Y_hat_test, thresh = 0.5, tresh_prop = 0.5, class = CLASS)
}

