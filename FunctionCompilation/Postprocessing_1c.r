## Post-processing

# define threschold to apply on probability masks
simple_postprocessing <- function(x, y, y_hat, thresh, tresh_prop, size, class){
  #for testing
  # x = X_test
  # y = Y_test
  # y_hat = Y_hat_test
  # thresh = 0.5
  # tresh_prop = 0.5
  # class = 1
  # size = 200
  # i = 1
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
  # remove small objects  
  # for (j in 1:(max(objectpred_mask) + 1)) {
  #   ifelse(length(objectpred_mask == j) < SIZE, 0, j)
  # }

  objectpred_mask_er = erode(objectpred_op, kern = makeBrush(13, shape = "disc"))
  objectpred_mask_er = bwlabel(objectpred_mask_er)
  objectpred_mask_er_WS = watershed(distmap(objectpred_mask_er), tolerance = 1, ext = 1)
  objectpred_mask_er_WS_prop = propagate(y_hat[i,,,1], objectpred_mask_er_WS, mask = (y_hat[i,,,1] > tresh_prop))
  
  montage_y_yhat = abind(
    paintObjects(y[i,,,1], colorLabels(bwlabel(y[i,,,1])), col = "red", thick = T),
    paintObjects(y[i,,,1], colorLabels(objectpred_mask_er_WS_prop), col = "red", thick = T),
    along = 1
  )

  writeImage(montage_y_yhat, files = paste0(Save_image, "/", "montage_y_yhat_label",i,".png"), quality = 100)

}

for(i in 1:5){
  simple_postprocessing(X_test, Y_test, Y_hat_test, thresh = 0.5, tresh_prop = 0.5, class = CLASS)
}

