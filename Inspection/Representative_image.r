
## saving some representative images with transformation
RANGE_IMAGES = 5

for(j in ifelse(ACTIVATION == "softmax", 2, 1):ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS)){
    for(i in 1:RANGE_IMAGES){

        J = ifelse(ACTIVATION == "softmax", j-1, j)

        bf_gt_val = paintObjects(Y_val[i,,,j],
                                combine_col(image_1 = normalize(X_val[i,,,ifelse(is.null(enc), 1, j)]),
                                            image_2 = Y_hat_val[i,,,j]/3,
                                            color_1 = "grey",
                                            color_2 = "green",
                                            dimension = c(HEIGHT, WIDTH)),
                                col = "red")
        writeImage(bf_gt_val, files = paste0(Save_image_semantic, "/", "bf_gt_val_class_",J,"_",i,".tif"),
                    quality = 100, type = "tiff")
        # display(bf_gt_val)
        # dapi_gt_val = paintObjects(Y_val[i,,,1],
        #                          combine_col(image_1 = normalize(X_dapi_val[i,,]),
        #                                      image_2 = Y_hat_val[i,,,1]/3,
        #                                      color_1 = "grey",
        #                                      color_2 = "green",
        #                                      dimension = c(HEIGHT, WIDTH)),
        #                          col = "red")
        # # display(dapi_gt_val)
        # writeImage(dapi_gt_val, files = paste0(Save_image_semantic, "/", "dapi_gt_val",i,".tif"),
        #             quality = 100, type = "tiff")

        bf_gt_test = paintObjects(Y_test[i,,,1],
                                combine_col(image_1 = normalize(X_test[i,,,ifelse(is.null(enc), 1, j)]),
                                            image_2 = Y_hat_test[i,,,j]/3,
                                            color_1 = "grey",
                                            color_2 = "green",
                                            dimension = c(HEIGHT, WIDTH)),
                                col = "red")
        # display(bf_gt_test)
        writeImage(bf_gt_test, files = paste0(Save_image_semantic, "/", "bf_gt_test_class_",J,"_",i,".tif"),
                    quality = 100, type = "tiff")

        # dapi_gt_test = paintObjects(Y_test[i,,,1],
        #                          combine_col(image_1 = normalize(X_dapi_test[i,,]),
        #                                      image_2 = Y_hat_test[i,,,1]/3,
        #                                      color_1 = "grey",
        #                                      color_2 = "green",
        #                                      dimension = c(HEIGHT, WIDTH)),
        #                          col = "red")
        # # display(dapi_gt_test)
        # writeImage(dapi_gt_test, files = paste0(Save_image_semantic, "/", "dapi_gt_test",i,".tif"),
        #             quality = 100, type = "tiff")

        vir_hat_test = img_to_viridis(Y_hat_test[i,,,j])
        # display(vir_hat_test)
        writeImage(vir_hat_test, files = paste0(Save_image_semantic, "/", "vir_hat_test_class_",J,"_",i,".tif"),
                    quality = 100, type = "tiff")

        vir_hat_val = img_to_viridis(Y_hat_val[i,,,j])
        # display(vir_hat_val)
        writeImage(vir_hat_val, files = paste0(Save_image_semantic, "/", "vir_hat_val_class_",J,"_",i,".tif"),
                    quality = 100, type = "tiff")

        vir_gt_test = img_to_viridis(Y_test[i,,,ifelse(is.null(enc), 1, j)])
        # display(vir_gt_test)
        writeImage(vir_gt_test, files = paste0(Save_image_semantic, "/", "vir_gt_test_class_",J,"_",i,".tif"),
                    quality = 100, type = "tiff")

        vir_gt_val = img_to_viridis(Y_val[i,,,ifelse(is.null(enc), 1, j)])
        # display(vir_gt_val)
        writeImage(vir_gt_val, files = paste0(Save_image_semantic, "/", "vir_gt_val_class_",J,"_",i,".tif"),
                    quality = 100, type = "tiff")

        vir_hat_gt_val = paintObjects(Y_val[i,,,j],
                            # rgbImage(blue = Y_hat_val_ab_p5[i,,]),
                            vir_hat_val,
                            col = "red",
                            thick = T)
        display(vir_hat_gt_val)
        writeImage(vir_hat_gt_val, files = paste0(Save_image_semantic, "/", "vir_hat_gt_val_class_",J,"_",i,".tif"),
                quality = 100, type = "tiff")

        vir_hat_gt_test = paintObjects(Y_test[i,,,ifelse(is.null(enc), 1, j)],
                            # rgbImage(blue = Y_hat_val_ab_p5[i,,]),
                            vir_hat_test,
                            col = "red",
                            thick = T)
        # display(vir_hat_gt_test)
        writeImage(vir_hat_gt_test, files = paste0(Save_image_semantic, "/", "vir_hat_gt_test_class_",J,"_",i,".tif"),
                quality = 100, type = "tiff")

        # gt_pred_bf_dapi_test = abind(vir_gt_test,
        #                             vir_hat_test,
        #                             combine_col(image_1 = normalize(X_test[i,,,1]),
        #                                         color_1 = "grey",
        #                                         dimension = c(HEIGHT, WIDTH)),
        #                             combine_col(image_1 = normalize(X_dapi_test[i,,]),
        #                                         color_1 = "grey",
        #                                         dimension = c(HEIGHT, WIDTH)),
        #                             along = 1)
        # display(gt_pred_bf_dapi_test)
        # writeImage(gt_pred_bf_dapi_test, files = paste0(Save_image_semantic, "/", "gt_pred_bf_dapi_test",i,".tif"),
        #            quality = 100, type = "tiff")

        # gt_pred_bf_dapi_val = abind(vir_gt_val,
        #                             vir_hat_val,
        #                             combine_col(image_1 = normalize(X_val[i,,,1]),
        #                                         color_1 = "grey",
        #                                         dimension = c(HEIGHT, WIDTH)),
        #                             combine_col(image_1 = normalize(X_dapi_val[i,,]),
        #                                         color_1 = "grey",
        #                                         dimension = c(HEIGHT, WIDTH)),
        #                             along = 1)
        # display(gt_pred_bf_dapi_val)
        # writeImage(gt_pred_bf_dapi_val, files = paste0(Save_image_semantic, "/", "gt_pred_bf_dapi_val",i,".tif"),
        #            quality = 100, type = "tiff")
        gt_pred_bf_test = abind(vir_gt_test,
                                    vir_hat_test,
                                    combine_col(image_1 = normalize(X_test[i,,,ifelse(is.null(enc), 1, j)]),
                                                color_1 = "grey",
                                                dimension = c(HEIGHT, WIDTH)),
                                    # combine_col(image_1 = normalize(X_dapi_test[i,,]),
                                    #             color_1 = "grey",
                                    #             dimension = c(HEIGHT, WIDTH)),
                                    along = 1)
        # display(gt_pred_bf_test)
        writeImage(gt_pred_bf_test, files = paste0(Save_image_semantic, "/", "gt_pred_bf_test_class_",J,"_",i,".tif"),
                quality = 100, type = "tiff")

        gt_pred_bf_val = abind(vir_gt_val,
                                    vir_hat_val,
                                    combine_col(image_1 = normalize(X_val[i,,,ifelse(is.null(enc), 1, j)]),
                                                color_1 = "grey",
                                                dimension = c(HEIGHT, WIDTH)),
                                    # combine_col(image_1 = normalize(X_dapi_val[i,,]),
                                    #             color_1 = "grey",
                                    #             dimension = c(HEIGHT, WIDTH)),
                                    along = 1)
        # display(gt_pred_bf_val, bg = "black")
        writeImage(gt_pred_bf_val, files = paste0(Save_image_semantic, "/", "gt_pred_bf_val_class_",J,"_",i,".tif"),
                quality = 100, type = "tiff")
    }

    # for(i in 1:2){
    #     ch = i
    #     writeImage(img_to_viridis(Y_val[4,,,ch]),
    #             files = paste0("~/Desktop/", "Y_val_channnel_",ch,".tif"),
    #                 quality = 100, type = "tiff")
    # }
}
