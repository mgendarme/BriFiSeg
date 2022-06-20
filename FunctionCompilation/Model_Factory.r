## LOAD MODEL ######################################################################################
    if(arc == "unet"){
        if(is.null(enc)){
            unet_levels = 7
            model = build_unet(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                            backbone = ENCODER,
                            nlevels = unet_levels,
                            upsample = 'upsampling', #c("upsampling", "transpose")
                            output_activation = ACTIVATION,
                            output_channels = ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS),
                            dec_filters = c(32, 64, 128, 256, 512, 512, 512, 512)[1:(unet_levels+1)],
                            dropout = c(dropout, 0, 0, 0, 0, 0, 0, 0)[1:(unet_levels+1)]
                            )
        } else {
            model = build_unet(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                            backbone = enc,
                            nlevels = NULL,
                            upsample = 'upsampling', #c("upsampling", "transpose")
                            output_activation = ACTIVATION,
                            output_channels = ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS),
                            dec_filters = c(16, 32, 64, 128, 256, 512),
                            dropout = c(dropout, 0, 0, 0, 0, 0)
                            )
        }
    } else if(arc == "fpn"){
        model = build_fpn(input_shape = c(HEIGHT, WIDTH, CHANNELS),
                            backbone = enc,
                            nlevels = NULL,
                            output_activation = ACTIVATION,
                            output_channels = ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS),
                            decoder_skip = FALSE,
                            dropout = dropout
                            )
    } else if(arc == "psp"){
        model = PSPNet(backbone_name=enc,
                    input_shape=c(HEIGHT, WIDTH, CHANNELS),
                    classes=ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS),
                    activation=ACTIVATION,
                    downsample_factor=redfct, # c(4, 8, 16)
                    psp_dropout=dropout)
    } else if(arc == "deeplab"){
        model = DeepLabV3plus(input_shape=c(HEIGHT, WIDTH, CHANNELS),
                                classes=ifelse(ACTIVATION == "softmax", CLASS + 1, CLASS), 
                                backbone=enc,
                                filters=256L,
                                backbone_trainable=TRUE,
                                depth_training=TRUE,
                                output_stride=redfct, # c(8, 16)
                                final_activation=ACTIVATION,
                                dropout=dropout)
    }