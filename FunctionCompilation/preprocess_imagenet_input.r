## Build based on link below
## https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

preprocess_imagenet_input = function(x, mode){
    # """Preprocesses a Numpy array encoding a batch of images.
    # # Arguments
    #     x: Input array, 3D or 4D.
    #     data_format: Data format of the image array.
    #     mode: One of "caffe", "tf" or "torch".
    #         - caffe: will convert the images from RGB to BGR,
    #             then will zero-center each color channel with
    #             respect to the ImageNet dataset,
    #             without scaling.
    #         - tf: will scale pixels between -1 and 1,
    #             sample-wise.
    #         - torch: will scale pixels between 0 and 1 and then
    #             will normalize each channel with respect to the
    #             ImageNet dataset.
    # # Returns
    #     Preprocessed Numpy array.
    # """
    # backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    # if not issubclass(x.dtype.type, np.floating):
    #     x = x.astype(backend.floatx(), copy=False)

    if(mode == 'tf'){
        x = x / 127.5
        x = x - 1.0
        return(x)
    } else if(mode == 'torch'){
        x = x/255.0
        mean = c(0.485, 0.456, 0.406)
        std = c(0.229, 0.224, 0.225)
    } else {
        # 'RGB'->'BGR' 
        if(length(dim(x)) == 4){
            x = abind(x[,,,3], x[,,,2], x[,,,1], along = 3)
        } else {
            x = abind(x[,,3], x[,,2], x[,,1], along = 3)
        }
        mean = c(103.939, 116.779, 123.68)
        std = NULL
    }
    # Zero-center by mean pixel
    if(length(dim(x)) == 4){
        x[,,, 1] = x[,,, 1] - mean[1]
        x[,,, 2] = x[,,, 2] - mean[2]
        x[,,, 3] = x[,,, 3] - mean[3]
        if(!is.null(std)){
            x[,,, 1] = x[,,, 1] / std[1]
            x[,,, 2] = x[,,, 2] / std[2]
            x[,,, 3] = x[,,, 3] / std[3]
        }
    } else {
        x[,, 1] = x[,, 1] - mean[1]
        x[,, 2] = x[,, 2] - mean[2]
        x[,, 3] = x[,, 3] - mean[3]
        if(!is.null(std)){
            x[,, 1] = x[,, 1] / std[1]
            x[,, 2] = x[,, 2] / std[2]
            x[,, 3] = x[,, 3] / std[3]
        }
    }
    
    return(x)

}