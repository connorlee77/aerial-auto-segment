import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf(probability_img, base_img, **params):
    H, W, C = probability_img.shape
    dcrf_probs = np.ascontiguousarray(probability_img.transpose(2, 0, 1))
    U = -np.log(dcrf_probs + 1e-3)
    d = dcrf.DenseCRF2D(W, H, C)  # width, height, nlabels
    U = U.reshape(C,-1) 

    
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=params['kernel'],
                          normalization=params['normalization'])

    d.addPairwiseBilateral(
        sxy=params['sxy'], 
        srgb=params['srgb'], 
        rgbim=base_img, 
        compat=params['compat'], 
        kernel=params['kernel'], 
        normalization=params['normalization'],
    ) 

    Q = d.inference(params['inference_steps'])
    map = np.argmax(Q, axis=0).reshape(H, W)
    return map