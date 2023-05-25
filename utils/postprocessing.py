import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf(probability_img, base_img, **params):
    H, W, C = probability_img.shape
    dcrf_probs = np.ascontiguousarray(probability_img.transpose(2, 0, 1))
    U = -np.log(dcrf_probs + 1e-3)
    d = dcrf.DenseCRF2D(W, H, C)  # width, height, nlabels
    U = U.reshape(C,-1) 

    
    d.setUnaryEnergy(U)
    d.addPairwiseBilateral(
        sxy=params['sxy'], 
        srgb=params['srgb'], 
        rgbim=base_img, 
        compat=params['compat'], 
        kernel=params['kernel'], 
        normalization=params['normalization'],
    ) 

    Q = d.inference(params['inference_steps'])
    map = np.argmax(Q, axis=0).reshape(512, 640)
    return map