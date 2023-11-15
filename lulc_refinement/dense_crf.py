import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (create_pairwise_bilateral,
                              create_pairwise_gaussian)


def dense_crf(unary_potential, input_features, **params):
    '''
        Dense Conditional Random Field inference on a single image

        ## Parameters:
            unary_potential: 2D probability map (H, W, C) to be used as unary potentials in the CRF
            input_features: base image-like features (H, W, C) to be used for the pairwise potentials in the CRF
            params: dictionary of hyperparameters for the CRF
        ## Returns:
            map: CRF MAP (labels) output (H, W)
            Q: Probabilities (C, H, W)
    '''

    if params['kernel'] == 'full':
        kernel = dcrf.FULL_KERNEL
    elif params['kernel'] == 'diag':
        kernel = dcrf.DIAG_KERNEL
    elif params['kernel'] == 'const':
        kernel = dcrf.CONST_KERNEL

    H, W, C = unary_potential.shape
    dcrf_probs = np.ascontiguousarray(unary_potential.transpose(2, 0, 1))
    U = -np.log(dcrf_probs + 1e-3)
    d = dcrf.DenseCRF2D(W, H, C)  # width, height, nlabels
    U = U.reshape(C, -1)

    d.setUnaryEnergy(U)

    _, _, n_channels = input_features.shape

    # If there is only one theta_beta value, use it for all channels (by passing a numerical type instead of a list)
    theta_betas_list = params['theta_betas']
    theta_betas = theta_betas_list[0] if len(theta_betas_list) == 1 else theta_betas_list

    pairwise_bilateral_energy = create_pairwise_bilateral(
        sdims=(params['theta_alpha'], params['theta_alpha']),
        schan=theta_betas,
        img=input_features,
        chdim=2,
    )

    pairwise_gaussian_energy = create_pairwise_gaussian(
        sdims=(params['theta_gamma'], params['theta_gamma']),
        shape=(H, W),
    )

    d.addPairwiseEnergy(pairwise_bilateral_energy,
                        compat=params['w1'], kernel=kernel, normalization=params['normalization'])
    d.addPairwiseEnergy(pairwise_gaussian_energy, compat=params['w2'],
                        kernel=kernel, normalization=params['normalization'])

    # TODO: Fix this hack. Should be simple removal of first if statement, but test it out.
    # if n_channels == 3:
    #     # Smoothness kernel
    #     d.addPairwiseGaussian(sxy=params['theta_gamma'], compat=params['w2'], kernel=kernel,
    #                         normalization=params['normalization'])
    #     # Appearance kernel
    #     d.addPairwiseBilateral(
    #         sxy=params['theta_alpha'],
    #         srgb=params['theta_beta'],
    #         rgbim=input_features,
    #         compat=params['w1'],
    #         kernel=kernel,
    #         normalization=params['normalization'],
    #     )
    # else:
    #     assert len(params['theta_beta_list']) == n_channels, 'theta_beta_list must have same number of channels as input features'
    #     pairwise_bilateral_energy = create_pairwise_bilateral(
    #         sdims=(params['theta_alpha'], params['theta_alpha']),
    #         schan=params['theta_beta_list'],
    #         img=input_features,
    #         chdim=2,
    #     )

    #     pairwise_gaussian_energy = create_pairwise_gaussian(
    #         sdims=(params['theta_gamma'], params['theta_gamma']),
    #         shape=(H, W),
    #     )

    #     d.addPairwiseEnergy(pairwise_bilateral_energy,
    #                         compat=params['w1'], kernel=kernel, normalization=params['normalization'])
    #     d.addPairwiseEnergy(pairwise_gaussian_energy, compat=params['w2'],
    #                         kernel=kernel, normalization=params['normalization'])

    Q = d.inference(params['inference_steps'])
    map = np.argmax(Q, axis=0).reshape(H, W)
    return map, np.asarray(Q).reshape(C, H, W)
