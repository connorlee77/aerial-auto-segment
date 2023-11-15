import numpy as np
from numbers import Number
import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import (create_pairwise_bilateral,
#                               create_pairwise_gaussian)


def create_pairwise_gaussian(sdims, shape, Z=None):
    """
    Util function that create pairwise gaussian potentials. This works for all
    image dimensions.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseGaussian`.
    shape: list or tuple
        The shape the CRF has.
    Z: np.array
        Height values for each pixel in the image.
    """
    # create mesh
    hcord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)

    if Z is not None:
        mesh = np.concatenate((mesh, Z[np.newaxis]), axis=0)
        assert len(sdims) == 3, 'sdims must have length 3 if Z dimension is used'

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s
    return mesh.reshape([len(sdims), -1])


def create_pairwise_bilateral(sdims, schan, img, chdim=-1, Z=None):
    """
    Util function that create pairwise bilateral potentials. This works for
    all image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseBilateral`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseBilateral`.
    schan: list or tuple
        The scaling factors per channel in the image. This is referred to
        `srgb` in `DenseCRF2D.addPairwiseBilateral`.
    img: numpy.array
        The input image.
    chdim: int, optional
        This specifies where the channel dimension is in the image. For
        example `chdim=2` for a RGB image of size (240, 300, 3). If the
        image has no channel dimension (e.g. it has only one channel) use
        `chdim=-1`.

    """
    # Put channel dim in right position
    if chdim == -1:
        # We don't have a channel, add a new axis
        im_feat = img[np.newaxis].astype(np.float32)
    else:
        # Put the channel dim as axis 0, all others stay relatively the same
        im_feat = np.rollaxis(img, chdim).astype(np.float32)

    # scale image features per channel
    # Allow for a single number in `schan` to broadcast across all channels:
    if isinstance(schan, Number):
        im_feat /= schan
    else:
        for i, s in enumerate(schan):
            im_feat[i] /= s

    # create a mesh
    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)
    
    if Z is not None:
        mesh = np.concatenate((mesh, Z[np.newaxis]), axis=0)
        assert len(sdims) == 3, 'sdims must have length 3 if Z dimension is used'

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])


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

    image_features, depth_feature = None, None
    if params['theta_alpha_z'] is not None and params['theta_gamma_z'] is not None:
        # If depth feature is available, it will be the last channel in the input_features
        image_features = input_features[:,:,:-1]
        depth_feature = input_features[:,:,-1].astype(np.float32)
        bilateral_sdims = (params['theta_alpha'], params['theta_alpha'], params['theta_alpha_z'])
        gaussian_sdims = (params['theta_gamma'], params['theta_gamma'], params['theta_gamma_z'])
    else:
        image_features = input_features
        bilateral_sdims = (params['theta_alpha'], params['theta_alpha'])
        gaussian_sdims = (params['theta_gamma'], params['theta_gamma'])

    pairwise_bilateral_energy = create_pairwise_bilateral(
        sdims=bilateral_sdims,
        schan=theta_betas,
        img=image_features,
        chdim=2,
        Z=depth_feature,
    )

    pairwise_gaussian_energy = create_pairwise_gaussian(
        sdims=gaussian_sdims,
        shape=(H, W),
        Z=depth_feature,
    )

    d.addPairwiseEnergy(pairwise_bilateral_energy,
                        compat=params['w1'], kernel=kernel, normalization=params['normalization'])
    d.addPairwiseEnergy(pairwise_gaussian_energy, compat=params['w2'],
                        kernel=kernel, normalization=params['normalization'])

    Q = d.inference(params['inference_steps'])
    map = np.argmax(Q, axis=0).reshape(H, W)
    return map, np.asarray(Q).reshape(C, H, W)
