import numpy as np
from numbers import Number
import tqdm
import cv2
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
    dcrf_probs = np.clip(np.ascontiguousarray(unary_potential.transpose(2, 0, 1)), 0, 1)

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
        image_features = input_features[:, :, :-1]
        depth_feature = input_features[:, :, -1].astype(np.float32)
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
    map_score = np.argmax(Q, axis=0).reshape(H, W)
    return map_score, np.asarray(Q).reshape(C, H, W)


def tiled_dense_crf(unary_potential, input_features, patch_size, stride, **params):
    '''
        Performs tiled inference on the input image.

        ## Parameters
            unary_potential: np.ndarray (H, W, n_classes)
                The unary potential.
            input_features: np.ndarray (H, W, n_features)
                The input features.
            patch_size: int
                The size of the patch.
            stride: int
                The stride of the patch.
            params: dict
                The parameters for the CRF.
        ## Returns
            map_score: np.ndarray (H, W)
                The predicted label.
            pred_prob: np.ndarray (n_classes, H, W)
                The predicted probability.
    '''

    _, _, n_classes = unary_potential.shape
    height, width, bands = input_features.shape

    C = int(np.ceil((width - patch_size) / stride) + 1)
    R = int(np.ceil((height - patch_size) / stride) + 1)
    # weight matrix B for avoiding boundaries of patches
    assert patch_size > stride, "patch_size {} should be larger than stride {}".format(patch_size, stride)
    w = patch_size
    s1 = stride
    s2 = w - s1
    d = 1 / (1 + s2)
    B1 = np.ones((w, w))
    B1[:, s1::] = np.dot(np.ones((w, 1)), (-np.arange(1, s2 + 1) * d + 1).reshape(1, s2))
    B2 = np.flip(B1)
    B3 = B1.T
    B4 = np.flip(B3)
    B = B1 * B2 * B3 * B4

    # not all the bands are uint8
    padded_input_features = np.zeros((patch_size + stride * (R - 1), patch_size +
                                     stride * (C - 1), bands), dtype=np.float32)
    ph, pw, pc = padded_input_features.shape
    padded_input_features[0:height, 0:width, :] = input_features
    padded_input_features[height:, :, :] = cv2.flip(padded_input_features[height - (ph - height):height, :, :], 0)
    padded_input_features[:, width:, :] = cv2.flip(padded_input_features[:, width - (pw - width):width, :], 1)

    padded_unaries = np.zeros((patch_size + stride * (R - 1), patch_size + stride * (C - 1), n_classes), dtype=np.float32)
    padded_unaries[0:height, 0:width, :] = unary_potential
    padded_unaries[height:, :, :] = cv2.flip(padded_unaries[height - (ph - height):height, :, :], 0)
    padded_unaries[:, width:, :] = cv2.flip(padded_unaries[:, width - (pw - width):width, :], 1)

    weight = np.zeros((patch_size + stride * (R - 1), patch_size + stride * (C - 1)), dtype=np.float32)

    pred_prob = np.memmap('pred_prob.dat', dtype=np.float32, mode='w+', shape=(n_classes,
                          patch_size + stride * (R - 1), patch_size + stride * (C - 1)))
    pred_prob[:] = np.zeros((n_classes, patch_size + stride * (R - 1), patch_size + stride * (C - 1)), dtype=np.float32)
    with tqdm.tqdm(total=R * C, desc="Tiled Dense CRF inference") as pbar:
        for r in range(R):
            for c in range(C):
                input_feature_tiles = padded_input_features[r * stride:r * stride + patch_size, c *
                                                            stride:c * stride + patch_size, :].astype(np.float32)
                unary_potential_tiles = padded_unaries[r * stride:r * stride + patch_size, c *
                                                       stride:c * stride + patch_size, :].astype(np.float32)

                map_score, pred = dense_crf(unary_potential_tiles, input_feature_tiles, **params)

                pred_prob[:, r * stride:r * stride + patch_size, c * stride:c * stride + patch_size] += pred * B
                weight[r * stride:r * stride + patch_size, c * stride:c * stride + patch_size] += B

                pbar.update(1)
    for b in range(n_classes):
        pred_prob[b, :, :] /= weight
    map_score = pred_prob.argmax(axis=0)
    return map_score[0:height, 0:width], pred_prob[:, 0:height, 0:width]
