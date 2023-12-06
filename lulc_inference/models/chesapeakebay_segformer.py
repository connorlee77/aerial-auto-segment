import json

import numpy as np
import tensorflow as tf
from doodleverse_utils.model_imports import (custom_resunet, custom_unet,
                                             segformer, simple_resunet,
                                             simple_satunet, simple_unet)


class ChesapeakeBaySegformer:
    '''
        From Doodleverse.
        Note: this model is not good
    '''
    def __init__(self):
        # Hardcode everything
        weights_list = ['pretrained_weights/ches_7class_naipRGB_512_segformer_v3_fullmodel.h5']
        self.model = get_model(weights_list)[0]
        print(self.model.summary())

    def __call__(self, x):
        # m = np.mean(x, axis=(2, 3), keepdims=True)
        # std = np.std(x, axis=(2, 3), keepdims=True)
        # m = np.mean(x, axis=(1, 2, 3), keepdims=True)
        # std = np.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - 0.4) / 0.2

        x = self.model.predict(x)
        x = tf.nn.softmax(x['logits'], axis=1)

        x = tf.image.resize(tf.transpose(x, perm=(0, 2, 3, 1)), size=(512, 512), method='nearest', antialias=True)
        return tf.transpose(x, perm=(0, 3, 1, 2))


def get_model(weights_list: list):
    """Loads models in from weights list and loads in corresponding config file
    for each model weights file(.h5) in weights_list

    Args:
        weights_list (list): full path to model weights files(.h5)

    Raises:
        Exception: raised if weights_list is empty
        Exception: An unknown model type was loaded from any of weights files in
        weights_list

    Returns:
       model, model_list, config_files, model_names
    """
    model_list = []
    config_files = []
    model_names = []
    if weights_list == []:
        raise Exception("No Model Info Passed")
    for weights in weights_list:
        # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
        # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
        configfile = weights.replace(".h5", ".json").strip()
        if "fullmodel" in configfile:
            configfile = configfile.replace("_fullmodel", "").strip()
        with open(configfile) as f:
            config = json.load(f)
        TARGET_SIZE = config.get("TARGET_SIZE")
        MODEL = config.get("MODEL")
        NCLASSES = config.get("NCLASSES")
        KERNEL = config.get("KERNEL")
        STRIDE = config.get("STRIDE")
        FILTERS = config.get("FILTERS")
        N_DATA_BANDS = config.get("N_DATA_BANDS")
        DROPOUT = config.get("DROPOUT")
        DROPOUT_CHANGE_PER_LAYER = config.get("DROPOUT_CHANGE_PER_LAYER")
        DROPOUT_TYPE = config.get("DROPOUT_TYPE")
        USE_DROPOUT_ON_UPSAMPLING = config.get("USE_DROPOUT_ON_UPSAMPLING")
        DO_TRAIN = config.get("DO_TRAIN")
        LOSS = config.get("LOSS")
        PATIENCE = config.get("PATIENCE")
        MAX_EPOCHS = config.get("MAX_EPOCHS")
        VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")
        RAMPUP_EPOCHS = config.get("RAMPUP_EPOCHS")
        SUSTAIN_EPOCHS = config.get("SUSTAIN_EPOCHS")
        EXP_DECAY = config.get("EXP_DECAY")
        START_LR = config.get("START_LR")
        MIN_LR = config.get("MIN_LR")
        MAX_LR = config.get("MAX_LR")
        FILTER_VALUE = config.get("FILTER_VALUE")
        DOPLOT = config.get("DOPLOT")
        ROOT_STRING = config.get("ROOT_STRING")
        USEMASK = config.get("USEMASK")
        AUG_ROT = config.get("AUG_ROT")
        AUG_ZOOM = config.get("AUG_ZOOM")
        AUG_WIDTHSHIFT = config.get("AUG_WIDTHSHIFT")
        AUG_HEIGHTSHIFT = config.get("AUG_HEIGHTSHIFT")
        AUG_HFLIP = config.get("AUG_HFLIP")
        AUG_VFLIP = config.get("AUG_VFLIP")
        AUG_LOOPS = config.get("AUG_LOOPS")
        AUG_COPIES = config.get("AUG_COPIES")
        REMAP_CLASSES = config.get("REMAP_CLASSES")
        try:
            # Get the selected model based on the weights file's MODEL key provided
            # create the model with the data loaded in from the weights file
            # Load in the model from the weights which is the location of the weights file
            model = tf.keras.models.load_model(weights)
        except BaseException:
            if MODEL == "resunet":
                model = custom_resunet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES,
                    kernel_size=(KERNEL, KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,  # 0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                    dropout_type=DROPOUT_TYPE,  # "standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                )
            elif MODEL == "unet":
                model = custom_unet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES,
                    kernel_size=(KERNEL, KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,  # 0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                    dropout_type=DROPOUT_TYPE,  # "standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                )
            elif MODEL == "simple_resunet":
                # num_filters = 8 # initial filters
                model = simple_resunet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    nclasses=NCLASSES,
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,  # 0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                    dropout_type=DROPOUT_TYPE,  # "standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    filters=FILTERS,  # 8,
                    num_layers=4,
                    strides=(1, 1),
                )
            # 346,564
            elif MODEL == "simple_unet":
                model = simple_unet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    nclasses=NCLASSES,
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,  # 0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                    dropout_type=DROPOUT_TYPE,  # "standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    filters=FILTERS,  # 8,
                    num_layers=4,
                    strides=(1, 1),
                )
            elif MODEL == "satunet":
                model = simple_satunet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    num_classes=NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1, 1),
                )
            elif MODEL=='segformer':
                id2label = {}
                for k in range(NCLASSES):
                    id2label[k]=str(k)
                model = segformer(id2label,num_classes=NCLASSES)
                # model.compile(optimizer='adam')
            else:
                raise Exception(f"An unknown model type {MODEL} was received. Please select a valid model.")
            # Load in custom loss function from doodleverse_utils
            # Load metrics mean_iou, dice_coef from doodleverse_utils
            # if MODEL!='segformer':
            #     model.compile(
            #         optimizer="adam", loss=dice_coef_loss(NCLASSES)
            #     )  # , metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])
            weights=weights.strip()
            model.load_weights(weights)

        model_names.append(MODEL)
        model_list.append(model)
        config_files.append(configfile)
    return model, model_list, config_files, model_names
