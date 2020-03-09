import json
from dotmap import DotMap
import os
import time
from base.base_config import Config
from pathlib import Path


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config_dotmap = DotMap(config_dict)
    config = buildSeBReConfigClass(config_dotmap)
    return config


def buildSeBReConfigClass(config_dotmap):
    config = Config()

    config.NAME = config_dotmap.exp.name
    config.MODEL_DIR = Path("experiments") / time.strftime("%Y-%m-%d/",time.localtime()) / config.NAME / "logs/"
    config.MODEL_SAVE_DIR = Path("experiments") / time.strftime("%Y-%m-%d/",time.localtime()) / config.NAME / "weights/"
    config.RAW_IMAGES = config_dotmap.data.RAW_IMAGES
    config.RAW_MASKS = config_dotmap.data.RAW_MASKS
    config.RANDOM_SEED = config_dotmap.data.RANDOM_SEED

    config.GPU_COUNT = config_dotmap.GPU.GPU_COUNT
    config.IMAGES_PER_GPU = config_dotmap.GPU.IMAGES_PER_GPU

    config.STEPS_PER_EPOCH = config_dotmap.trainer.STEPS_PER_EPOCH
    config.NUM_CLASSES = config_dotmap.model.NUM_CLASSES
    config.IMAGE_MIN_DIM = config_dotmap.trainer.IMAGE_MIN_DIM
    config.VALIDATION_STEPS = config_dotmap.trainer.VALIDATION_STEPS

    config.RPN_ANCHOR_SCALES = [int(x) for x in config_dotmap.RPN.RPN_ANCHOR_SCALES.split(",")] 
    config.RPN_ANCHOR_RATIOS = [float(x) for x in config_dotmap.RPN.RPN_ANCHOR_RATIOS.split(",")] 
    config.RPN_ANCHOR_STRIDE = config_dotmap.RPN.RPN_ANCHOR_STRIDE
    config.RPN_NMS_THRESHOLD = config_dotmap.RPN.RPN_NMS_THRESHOLD
    config.TRAIN_ROIS_PER_IMAGE = config_dotmap.RPN.TRAIN_ROIS_PER_IMAGE
    config.RPN_TRAIN_ANCHORS_PER_IMAGE = config_dotmap.RPN.RPN_TRAIN_ANCHORS_PER_IMAGE
    config.POST_NMS_ROIS_TRAINING = config_dotmap.RPN.POST_NMS_ROIS_TRAINING
    config.POST_NMS_ROIS_INFERENCE = config_dotmap.RPN.POST_NMS_ROIS_INFERENCE
    config.BACKBONE_STRIDES = [int(x) for x in config_dotmap.RPN.BACKBONE_STRIDES.split(",")] 

    config.USE_MINI_MASK = config_dotmap.RPN.USE_MINI_MASK
    config.MINI_MASK_SHAPE = [int(x) for x in config_dotmap.RPN.MINI_MASK_SHAPE.split(",")] 
    config.ROI_POSITIVE_RATIO = config_dotmap.RPN.ROI_POSITIVE_RATIO
    config.POOL_SIZE = config_dotmap.RPN.POOL_SIZE
    config.MASK_POOL_SIZE = config_dotmap.RPN.MASK_POOL_SIZE
    config.MASK_SHAPE = [int(x) for x in config_dotmap.RPN.MASK_SHAPE.split(",")] 
    config.IMAGE_PADDING = config_dotmap.RPN.IMAGE_PADDING
    config.MEAN_PIXEL = [float(x) for x in config_dotmap.RPN.MEAN_PIXEL.split(",")] 
    config.MAX_GT_INSTANCES = config_dotmap.RPN.MAX_GT_INSTANCES

    config.NUM_CLASSES = config_dotmap.model.NUM_CLASSES
    config.LEARNING_RATE = config_dotmap.model.LEARNING_RATE
    config.LEARNING_MOMENTUM = config_dotmap.model.LEARNING_MOMENTUM
    config.WEIGHT_DECAY = config_dotmap.model.WEIGHT_DECAY
    
    config.NUM_CLASSIMAGE_MAX_DIMES = config_dotmap.trainer.IMAGE_MAX_DIM
    config.DETECTION_MAX_INSTANCES = config_dotmap.trainer.DETECTION_MAX_INSTANCES
    config.USE_RPN_ROIS = config_dotmap.trainer.USE_RPN_ROIS
    config.DETECTION_MIN_CONFIDENCE = config_dotmap.trainer.DETECTION_MIN_CONFIDENCE
    config.DETECTION_NMS_THRESHOLD = config_dotmap.trainer.DETECTION_NMS_THRESHOLD

    if not Path(config_dotmap.data.BRAIN_STRUCTURES).exists():
        raise FileNotFoundError("couldn't find file: {}".format(config_dotmap.data.BRAIN_STRUCTURES))

    config.BRAIN_STRUCTURES = config_dotmap.data.BRAIN_STRUCTURES

    # Check if the training data folders exists
    if not Path(config_dotmap.trainer.TRAINING_IMAGES_FOLDER).exists():
        raise FileNotFoundError("couldn't find folder: {}".format(config_dotmap.trainer.TRAINING_IMAGES_FOLDER))

    config.TRAINING_IMAGES_FOLDER = config_dotmap.trainer.TRAINING_IMAGES_FOLDER

    if not Path(config_dotmap.trainer.TRAINING_MASKS_FOLDER).exists():
        raise FileNotFoundError("couldn't find folder: {}".format(config_dotmap.trainer.TRAINING_MASKS_FOLDER))

    config.TRAINING_MASKS_FOLDER = config_dotmap.trainer.TRAINING_MASKS_FOLDER


    # Check if the validation data folders exists
    if not Path(config_dotmap.validator.VALIDATION_IMAGES_FOLDER).exists():
        raise FileNotFoundError("couldn't find folder: {}".format(config_dotmap.validator.VALIDATION_IMAGES_FOLDER))

    config.VALIDATION_IMAGES_FOLDER = config_dotmap.validator.VALIDATION_IMAGES_FOLDER

    if not Path(config_dotmap.validator.VALIDATION_MASKS_FOLDER).exists():
        raise FileNotFoundError("couldn't find folder: {}".format(config_dotmap.validator.VALIDATION_MASKS_FOLDER))

    config.VALIDATION_MASKS_FOLDER = config_dotmap.validator.VALIDATION_MASKS_FOLDER

    if config_dotmap.model.WEIGHTS_HDF5 != "":
        if not Path(config_dotmap.model.WEIGHTS_HDF5).exists():
            raise FileNotFoundError("couldn't find file: {}".format(config_dotmap.model.WEIGHTS_HDF5))
        config.WEIGHTS_HDF5 = config_dotmap.model.WEIGHTS_HDF5
    else:
        config.WEIGHTS_HDF5 = None

    return config    




def process_config(json_file):
    config = get_config_from_json(json_file)    
    return config
