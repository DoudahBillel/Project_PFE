
from pathlib import Path
from shutil import copy,copytree
import random
from tqdm import tqdm

class spliter(object):

    def __init__(self,config, percent):
        self.config = config
        self.percent = percent

        self.Train_images = []
        self.Train_masks = []
        self.Validation_images =[]
        self.Validation_masks = []


    def copyTrainingData(self):                        
        for i in tqdm(range(len(self.Train_images)), desc="generating training data"):
            outimg = Path(self.config.TRAINING_IMAGES_FOLDER) / self.Train_images[i].name
            copy(self.Train_images[i], outimg)
            outdir = Path(self.config.TRAINING_MASKS_FOLDER) / self.Train_masks[i].stem
            copytree(self.Train_masks[i], outdir)        
    
    def copyValidationData(self):

        for i in tqdm(range(len(self.Validation_images)),desc="generating calidation data"):
            outimg = Path(self.config.TRAINING_IMAGES_FOLDER) / self.Validation_images[i].name
            copy(self.Validation_images[i], outimg)
            outdir = Path(self.config.TRAINING_MASKS_FOLDER) / self.Validation_masks[i].stem
            copytree(self.Validation_masks[i], outdir)

    
    def split_data(self):        
        ids = [ x.stem.split("_")[0] for x in Path(self.config.RAW_IMAGES).glob("*.jpg") ]
        ids = set(ids)
        
        # For reproducibility
        random.seed(self.config.RANDOM_SEED)

        # From each slice sample a group of images
        for img in ids:
            # Get the list of all the images associated with this id
            pattern = "{}*.jpg".format(img)
            id_images = [ x for x in Path(self.config.RAW_IMAGES).glob(pattern) ]

            # Sample a group of them for training
            train = random.sample(id_images, k= int( len(id_images) * self.percent ) )
            self.Train_images.extend(train)
            train_masks = [Path(self.config.RAW_MASKS) / f.stem for f in train]
            self.Train_masks.extend(train_masks)

            # the rest will be used for validation
            validation = list( set(id_images) - set(train))
            self.Validation_images.extend(validation)
            validation_masks = [Path(self.config.RAW_MASKS) / f.stem for f in validation]
            self.Validation_masks.extend(validation_masks)
                            
        # Copy Training data
        self.copyTrainingData()

        # Copy validation data
        self.copyValidationData()
    



    