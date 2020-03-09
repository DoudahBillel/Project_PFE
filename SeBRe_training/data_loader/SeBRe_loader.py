from base.base_datasets import Dataset
from natsort import natsorted, ns #for sorting filenames in a directory
from pathlib import Path
import skimage
import pandas as pd
import numpy as np
import re
import cv2




class SeBReDataLoader(Dataset):
    """Generates the brain section dataset. The dataset consists of locally stored 
    brain section images, to which file access is required.
    """

    def __init__(self, class_map, config, mode):         
        """initialization of the data loading parameters
        
        Arguments:
            config {Config} -- A Config class object containing the configuration
            mode {str} -- A string specifiying the type of the data to load (training or validation),        
        """
        super(SeBReDataLoader, self).__init__(class_map)
        assert mode in ['training', 'validation']
        self.mode = mode
        self.config = config
        self.all_structues = None

    def load_brain(self):
        """
        for naming image files follow this convention: '*_(image_id).jpg'
        """
        # Read the list of the brain structures and add them to the list
        self.all_structues = (pd
                 .read_csv(self.config.BRAIN_STRUCTURES)
                 .pipe(lambda x: x.loc[x.depth==7,['atlas_id','name']])
                 .assign(name = lambda x: [ re.sub(r'\W+','',re.sub(r'\s+','_',str(i))) for i in x.name]))

        for idx, i in enumerate(self.all_structues.name.tolist()):
            self.add_class('brain',str(idx),i)

        if self.mode == "training":
            img_list = [ str(x) for x in Path(self.config.TRAINING_IMAGES_FOLDER).glob("*.jpg")] 
        elif mode == "validation":
            img_list = [ str(x) for x in Path(self.config.VALIDATION_IMAGES_FOLDER).glob("*.jpg")]
        
        for i in img_list: #image_ids start at 0 (to keep correspondence with load_mask which begins at image_id=0)!
                img = skimage.io.imread(i) #grayscale = 0
                im_dims = np.shape(img)
                self.add_image("brain", image_id=i[:-4], path = cwd+'/'+i,height = im_dims[0], width = im_dims[1])#, depth = im_dims[2])
    
    def load_mask(self,image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks."""

        masks_folder = self.config.TRAINING_MASKS_FOLDER if self.config.mode == "training" else self.config.VALIDATION_MASKS_FOLDER
        subfolder = Path(masks_folder) / str(self.image_info[image_id]['id'])

        mk_list = [ str(x) for x in subfolder.glob(".png")]
        count = len(mk_list)

        mk_id = 0
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        class_ids = np.zeros(count)

        for m in mk_list:
            bin_mask = np.array(cv2.imread(m)[:,:,0]) # grayscale=0
            mk_size = np.shape(bin_mask)
            mask[:, :, mk_id]= bin_mask
            
            # Map class names to class IDs.
            class_ids[mk_id] = self.all_structues.loc[ self.all_structues.name==m[:-9],'atlas_id'] #fifth last position from mask_image name = class_id #need to update(range) if class_ids become two/three-digit numbers 
            mk_id += 1            
        return mask, class_ids.astype(np.int32)

        




