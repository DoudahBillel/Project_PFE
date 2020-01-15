#!/usr/bin/env python
# coding: utf-8


from matplotlib import pyplot as plt
from matplotlib.image import imread
import glob
import cv2
import scipy
import imageio
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed  

# Create a directory to hold the masks (one svg and one png)


from albumentations import (OneOf, Compose, IAAFliplr, IAAFlipud, IAAPerspective, IAAAffine,
                            Resize,
                            Rotate, RandomRotate90, VerticalFlip, CenterCrop,
                            Transpose,
                            ToGray, RGBShift, RandomContrast, RandomGamma, RandomBrightness,
                            RandomCrop, ShiftScaleRotate

                            )


def AugmentApply(OI, OM, augm, pxlT, PathM, MN, II, TN, **var):

    if(pxlT == None):
        tf = Compose([A(p=1, **var) for A in augm])
    elif(augm == None):
        l = [A(p=1, **var) for A in augm]
        l.append(pxlT(p=1))
        tf = Compose(l)
        del l
    else:
        tf = pxlT(p=1)
    data = {"image": OI, "masks": OM}

    augmented = tf(**data)

    image, masks = augmented["image"], augmented["masks"]

    imageio.imwrite('TransformedJPG/'+II+'_'+TN+'.jpg', image)

    folder = PathM/(II+'_'+TN)

    folder.mkdir(parents=True, exist_ok=True)

    for idx, im in enumerate(masks):
        imageio.imwrite(folder/(MN[idx]+".png"), im)

    del folder, image, masks, data, tf, augmented

    gc.collect()


## This function will be called in parallel for each image
def AugmentSingleImage(i, Masks):

    Original_image = imread("JPG_Coronal/"+i+".jpg")
    mask_file = glob.glob("Mask_png/"+i+"/*.*")
    Original_Maks = [imread(file) for file in mask_file]
    masks_names = [i.split("/")[2][:-4] for i in mask_file]

    # Transformations
    # flip Righ

    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[IAAFliplr],
                 pxlT=None,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="flip")

    # flip UP

    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[IAAFlipud],
                 pxlT=RandomBrightness,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="flipUp")

    # flip UpRight

    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[IAAFliplr, IAAFlipud],
                 pxlT=None,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="flipUpRight")

    # Transpose

    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[Transpose],
                 pxlT=RandomGamma,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="Transpose")

    # RandomeRotate

    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[RandomRotate90],
                 pxlT=RandomContrast,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="RandomeRotate")

    # ShiftScaleRotate
    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[ShiftScaleRotate],
                 pxlT=RGBShift,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="ShiftScaleRotate",
                 rotate_limit=160,
                 shift_limit=0.1,
                 scale_limit=0.3)
    # Just tp Gray

    AugmentApply(OI=Original_image,
                 OM=Original_Maks,
                 augm=[ShiftScaleRotate],
                 pxlT=ToGray,
                 PathM=Masks,
                 MN=masks_names,
                 II=i,
                 TN="ToGray")


# Just put the entry code in function [option, just to be cleaner]
def main():
    rootDir = Path()
    avaible_files = glob.glob("JPG_Coronal/*.*")
    ids = [i[12:-4] for i in avaible_files]

    TransformedJPG = rootDir/"TransformedJPG"
    TransformedJPG.mkdir(parents=True, exist_ok=True)

    Masks = rootDir / "TransformedMasks"
    Masks.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=4)(delayed(AugmentSingleImage)(i= i,Masks= Masks) for i in tqdm(ids))

if __name__ == "__main__":
    main()
