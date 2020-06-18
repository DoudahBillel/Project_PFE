#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from base64 import b64encode

from svgpathtools import svg2paths
from lxml import etree

from copy import deepcopy
from cairosvg.surface import PNGSurface
import re

from allensdk.api.queries.image_download_api import ImageDownloadApi
from allensdk.api.queries.svg_api import SvgApi
from allensdk.config.manifest import Manifest
from skimage.io import imread
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.ontologies_api import OntologiesApi
from IPython.display import HTML, display
from pathlib import Path
from tqdm import tqdm


# In[4]:


def main(depth=3):
    image_api = ImageDownloadApi()
    svg_api = SvgApi()


    atlas_id = 1

    # image_api.section_image_query(section_data_set_id) is the analogous method for section data sets
    atlas_image_records = image_api.atlas_image_query(atlas_id)

    # this returns a list of dictionaries. Let's convert it to a pandas dataframe
    atlas_image_dataframe = pd.DataFrame(atlas_image_records)[['id']]

    # and use the .head() method to display the first few rows
    #atlas_image_dataframe.head()
    # Create a directory to hold the masks (one svg and one png)

    rootDir  = Path()

    all_structues = pd.read_csv("complet.csv")

    #all_structues.head()
    for i in tqdm(atlas_image_dataframe.id):
        #print(i)
        # Load Image jpg
        maskSvgDir = rootDir / "Mask_svg"/str(i)
        maskSvgDir.mkdir(parents=True, exist_ok=True)

        maskPNGDir = rootDir / "Mask_png"/str(i)    
        maskPNGDir.mkdir(parents=True, exist_ok=True)

        svg_path = 'SVG_Coronal/'+str(i)+'.svg'

        original_image = imread('JPG_Coronal/'+str(i)+'.jpg')

        #original_image.shape

        #Load attributes and paths for SVG file
        paths, attributes = svg2paths(svg_path)
        # Retrive IDs structure
        sturctures = set([ v['structure_id'] for v in attributes])

        # it seems that level 7 is better, level 3 is very broad
        # For example here, level 3 is the Cerebral cortex, which is a large region that include other regions
        #all_structues[all_structues.atlas_id.isin(sturctures)]
        structures_toUse = all_structues[ (all_structues["atlas_id"].isin(sturctures)) & (all_structues['depth']==depth)].atlas_id.values

        # Read the download svg image
        with open(svg_path, 'rb') as svg_file:
            svg = svg_file.read()    

        # Convert to xlm element
        svg_elem = etree.fromstring(svg)

        # Scale the svg image according to the original image
        scale_x = float(original_image.shape[1])/float(svg_elem.get("width"))
        scale_y = float(original_image.shape[0])/float(svg_elem.get("height"))

        # Update the attributes of the svg image
        svg_elem.set("transform", "scale(%f %f)" % (scale_x, scale_y))
        new_width = str(original_image.shape[1])
        new_hight = str(original_image.shape[0])
        svg_elem.set('width', new_width)
        svg_elem.set('height', new_hight)

        #print("({},{})".format(svg_elem.get("height"), svg_elem.get("width") ))
        # We create an svg image for each mask
        # Make sure the colors belong to the list of regions.
        for structure_toKeep in structures_toUse:
            tmp_elem = deepcopy(svg_elem)
            # Modify all the path
            for elem in tmp_elem.findall(".//{http://www.w3.org/2000/svg}path"):                            
                    # remove all the paths that are are not from this region
                    if elem.attrib['structure_id'] != str(structure_toKeep):
                        elem.getparent().remove(elem)               

            # Generate the new svg
            maskedSvg = etree.tostring(tmp_elem,xml_declaration=False,standalone=True,pretty_print=True)            

            # Get the region name    
            region_name = all_structues[all_structues.atlas_id == structure_toKeep].name.values
            
                
            # remove special characters
            region_name= re.sub(r'\s+','_',str(region_name))
            region_name= re.sub(r'\W+','',region_name)


            # Save it as svg
            fout = maskSvgDir / "{}_mask.svg".format(region_name)
            fid = open(fout, 'wb')
            fid.write(maskedSvg)
            fid.close()


            # You can even save it as png
            png_out = maskPNGDir / "{}_mask.png".format(region_name)
            PNGSurface.convert(
                    bytestring=maskedSvg,
                    write_to=open(png_out, 'wb')
                    )


# In[5]:


if __name__ == '__main__':
    main()


# In[ ]:




