from utils.ResizeImages import resizeImage
import pandas as pd

#read json with all images (version with each image occuring only once)
#take series (column) with the image path and convert it to list
#loop over that list and resize images, just overwriting them with their smaller versions (input directory = output directory)
