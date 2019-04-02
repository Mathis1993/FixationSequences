from utils.ResizeImages import resizeImage
import pandas as pd

#read json with all images (version with each image occuring only once)
#read json with all images (version with each image occuring only once)
allImages = pd.read_json("allImages.json", orient="split")

#take series (column) with the image path and convert it to list
#loop over that list and resize images, just overwriting them with their smaller versions (input directory = output directory)
basic_path = "figrim/fillerData/"
size = (100,100)

for i in range(len(allImages)):
    filename = allImages.loc[i, "filename"]
    im_path = allImages.loc[i, "impath"]
    cur_path = basic_path + im_path
    resizeImage(infile=cur_path, infile_name_only=filename, output_dir=cur_path, size=size)

print("Successfully resized images to {}.".format(size))