def resizeImage(infile, infile_name_only, output_dir="", size=(1024,768)):
  '''
  Resize Images to a requestet size (not considerinng aspect ratio)
  Input:
  - infile: image to be resized (with path)
  - infile_name_only: image to be resized (filename only)
  - output_dir: where resized images should be stored
  - size: output size (tupel of (height, width))
  '''
  
  outfile = os.path.splitext(infile_name_only)[0]
  extension = os.path.splitext(infile)[1]
  
  if infile != outfile:
    if not os.path.isfile(output_dir + "/" + outfile + extension):
      try :
        im = PIL.Image.open(infile)
        #crops to requested size independt from aspec ratio
        im = im.resize(size, PIL.Image.ANTIALIAS) 
        im.save(output_dir + "/" + outfile + extension)
      except IOError:
        print("cannot reduce image for ", infile)

#output_dir = "dataset/resized"
#size = (64, 64)
#filenames_dir = list(img_info["image_path"])
#filenames = list(img_info["filename"])
            
#for i in range(len(filenames)):
#  resizeImage(filenames_dir[i], filenames[i], output_dir=output_dir, size=size)