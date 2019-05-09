import zipfile as zf

files = zf.ZipFile("figrim.zip", 'r')
files.extractall()
files.close()