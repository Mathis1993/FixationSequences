import zipfile as zf

files = zf.ZipFile("data.zip", 'r')
files.extractall()
files.close()