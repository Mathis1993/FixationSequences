
# coding: utf-8

# ### Description of the dataset in Matlab

# We are using the FIGRIM Fixations Dataset obtained from 
# [here](http://figrim.mit.edu/index_eyetracking.html).
# 
# More specifically, we are using a subset of 2157 filler images presented once to an average of 15 subjects while recording eye fixation data. The images belong to 21 indoor and outdoor scene categories.
# 
# The dataset is available as a .mat-file.
# 
# In a first step, the dataset is going to be translated to a pandas dataframe in python.

# ![grafik.png](attachment:grafik.png)

# The data is saved in a structure, containing four fields:
# -  the filename of the image
# -  the category of the image
# -  the path to the image
# -  another structure labeled "userdata"
# 
# So each row of the structure represents the data for one filler image.

# ![grafik.png](attachment:grafik.png)

# The "userdata"-structure contains data for every subject a picture has been presented to:
# - a trial number
# - a label for the recording session
# - sinit, a field about which we have no further information
# - the subjects' response in the continous recognition task: 2 = false alarm, 4 = correct rejection (as these are filler images only presented once, each image should be rejected as it has never been presented before and the second and third entry are missing values)
# - fixations: A structure with fields for fixation data for potential 3 presentations of an image (as we only regard fillers, only the field for the first presentation holds data)

# ### Conversion of Matlab-Structure to Python

# #### Preparation of the dataset in Matlab

# In order to simplify the data's representation, empty rows in the "userdata"-structure are deleted. Also, the structure stored in the field "fixations" is resolved to this field directly holding the fixation data for a filler image's only presentation. Both of this is done in this [matlab script](files/Matlab/preparation.m).

# ![grafik.png](attachment:grafik.png)

# After applying the script, the "userdata"-structure contains no more empty rows and the fixation data for each filler image is directly stored in the field "fixations" as a matrix containing one row for each fixation (ordered sequentially, so that the first row is the first fixation), and one column for the respective x- and y-coordinate of each fixation.

# #### Rebuilding the dataset in Python

# In[1]:


#loadmat can read .mat-files
from scipy.io import loadmat
import pandas as pd
import imageio
import numpy as np
import pickle


# In[2]:

def convert_data():
    datensatz = loadmat("figrim/fillerData/allImages_edited.mat")


    # In[3]:


    #type(datensatz)
    #for key, value in datensatz.items():
    #    print(key)


    # As we can see, the dataset is represented as a dict in python, containing one key "all".

    # In[4]:


    datensatz["all"].shape


    # In[5]:


    datensatz["all"][0,0]


    # Under this key, the whole dataset is stored as a matrix with one row and 2157 columns, so one column per filler image.

    # In[6]:


    datensatz["all"]["filename"]


    # The fields from the .mat-file can still be adressed.

    # In[7]:


    datensatz["all"][0,0][3][0,0][4]
    datensatz["all"]["userdata"][0,0][0,0][4]


    # In total, the (1, 2157)-matrix holds in each column an array (or rather a vector)  containing all data for one image. The elements of this array are themselves arrays, or in the case of the "userdata", nested arrays. Single elements can be adressed by selecting a column (corresponding to a row in the matlab-structure) of the (1,2157)-array and then an element of this column, which again can have elements on its own. To get to the fixation data of the first subject for the first image, we select the first column of the (1,2157)-array. From this column, we select the fourth entry (or the entry labeled "userdata"). "userdata" itself is an array with one row and one column per subject, so we select the first column from this array and then the fifth element from this column, corresponding to the fixation data.   

    # To rebuild the dataset in a way that resembles the matlab-structure and allows easy indexing/assessing of the data, we will build a pandas dataframe containing the columns
    # - filenmame
    # - category
    # - impath
    # - userdata
    # 
    # and one row per image, just like in the matlab-structure.
    # 
    # The column "userdata" contains further dataframes, containing the columns
    # - trial
    # - recording session
    # - sinit
    # - SDT
    # - fixations
    # 
    # thus storing all information about each subject that has been presented an image.
    # 

    # In[8]:


    #recreate filename, category and impath, each as list
    filename = []
    category = []
    impath = []

    lists = [filename, category, impath]
    for h in range(0,len(lists)):
        #for each column (for each image)
        for i in range(datensatz["all"].size):
            #extract the filename, category and impath
            lists[h].append(datensatz["all"][0,i][h][0])


    # In[9]:


    #recreate userdata as list of dataframes
    userdata = []

    #for each column (for each image)
    for h in range(0,datensatz["all"].size):
        trial = []
        #iterate through every subject in "userdata"
        for i in range(0,datensatz["all"][0,h][3].size):
            #and extract the trial-number
            trial.append(datensatz["all"][0,h][3][0,i][0][0,0])
        len(trial)

        recording_session = []
        for i in range(0,datensatz["all"][0,h][3].size):
            recording_session.append(datensatz["all"][0,h][3][0,i][1][0])
        len(recording_session)

        sinit = []
        for i in range(0,datensatz["all"][0,h][3].size):
            sinit.append(datensatz["all"][0,h][3][0,i][2][0])
        len(sinit)

        sdt = []
        for i in range(0,datensatz["all"][0,h][3].size):
            sdt.append(datensatz["all"][0,h][3][0,i][3])
        sdt

        fixations = []
        for i in range(0,datensatz["all"][0,h][3].size):
            fixations.append(datensatz["all"][0,h][3][0,i][4])
        fixations[0].shape

        #for each column (each image), put all lists together as a df and append it to the list of dfs
        daten = {"trial":trial, "recording_session_label":recording_session, "sinit":sinit, "SDT":sdt, "fixations":fixations}
        df = pd.DataFrame(data=daten, columns = ["trial", "recording_session_label", "sinit", "SDT", "fixations"])
        userdata.append(df)


    # In[10]:


    #put everything into one dataframe containing filename, category, impath and userdata
    daten = {"filename":filename, "category":category, "impath":impath, "userdata":userdata}
    allImages = pd.DataFrame(data=daten, columns = ["filename", "category", "impath", "userdata"])
    allImages
    allImages.loc[0,"userdata"]


    # ##### Unfolding

    # In order to obtain a two-dimensional dataframe, now every subject per image is treated as one case (one row).

    # In[11]:


    #Copy df
    allImages_unfolded = allImages.copy()


    # In[12]:


    #add filename, category and impath to each userdata-df
    new_cols = ["filename", "category", "impath"]
    for i in range(0,len(allImages_unfolded)):
        for j in range(0,len(new_cols)):
            for k in range(0,len(allImages_unfolded.loc[i,"userdata"])):
                allImages_unfolded.loc[i,"userdata"].loc[k,new_cols[j]] = allImages_unfolded.loc[i,new_cols[j]]


    # In[13]:


    #concatenate all userdata-dfs together
    frames = []
    for i in range(0,len(allImages_unfolded)):
        frames.append(allImages_unfolded.loc[i,"userdata"])
    allImages_unfolded_total = pd.concat(frames, ignore_index=True)
    allImages_unfolded = allImages_unfolded_total


    # In[14]:


    #change column order
    allImages_unfolded = allImages_unfolded[["filename", "category", "impath", "trial", "recording_session_label", "sinit", "SDT", "fixations"]]


    # In[15]:


    #result: two-dimensional df
    allImages_unfolded


    # This can now be stored in a .json-file

    # In[16]:


    #orient="split" stores index and column information in lists, to keep the order of the df
    allImages_unfolded.to_json("allImages_unfolded.json", orient="split")


    # In[18]:


    #use orient="split" on reading as well
    allImages_unfolded = pd.read_json("allImages_unfolded.json", orient="split")
    allImages_unfolded


    # **Result**
    # <br>
    # `allimages_unfolded.json` containing an unfolded version of the previous Matlab-Structure.
    # <br>
    # This contains `{{len(allImages_unfolded)}}` cases. Each case is an image and fixation data for this image from one subject. An average of 15 subjects viewed each image.
    # 
    # 
    return print("Successfully converted dataset from Matlab to Python.")
