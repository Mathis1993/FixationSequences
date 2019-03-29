#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import math
import os
import numpy as np


# **Clean the data**
# <br>
# There seem to be cases in the dataset, where the fixation-vector is empty. This causes problems eg for the padding (expansion of the targets). Find and remove these cases.

# In[2]:


allImages_unfolded = pd.read_json("allImages_unfolded.json", orient="split")

#identify problematic cases: dimension of fixations-vector is less than 2 (less than 2 columns)
problems = []
for i in range(len(allImages_unfolded)):
    dims = torch.FloatTensor(allImages_unfolded.loc[i,"fixations"]).dim()
    if dims < 2:
        problems.append(i)

print(len(problems))

#drop these cases
allImages_unfolded_clean = allImages_unfolded.drop(problems)

#save to disk
allImages_unfolded_clean.to_json("allImages_unfolded_clean.json", orient="split", index=False)


# **Split the data into Train/Validation/Test Sets**

# In[3]:


#shuffle df: create random sample (specifying frac=1 so 100% of the original data)
allImages_unfolded_clean = allImages_unfolded_clean.sample(frac=1)

#define split
train_split = 0.7
val_split = 0.9
#rest is test_split

#split
allImages_unfolded_train = allImages_unfolded_clean.iloc[:math.ceil(train_split*len(allImages_unfolded_clean)), :]
print("train set size: " + str(len(allImages_unfolded_train)))
allImages_unfolded_val = allImages_unfolded_clean.iloc[len(allImages_unfolded_train):math.ceil(val_split*len(allImages_unfolded_clean)), :]
print("val set size: " + str(len(allImages_unfolded_val)))
allImages_unfolded_test = allImages_unfolded_clean.iloc[len(allImages_unfolded_train)+len(allImages_unfolded_val):, :]
print("test set size: " + str(len(allImages_unfolded_test)))

#save to disk
#orient="split" stores index and column information in lists, to keep the order of the df
#do not keep indices of the old whole dataset
allImages_unfolded_train.to_json("allImages_unfolded_train.json", orient="split", index=False)
allImages_unfolded_val.to_json("allImages_unfolded_val.json", orient="split", index=False)
allImages_unfolded_test.to_json("allImages_unfolded_test.json", orient="split", index=False)

