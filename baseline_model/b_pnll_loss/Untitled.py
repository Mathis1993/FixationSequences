#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt


# In[6]:


train_loss = [1,2,3,4,5]
valid_loss = [2,3,4,5,6]


# In[10]:


#turn interactive mode off, because plot cannot be displayed in console
plt.ioff()

fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
#plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.title("Training and Validation Loss per Epoch", fontsize=20)
plt.tight_layout()
#plt.show() no showing, only saving
fig.savefig('loss_plot.png', bbox_inches='tight')


# In[ ]:




