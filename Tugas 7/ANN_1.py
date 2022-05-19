#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def forwardPass(inputs, weight, bias):
        w_sum = np.dot(inputs, weight) + bias

        # Linier Activation f(x) = x
        act = w_sum

        return act


# In[3]:


#pre-Trained Weight & Biases after Training
W= np.array([[2.9999999999]])
b= np.array([[1.999999999]])


# In[4]:


#Initialize Input Data
inputs = np.array([[7], [8], [9], [10]])


# In[5]:


# Output of Output Layer
o_out = forwardPass(inputs, W, b)


# In[6]:


print('Output Layer Output (Linear)')
print('===============================')
print(o_out, "\n")


# In[ ]:




