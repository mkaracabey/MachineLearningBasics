#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Machine Learning - Lab Work - 3 - TASK 3
# by Mercan Karacabey

# In[1]:


#LOAD LIBRARIES
import sklearn.datasets as datasets
import pandas as pd


# In[2]:


iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target


# In[3]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)


# Since the PydotPlus module used for Python 2.7. And I am using Python version 3.7. So it gives some errors.(Module not found)

# At the below code, draw decision path.

# In[4]:


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


# In[5]:


dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

