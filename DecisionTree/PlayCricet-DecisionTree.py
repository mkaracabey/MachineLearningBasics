#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Machine Learning - Lab Work - 3
# by Mercan Karacabey

# Decision Tree Implementation with 2 Method -- Gini & Entropy

# 1 - Using Gini Method

# In[1]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[10]:


pima = pd.read_csv("C:\\Users\\mercan.karacabey\\Desktop\\ml_dataset-1.xls")


# In[14]:


from pandas import ExcelWriter
from pandas import ExcelFile
 
df = pd.read_excel("C:\\Users\\mercan.karacabey\\Desktop\\ml_dataset-1.xlsx")
 
print("Column headings:")
print(df.columns)


# In[15]:


df.head()


# In[30]:


df['PlaysCricet']


# In[20]:


X = df.values[:, 1:5]
Y = df.values[:,0]

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


# In[21]:


## DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)


# In[22]:


clf_gini.fit(X_train, y_train)


# In[23]:


DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')


# In[31]:


y_pred = clf_gini.predict(X_test)
print("Prediction Result:",y_pred)


# In[29]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# 2 - Using Entropy Method

# In[33]:


clf_gini_ent = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=3, min_samples_leaf=5)


# In[32]:


DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')


# In[35]:


clf_gini_ent.fit(X_train, y_train)


# In[37]:


y_pred_ent = clf_gini_ent.predict(X_test)
print("Prediction Result:",y_pred_ent)


# In[38]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

