#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Machine Learning - Lab Work - 3 - TASK 4
# by Mercan Karacabey

# FOR IRIS DATASET - SOME DECISION TREE ALGORITHM RESULTS

# In[1]:


from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split


# In[2]:


iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
y = iris.target
df = iris.data


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[4]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[5]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


list1 = []
list2 = []

from sklearn.metrics import accuracy_score

for i in y:
    for z in df:
        list1.append(accuracy_score(i, y_pred))
        list2.append(accuracy_score(z, y_pred))

