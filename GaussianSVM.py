#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Machine Learning - LabWork-2 - Mercan Karacabey

# In[1]:


"""BDA502 SPRING 2018-19

- Gaussian Naive Bayes
- Support Vector Machines

TASK-1: Please explain how the Naive Bayesian algorithm work very briefly by using the example below.

TASK-2: Please use the small dataset about gender classification with Gaussian Naive Bayes.

TASK-3: Please use the digits dataset with comparing Gaussian Naive Bayes and Support Vector Machine algorithms. 

TASK-4: Illustrate a linear and non-linear SVM model with plots. You need to find a suitable example, run it and explain it.

"""


# ### TASK-1: Simple Example for GaussianNB from scikit-learn
#  x -> features
#  y -> labels
# 
#  GaussianNB algorithm --  Classification Algorithm
#  We have two sets. 1 and 2 . Then we want to show that new value is which set. fit function -- run the gaussian method for A dataset and B sets.
#  
#  1- Class prior value - Which set is appropriate for new values? And close which set.
#  2- Class Count value - During the training step , how much class was generated?
#  3- Score function - Algorithm and training result for new values

# In the below code, I calculated time difference of fit the algorithm GaussianNB.

# In[4]:


import time
start = time.time()
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 1, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None)
clf.fit(X, Y) # GaussianNB(priors=None)
done = time.time()
elapsed = done - start
print(elapsed)


# Time difference : 0.0019

# GaussianNB is a classification algorithm. Calculate probability value and decide which set is appropriate for value depends on probability value

# In[5]:


print(clf.predict([[-0.8, -1]]))
print(clf.class_prior_)  # probability of each class.
print(clf.class_count_)  # number of training samples observed in each class.
print(clf.theta_)  # mean of each feature per class.
print(clf.sigma_)  # variance of each feature per class
print(clf.predict_proba([[0.8, 1]]))  # Return probability estimates for the test vector X.
print(clf.predict_log_proba([[0.8, 1]]))  # Return log-probability estimates for the test vector X.
print(clf.score([[0.8, 1]],[1]))  # Returns the mean accuracy on the given test data and labels.
print(clf.score([[0.8, 1]],[2]))  # Returns the mean accuracy on the given test data and labels.


# Predict function - Guess the set 1 or set 2. 
# 
# Class-Prior Value - Dataset density based on count set1 and set2
# 
# Class-Count Value - Count set1 and set2
# 
# Theta Value - Mean Value of each class
# 
# Sigma Value - Variance of each class
# 
# Score function - Shows that accuracy rate for given value.

# #### Interpretation Results

# Predict Function Result :  1 => Given value is in First Set.
#     
# Class Prior Value : 0.66 / 0.33 => In the set, Count of 1 : 4/6 Count of 2 : 2/6
#             
# Score Function : 1.0 / 0.0 => Given value: Is 0.8, 1 in [1] ? Yes given value is in set 1. TRUE 1.0
# 
# And against to this => FALSE 0.0

# ### TASK-2: Please use the small dataset about gender classification with Gaussian Naive Bayes and compare the results.
# 
# http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvTmFpdmVfQmF5ZXNfY2xhc3NpZmllcg
# 

# In[7]:


## Height, Weight and Foot Size is input and we access to new variable in which set.
## It has done Probability based estimation. This algorithm creates continuous function and trains.
## Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
import time
start = time.time()

A = np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])
B = np.array([1,1,1,1,2,2,2,2])

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None)
clf.fit(A, B) # GaussianNB(priors=None)

done = time.time()
elapsed = done - start
print(elapsed)


# Time difference : 0.00099

# In the below and above Cell values was taken by wikizero site.

# In[8]:


print(clf.predict([[5,150 ,10]]))
print(clf.class_prior_)  # probability of each class.
print(clf.class_count_)  # number of training samples observed in each class.
print(clf.theta_)  # mean of each feature per class.
print(clf.sigma_)  # variance of each feature per class
print(clf.predict_proba([[5,150 ,10]]))  # Return probability estimates for the test vector X.
print(clf.predict_log_proba([[5,150 ,10]]))  # Return log-probability estimates for the test vector X.
print(clf.score([[6, 160 ,10]],[1]))  # Returns the mean accuracy on the given test data and labels.
print(clf.score([[6, 160,10]],[2]))  # Returns the mean accuracy on the given test data and labels.


# Male Count : 4 and Female Count : 4 Class Prior -> 0.5 0.5
#         
# Predict Function Result -> [5,150,10] : Set [2] 
#     
# [6,160,10] is in set [1] ? Yes , Result is TRUE. Against to it set [2] is FALSE. 
# 
# Match the results.

# For more small dataset and different from wikipage.

# In[10]:


import time
start = time.time()

A = np.array([[5.67,150,11],[6,190,9],[5.48,150,12]])
B = np.array([2,1,2])

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None)
clf.fit(A, B) # GaussianNB(priors=None)

done = time.time()
elapsed = done - start
print(elapsed)


# Since the same algorithms run, time differences results are same.

# In[11]:


print(clf.predict([[5,150 ,10]]))
print(clf.class_prior_)  # probability of each class.
print(clf.class_count_)  # number of training samples observed in each class.
print(clf.theta_)  # mean of each feature per class.
print(clf.sigma_)  # variance of each feature per class
print(clf.predict_proba([[5,150 ,10]]))  # Return probability estimates for the test vector X.
print(clf.predict_log_proba([[5,150 ,10]]))  # Return log-probability estimates for the test vector X.
print(clf.score([[6, 160 ,10]],[1]))  # Returns the mean accuracy on the given test data and labels.
print(clf.score([[6, 160,10]],[2]))  # Returns the mean accuracy on the given test data and labels.


# When the dataset is small, predict value was changed. And found false. 
# So, dataset size is important we can see from this result. 

# ### TASK-3: Please use the digits dataset with comparing Gaussian Naive Bayes and Support Vector Machine algorithms. 
# 
# Classification applications on the handwritten digits data 
# In this example, you will see two different applications of Naive Bayesian Algorithm on the digits data set.
# 

# In[12]:


import pylab as pl
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pylab as plt

########################################################################################################################
##################################### PART A ###########################################################################
########################################################################################################################
np.random.seed(42)  # random seeding is performed
digits = load_digits()  # the whole data set with the labels and other information are extracted
data = scale(digits.data)  # the data is scaled with the use of z-score
n_samples, n_features = data.shape  # the no. of samples and no. of features are determined with the help of shape
n_digits = len(np.unique(digits.target))  # the number of labels are determined with the aid of unique formula
labels = digits.target  # get the ground-truth labels into the labels

print(digits)
print(data)



# In[13]:


print(labels)  # the labels are printed on the screen
print(digits.keys())  # this command will provide you the key elements in this dataset
print(digits.DESCR)  # to get the descriptive information about this dataset

pl.gray()  # turns an image into gray scale
pl.matshow(digits.images[0])
pl.show()
print(digits.images[0])


# In the below code , fit the normal digit data. Applied to GaussianNB algorithm and predict it.

# In[16]:


# Train-test split is one of the most critical issues in ML
# Try this example with different test size like 0.1 or 0.5
import time
start = time.time()
y = digits.target
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
X1 = data
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.3, random_state=10)

print(len(X))
print(len(X_train))


gnb = GaussianNB(priors=None)
# gnb = GaussianNB(priors=[0.1, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
fit = gnb.fit(X_train, y_train)
predicted = fit.predict(X_test)
print(confusion_matrix(y_test, predicted))
print(accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print(accuracy_score(y_test, predicted, normalize=False))  # the number of correct predictions
print(len(predicted))  # the number of all of the predictions
unique_p, counts_p = np.unique(predicted, return_counts=True)
print(unique_p, counts_p)
print((predicted == 0).sum())
print(fit.get_params())
print(fit.predict_proba(X))

done = time.time()
elapsed = done - start


# In[17]:


print(elapsed)


# Time Difference : 0.036 

# Confusion Matrix => Rate of true predict for each color of digit array/image.
# Accuracy Rate => 0.81 . This result find by diagonal line values / Total Values

# In the below code, for the scaled data.

# In[15]:


# Train-test split is one of the most critical issues in ML
# Try this example with different test size like 0.1 or 0.5
y = digits.target
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
X1 = data
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.3, random_state=10)

print(len(X))
print(len(X_train))


gnb = GaussianNB(priors=None)
# gnb = GaussianNB(priors=[0.1, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
fit = gnb.fit(X_train1, y_train1)
predicted = fit.predict(X_test1)
print(confusion_matrix(y_test1, predicted))
print(accuracy_score(y_test1, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print(accuracy_score(y_test1, predicted, normalize=False))  # the number of correct predictions
print(len(predicted))  # the number of all of the predictions
unique_p, counts_p = np.unique(predicted, return_counts=True)
print(unique_p, counts_p)
print((predicted == 0).sum())
print(fit.get_params())
print(fit.predict_proba(X1))


# Confusion Matrix => Rate of true predict for each color of digit array/image. Accuracy Rate => 0.76 . This result find by diagonal line values / Total Values
# 
# Accuracy rate was decreased. Because dataset is not normal distributed. So, if the data was scaling, accuracy rate was descreased.

# ### TASK - 4 Same dataset with SVM Algorithm

# In[19]:



import numpy as np
np.random.seed(42)  # random seeding is performed
digits = load_digits()  # the whole data set with the labels and other information are extracted
n_samples, n_features = data.shape  # the no. of samples and no. of features are determined with the help of shape
n_digits = len(np.unique(digits.target))  # the number of labels are determined with the aid of unique formula
labels = digits.target  # get the ground-truth labels into the labels

print(digits)

print(labels)  # the labels are printed on the screen
print(digits.keys())  # this command will provide you the key elements in this dataset
print(digits.DESCR)  # to get the descriptive information about this dataset


# SVM algorithm seperate classes for each label. (According to their distances(Ocliddean Distances))

# In[22]:


import time
start = time.time()

from sklearn.svm import SVC

y = digits.target
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
clf = SVC()
clf.fit(X, y)

fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
print(confusion_matrix(y_test, predicted))
print(accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print(accuracy_score(y_test, predicted, normalize=False))  # the number of correct predictions
print(len(predicted))  # the number of all of the predictions
unique_p, counts_p = np.unique(predicted, return_counts=True)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

done = time.time()
elapsed = done - start


# In[23]:


print(elapsed)

