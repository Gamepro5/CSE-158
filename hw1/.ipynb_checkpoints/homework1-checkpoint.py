#!/usr/bin/env python
# coding: utf-8

# In[189]:


import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import sklearn
import numpy
import random
import gzip
import math


# In[190]:


import warnings
warnings.filterwarnings("ignore")


# In[191]:


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[192]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[193]:


len(dataset)


# In[194]:


answers = {} # Put your answers to each question in this dictionary


# In[195]:


dataset[0]


# In[196]:


### Question 1


# In[197]:


def feature(datum):
    # your implementation
    return datum['review_text'].count("!")


# In[198]:


ratings = [d['rating'] for d in dataset]
exclamationMarks = [feature(d) for d in dataset]


# In[199]:


X = [[m] for m in exclamationMarks]
y = ratings


# In[200]:


model = sklearn.linear_model.LinearRegression(fit_intercept=True)
model.fit(X, y)


# In[201]:


(theta0, theta1, mse) = [float(model.intercept_), float(model.coef_[0]), float(sklearn.metrics.mean_squared_error(y,model.predict(X)))]


# In[202]:


answers['Q1'] = [theta0, theta1, mse]
answers['Q1']


# In[203]:


assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


# In[204]:


### Question 2


# In[205]:


def feature(datum):
    return [1,datum['review_text'].count("!"), len(datum["review_text"])]


# In[206]:


ratings = [d['rating'] for d in dataset]
features = [feature(d) for d in dataset]


# In[207]:


X = [m for m in features]
y = ratings


# In[208]:


model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)


# In[209]:


(theta0, theta1, theta2, mse) = [float(model.coef_[0]), float(model.coef_[1]), float(model.coef_[2]), float(sklearn.metrics.mean_squared_error(y,model.predict(X)))]


# In[210]:


answers['Q2'] = [theta0, theta1, theta2, mse]


# In[211]:


assertFloatList(answers['Q2'], 4)


# In[212]:


### Question 3


# In[213]:


def feature(datum, deg):
    # feature for a specific polynomial degree
    result = []
    count = datum['review_text'].count("!")
    for i in range(deg):
        result.append(pow(count,i+1))
    return result


# In[214]:


def trainModel(degree, _dataset): #returns the mean squared error for that model.
    ratings = [d['rating'] for d in _dataset]
    y = ratings
    features = [feature(d,degree) for d in _dataset]
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(features, y)
    return float(sklearn.metrics.mean_squared_error(y,model.predict(features)))

mses = [trainModel(d,dataset) for d in [1,2,3,4,5]]


# In[215]:


answers['Q3'] = mses


# In[216]:


assertFloatList(answers['Q3'], 5)# List of length 5


# In[217]:


### Question 4


# In[218]:


training = dataset[:len(dataset)//2]
test = dataset[len(dataset)//2:]

mses = []
def trainModel2(degree, training_dataset, test_dataset): #returns the mean squared error for that model.
    training_ratings = [d['rating'] for d in training_dataset]
    training_features = [feature(d,degree) for d in training_dataset]
    test_ratings = [d['rating'] for d in test_dataset]
    test_features = [feature(d,degree) for d in test_dataset]
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(training_features, training_ratings)
    return float(sklearn.metrics.mean_squared_error(test_ratings,model.predict(test_features)))

mses = [trainModel2(d,training, test) for d in [1,2,3,4,5]]



# In[219]:


answers['Q4'] = mses


# In[220]:


assertFloatList(answers['Q4'], 5)


# In[221]:


### Question 5


# In[222]:


ratings = [d['rating'] for d in test]
y = ratings
features = [[1] for d in test]
model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(features, y)
float(model.coef_[0])
mae = float(sklearn.metrics.mean_absolute_error(y,model.predict(features)))


# In[223]:


answers['Q5'] = mae


# In[224]:


assertFloat(answers['Q5'])


# In[225]:


### Question 6


# In[226]:


f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))


# In[227]:


def feature(datum):
    # your implementation
    return datum['review/text'].count("!")


# In[228]:


isFemale = [d['user/gender'] == "Female" for d in dataset]
exclamationMarks = [feature(d) for d in dataset]
X = [[m] for m in exclamationMarks]
y = isFemale
model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
model.fit(X, y)

prediction = model.predict(X)
truePositives = 0
falsePositives = 0
trueNegatives = 0
falseNegatives = 0
for i in range(len(isFemale)):
    if prediction[i] == True:
        if isFemale[i] == True:
            truePositives+=1
        else:
            falsePositives+=1
    else:
        if isFemale[i] == True:
            falseNegatives+=1
        else:
            trueNegatives+=1
(TP,TN,FP,FN,BER) = (truePositives, trueNegatives, falsePositives, falseNegatives, 1 - 0.5*(truePositives / (truePositives + falseNegatives) + trueNegatives / (trueNegatives + falsePositives)))
(TP,TN,FP,FN,BER)


# In[229]:


answers['Q6'] = [TP, TN, FP, FN, BER]


# In[230]:


assertFloatList(answers['Q6'], 5)


# In[231]:


### Question 7


# In[232]:


isFemale = [d['user/gender'] == "Female" for d in dataset]
exclamationMarks = [feature(d) for d in dataset]
X = [[m] for m in exclamationMarks]
y = isFemale
model = sklearn.linear_model.LogisticRegression(fit_intercept=True, class_weight='balanced')
model.fit(X, y)

prediction = model.predict(X)
truePositives = 0
falsePositives = 0
trueNegatives = 0
falseNegatives = 0
for i in range(len(isFemale)):
    if prediction[i] == True:
        if isFemale[i] == True:
            truePositives+=1
        else:
            falsePositives+=1
    else:
        if isFemale[i] == True:
            falseNegatives+=1
        else:
            trueNegatives+=1
(TP,TN,FP,FN,BER) = (truePositives, trueNegatives, falsePositives, falseNegatives, 1 - 0.5*(truePositives / (truePositives + falseNegatives) + trueNegatives / (trueNegatives + falsePositives)))
(TP,TN,FP,FN,BER)


# In[233]:


answers["Q7"] = [TP, TN, FP, FN, BER]


# In[234]:


assertFloatList(answers['Q7'], 5)


# In[235]:


### Question 8


# In[236]:


def presisionAtK(scoresLabels, k):
    count = 0
    topK = scoresLabels[:k]
    for i in topK:
        if i[1] == True:
           count+=1
    return count/k


# In[247]:


confidenceScores = model.decision_function(X)
scoresLabels = list(zip(confidenceScores, isFemale))
scoresLabels.sort(reverse=True)
scoresLabels[10000-1]


# In[241]:


precisionList = [presisionAtK(scoresLabels,k) for k in [1,10,100,1000,10000]]


# In[242]:


answers['Q8'] = precisionList


# In[243]:


assertFloatList(answers['Q8'], 5) #List of five floats


# In[244]:


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()

