# -*- coding: utf-8 -*-
"""
AUTHOR: Emily Linebarger 
PURPOSE: Select a machine learning model to fit module/intervention/code from an activity description 
    for Global Fund budgets and PU/DRs. 
"""

#Import your libraries 
import nltk
import sys
import os
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

#---------------------------------------------------------
# To-do list for this code: 
#   import actual training data split by language 
#   verify that you have the same model fitting for all languages
#  
#---------------------------------------------------------
 
#---------------------------------------------------------
# Load in your data 
#---------------------------------------------------------

#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate. 
#Forget about RSSH? 
all_rt_files = pandas.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/budget_pudr_iterations.csv", encoding = "latin-1") #Right now, feed it all raw files in one language to test code. Will eventually want to feed validated training data. 
all_rt_files = all_rt_files[['sda_activity', 'abbrev_intervention', 'abbrev_module', 'code', 'country']]

french_data = pandas.read_csv("J:/Project/Evaluation/GF/resource_tracking/cod/prepped/budget_pudr_iterations.csv", encoding = "latin-1")
french_data = french_data[french_data.grant_period == "2018-2020"] #These dates are questionable. How can we just get to a cleaned training dataset? 
french_data = french_data[['abbrev_module', 'abbrev_intervention', 'code', 'activity_description']]
french_data = french_data[french_data.activity_description != "All"]

print(french_data.shape) # 17,201 rows of data here. 

dataset = french_data #Make it easy to switch between languages.

#Remove all rows with NA in activity description. We won't be able to classify these by this method. 
print(dataset.size)
dataset = dataset.dropna(subset=['activity_description'])
print(dataset.size)

#-------------------------------------------------------------------------
# Common natural language processing prep before running machine learning 
#-------------------------------------------------------------------------
activities = list(dataset['activity_description'])

#Fix acronyms and diacritical marks (?)

#Remove punctuation 

#Remove numbers 

#Tokenize - split strings into sentences or words 

#Remove common words (is, the, a) that don't give phrase any meaning 
stopWords = list(stopwords.words('french')) #This is a built-in list, but you could reformat. 
print(stopWords)

#Tokenize - split strings into sentences or words and vectorize data (store words as numbers so it's easier for a computer to understand) using Bag-Of-Words 
vectorizer = CountVectorizer()
X = vectorizer.fit(activities) 

#What do your vocabulary and dictionary look like? 
print(vectorizer.vocabulary_) 
print(vectorizer.get_feature_names())

#For each row in activity description, create a vector and append it to the dataset. 
activity_vectors = []
for activity in activities:
    print(activity)
    vector = vectorizer.transform([activity])
    vector = vector.toarray()
    activity_vectors.append(vector)

#-------------------------------------------------------------------------
# From vectorized data, run some analysis to make sure it's still comparable
#   to training data, and store dictionary for reference.     
#-------------------------------------------------------------------------
#Store the dictionary 
dictionary = vectorizer.get_feature_names()
activity_df = pandas.DataFrame(numpy.array(activity_vectors).reshape(17127, 1992), columns=dictionary) #No magiv numbers here! Program in these shape variables. 

#-------------------------------------------------------------------------
# Test different machine learning models using 5-fold cross-validation. 
#   Pick the model with the highest accuracy. 
#-------------------------------------------------------------------------
#Separate out a validation dataset
X = activity_df.values #These are your independent variables
Y_array = dataset.values
Y = Y_array[:,2] #These are your dependent variables 
validation_size = 0.20 #Save 20% of your data for validation
seed = 7 #Set a random seed of 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7 #Pick a random seed. We'll want to reset this every time to make sure the data is always split in the best way. 
scoring = 'accuracy' #We want to pick the model that's the most accurate. 

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=5, random_state=seed) #Set up 5-fold cross-validation with n_splits here. 
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    

#-----------------------------------------------------------------------------------
# Which model did you end up picking? - Decision Tree Classifier has 95.1% accuracy. 
#   Run a confusion matrix, accuracy score, and classification score to see what 
#   we can improve. 
#------------------------------------------------------------------------------------
#Run a confusion matrix to see what types of errors the model is creating! Anything off of the diagonal we need to verify. 
#This also shows the accuracy score, and the classification report. 
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))   
    
    
    
 

















   
#--------------------------------------
#
# EXAMPLE CODE TO PULL FROM BELOW. 
#
#--------------------------------------

#word_tokens = word_tokenize(test_activity) 
#  
#filtered_sentence = [] 
#  
#for w in word_tokens: 
#    if w not in stop_words: 
#        filtered_sentence.append(w) 
#  
#print(word_tokens) 
#print(filtered_sentence) 

#3. Remove acronyms 
    
#Resources 
#Example, worked from https://machinelearningmastery.com/make-predictions-scikit-learn/
#This link on what a confusion matrix is is also very helpful! https://machinelearningmastery.com/make-predictions-scikit-learn/
#Information on Bag-of-words https://medium.freecodecamp.org/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04


#
##Import the iris dataset 
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pandas.read_csv(url, names=names)
#
##Step 2. Separate out a validation dataset. 
##Create an array of your data, and split it into X and Y variables. 
#array = dataset.values
#X = array[:,0:4] #These are your independent variables 
#Y = array[:,4] #These are your dependent variables 
#validation_size = 0.20 #Save 20% of your data for validation
#seed = 7 #Set a random seed of 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#
#seed = 7 #Pick a random seed. We'll want to reset this every time to make sure the data is always split in the best way. 
#scoring = 'accuracy' #We want to pick the model that's the most accurate. 
#
##Step 4. Test several different machine learning models, and pick the one with the best validity. 
#models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
## evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed) #Set up 10-fold cross-validation with n_splits here. 
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#
#
##Run a confusion matrix to see what types of errors the model is creating! Anything off of the diagonal we need to verify. 
##This also shows the accuracy score, and the classification report. 
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)
#predictions = knn.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#
#
#
#
#
#from sklearn.feature_extraction.text import CountVectorizer
## list of text documents
#text = ["The quick brown fox jumped over the lazy dog."]
## create the transform
#vectorizer = CountVectorizer()
## tokenize and build vocab
#vectorizer.fit(text)
## summarize
#print(vectorizer.vocabulary_)
## encode document
#vector = vectorizer.transform(text)
## summarize encoded vector
#print(vector.shape)
#print(type(vector))
#print(vector.toarray())