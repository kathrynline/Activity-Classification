# -*- coding: utf-8 -*-
"""
AUTHOR: Emily Linebarger 
PURPOSE: Select a machine learning model to fit module/intervention/code from an activity description 
    for Global Fund budgets and PU/DRs. 
DATE: February 2019 
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
import string

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
#   Add RSSH to training dataset!! There are no RSSH observations tagged in here. 
#   Add disease as an independent variable in the model. 
#   Run some descriptive statistics on the types and counts of codes that are in the training data right now. 
#---------------------------------------------------------
 
#---------------------------------------------------------
# Load in the training data, and reformat 
#---------------------------------------------------------

#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate. 
training_data = pandas.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample.csv", encoding = "latin-1")
training_data = training_data[['gf_module', 'gf_intervention', 'sda_activity', 'disease_lang_concat']]

#Separate out the language and disease variables 
training_data.disease_lang_concat = training_data.disease_lang_concat.astype(str)
training_data['disease']= training_data.disease_lang_concat.str.slice(0, 1)
training_data['disease']= training_data['disease'].map({'h': 'hiv', 't': 'tb', 'm':'malaria', 'r':'rssh'})

training_data['language']= training_data.disease_lang_concat.str[-1:]
training_data['language']= training_data['language'].map({'p': 'spanish', 'g': 'english', 'r':'french'})

#Pull in code using module, intervention, and disease 
codes = pandas.read_csv("J:/Project/Evaluation/GF/mapping/multi_country/intervention_categories/all_interventions.csv", encoding = "latin-1")
codes = codes.rename(index = str, columns={"Module":"gf_module", "Intervention":"gf_intervention", "Code":"code"})

#Correct data before trying to merge. 
rssh_mods = ['Health management information system and monitoring and evaluation', 'Community responses and systems', 'Human resources for health, including community health workers', 
                               'Procurement and supply chain management systems']
for mod in rssh_mods:
    training_data.loc[training_data.gf_module == mod, 'disease'] = 'rssh'

#Make everything lowercase, remove punctuation, and remove spaces for merge 
translator = str.maketrans('', '', string.punctuation)
training_data['gf_module'] = training_data['gf_module'].str.lower()
training_data['gf_intervention'] = training_data['gf_intervention'].str.lower()
training_data['gf_module'] = training_data['gf_module'].str.replace(" ", "")
training_data['gf_intervention'] = training_data['gf_intervention'].str.replace(" ", "")
training_data['gf_module'] = training_data['gf_module'].str.translate(translator)
training_data['gf_intervention'] = training_data['gf_intervention'].str.translate(translator)

codes['gf_module'] = codes['gf_module'].str.lower()
codes['gf_intervention'] = codes['gf_intervention'].str.lower()
codes['gf_module'] = codes['gf_module'].str.replace(" ", "")
codes['gf_intervention'] = codes['gf_intervention'].str.replace(" ", "")
codes['gf_module'] = codes['gf_module'].str.translate(translator)
codes['gf_intervention'] = codes['gf_intervention'].str.translate(translator)

print(training_data.head())
print(codes.head())

training_data_merge = pandas.merge(training_data, codes, on=['gf_module', 'gf_intervention', 'disease'], how='outer')
training_data_merge = training_data_merge.rename(index=str, columns = {'sda_activity':'activity_description'})
training_data_merge.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample_feb2019.csv")

#---------------------------------------------------------
# #Work on validating two error cases - where we have codes 
#   we'd like our model to map to but they aren't in the training data, and 
#   where we're not getting a full merge between the training data module/intervention 
#   and the codes file. 
#---------------------------------------------------------
codes_only = training_data_merge[training_data_merge.activity_description.isnull()]
data_only = training_data_merge[training_data_merge.code.isnull()]


#french_data = pandas.read_csv("J:/Project/Evaluation/GF/resource_tracking/cod/prepped/budget_pudr_iterations.csv", encoding = "latin-1")
#french_data = french_data[french_data.grant_period == "2018-2020"] #These dates are questionable. How can we just get to a cleaned training dataset? 
french_data = training_data_merge.loc[training_data_merge['language'] == 'french']
french_data = french_data[['gf_module', 'gf_intervention', 'code', 'activity_description']] #Do we also want to make disease an independent variable here? 
french_data = french_data[french_data.activity_description != "All"]

print(french_data.shape) # 147 rows of data here. 

dataset = french_data #Make it easy to switch between languages.

#Remove all rows with NA in activity description. We won't be able to classify these by this method. 
print(dataset.shape)
dataset = dataset.dropna(subset=['activity_description'])
print(dataset.shape)
dataset = dataset.dropna(subset=['code'])
print(dataset.shape)

#-------------------------------------------------------------------------
# Common natural language processing prep before running machine learning 
#-------------------------------------------------------------------------
activities = list(dataset['activity_description'])

#Fix acronyms and diacritical marks (?)

#Remove numbers and punctuation
new_activities = []
translator1 = str.maketrans('', '', string.digits)
translator2 = str.maketrans('', '', string.punctuation)
for activity in activities:
    activity = activity.translate(translator1)
    activity = activity.translate(translator2)
    print(activity)
    new_activities.append(activity)

activities = new_activities 

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
print(numpy.array(activity_vectors).shape) #Assign your reshape with these dimensions. 
activity_vector_arr = numpy.array(activity_vectors)
activity_df = pandas.DataFrame(activity_vector_arr.reshape(activity_vector_arr.shape[0], activity_vector_arr.shape[2]), columns=dictionary) #No magiv numbers here! Program in these shape variables. 

#-------------------------------------------------------------------------
# Test different machine learning models using 5-fold cross-validation. 
#   Pick the model with the highest accuracy. 
#-------------------------------------------------------------------------
#Separate out a validation dataset
array = dataset.values
X = activity_df.values #These are your independent variables
Y = array[:,2] #These are your dependent variables 
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
# Which model did you end up picking? - set as the 'model' variable below. 
#   Run a confusion matrix, accuracy score, and classification score to see what 
#   we can improve. 
#------------------------------------------------------------------------------------
#Run a confusion matrix to see what types of errors the model is creating! Anything off of the diagonal we need to verify. 
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
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