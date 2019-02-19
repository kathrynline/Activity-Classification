# -*- coding: utf-8 -*-
"""
AUTHOR: Emily Linebarger 
PURPOSE: Select a machine learning model to fit module/intervention/code from an activity description 
    for Global Fund budgets and PU/DRs. 
DATE: February 2019 
"""

#Import your libraries 
import numpy as np
import matplotlib
import pandas as pd
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
#   verify that you have the same model fitting for all languages
#   Add disease as an independent variable in the model. 
#   Run some descriptive statistics on the types and counts of codes that are in the training data right now. 
#---------------------------------------------------------
 
#---------------------------------------------------------
# Load in the training data, and reformat 
#---------------------------------------------------------

#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate. 
training_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/budget_pudr_iterations.csv", encoding = "latin-1")

#Subset to only files that have already been formatted in the modular framework 
training_data = training_data[training_data['grant_period'].isin(['2018-2020', '2018', '2019-2021', '2019-2022'])]
print(training_data.fileName.unique()) # - Verify that all of these files follow the modular framework, and they should all 
print

training_data = training_data[['code', 'module', 'intervention', 'activity_description', 'disease', 'loc_name', 'lang']]
training_data = training_data.rename(index = str, columns={"module":"gf_module", "intervention":"gf_intervention"})

#Make everything lowercase, remove punctuation, and remove spaces for merge 
translator = str.maketrans('', '', string.punctuation)
training_data['gf_module'] = training_data['gf_module'].str.lower()
training_data['gf_intervention'] = training_data['gf_intervention'].str.lower()
training_data['gf_module'] = training_data['gf_module'].str.replace(" ", "")
training_data['gf_intervention'] = training_data['gf_intervention'].str.replace(" ", "")
training_data['gf_module'] = training_data['gf_module'].str.translate(translator)
training_data['gf_intervention'] = training_data['gf_intervention'].str.translate(translator)

training_data = training_data[training_data.activity_description != "All"]

training_data.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample_feb2019.csv")


#---------------------------------------------------------
# Show some descriptive statistics on the training data. 
#---------------------------------------------------------
french_data = training_data[training_data['lang'] == 'fr']
spanish_data = training_data[training_data['lang'] == 'esp']
english_data = training_data[training_data['lang'] == 'eng']
# What is the representation of all codes in the training data? 

#---------------------------------------------------------
# Split the data by language, and prep the inputs for the model.
#---------------------------------------------------------
dataset = spanish_data #Make it easy to switch between languages.
print(dataset.shape) # 17,201 rows of data here. 

#Remove all rows with NA in activity description. We won't be able to classify these by this method. 
print(dataset.shape) #17,201 french, 2,920 spanish
dataset = dataset.dropna(subset=['activity_description'])
print(dataset.shape) #17,127
dataset = dataset.dropna(subset=['code'])
print(dataset.shape) #17,127

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
print(np.array(activity_vectors).shape) #Assign your reshape with these dimensions. 
activity_vector_arr = np.array(activity_vectors)
activity_df = pd.DataFrame(activity_vector_arr.reshape(activity_vector_arr.shape[0], activity_vector_arr.shape[2]), columns=dictionary) #No magiv numbers here! Program in these shape variables. 
activity_df.shape #Make sure this is exactly the same as the dataset before, and append with original dataset so you can match with disease. 

dataset = dataset.reset_index()
disease_col = dataset.disease
activity_df = activity_df.join(disease_col, lsuffix = "left")
activity_df['disease'] = activity_df['disease'].map({'hiv':1, 'tb':2, 'malaria':3, 'rssh':4}) #Save this encoding for later! 
#-------------------------------------------------------------------------
# Test different machine learning models using 5-fold cross-validation. 
#   Pick the model with the highest accuracy. 
#-------------------------------------------------------------------------
#Separate out a validation dataset
X = activity_df.values #These are your independent variables - disease and the vectorized activity description. 
Y = dataset.values
Y = Y[:,1] #These are your dependent variables - just 'code'. 
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
#models.append(('SVM', SVC(gamma='auto'))) This model is taking forever - run on cluster? 
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
model = DecisionTreeClassifier() #Accuracy on french dataset: 95.5%
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions)) #96.1% accuracy on french data, 98.4% accuracy on English data 
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))   
    
    
    
 


#------------------------------------------------------
# Alternate training data based on observations we coded by hand 
#--------------------------------------------------------
#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate. 
#initial_training = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample.csv", encoding = "latin-1")
#initial_training = initial_training[['gf_module', 'gf_intervention', 'sda_activity', 'disease_lang_concat']]
#initial_training = initial_training.rename(index = str, columns={"gf_module":"corrected_module", "gf_intervention":"corrected_intervention"})
# 
#iteration2 = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration2/iteration2.csv", encoding = "latin-1")
#iteration2 = iteration2[['corrected_module', 'corrected_intervention', 'sda_activity', 'disease_lang_concat']]
#
#iteration3 = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration3/iteration3.csv", encoding = "latin-1")
#iteration3 = iteration3[['corrected_module', 'corrected_intervention', 'sda_activity', 'disease_lang_concat']]
#
#iteration4 = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration4/iteration4.csv", encoding = "latin-1")
#iteration4 = iteration4[['corrected_module', 'corrected_intervention', 'sda_activity', 'disease_lang_concat']]
#
#iteration5 = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration5/iteration5.csv", encoding = "latin-1")
#iteration5 = iteration5[['corrected_module', 'corrected_intervention', 'sda_activity', 'disease_lang_concat']]
#
#iterations = [iteration2, iteration3, iteration4, iteration5]
#training_data = initial_training.append(iterations, ignore_index = True)



#Another alternate training data: 
#training_data = training_data.rename(index = str, columns={"corrected_module":"gf_module", "corrected_intervention":"gf_intervention"})
#
##Separate out the language and disease variables 
#training_data.disease_lang_concat = training_data.disease_lang_concat.astype(str)
#training_data['disease']= training_data.disease_lang_concat.str.slice(0, 1)
#training_data['disease']= training_data['disease'].map({'h': 'hiv', 't': 'tb', 'm':'malaria', 'r':'rssh'})
#
#training_data['language']= training_data.disease_lang_concat.str[-1:]
#training_data['language']= training_data['language'].map({'p': 'spanish', 'g': 'english', 'r':'french'})
#
##Pull in code using module, intervention, and disease 
#codes = pd.read_csv("J:/Project/Evaluation/GF/mapping/multi_country/intervention_categories/all_interventions.csv", encoding = "latin-1")
#codes = codes.rename(index = str, columns={"Module":"gf_module", "Intervention":"gf_intervention", "Code":"code"})
#
##Correct data before trying to merge. 
#rssh_mods = ['Health management information system and monitoring and evaluation', 'Community responses and systems', 'Human resources for health, including community health workers', 
#                               'Procurement and supply chain management systems']
#for mod in rssh_mods:
#    training_data.loc[training_data.gf_module == mod, 'disease'] = 'rssh'
#
#codes['gf_module'] = codes['gf_module'].str.lower()
#codes['gf_intervention'] = codes['gf_intervention'].str.lower()
#codes['gf_module'] = codes['gf_module'].str.replace(" ", "")
#codes['gf_intervention'] = codes['gf_intervention'].str.replace(" ", "")
#codes['gf_module'] = codes['gf_module'].str.translate(translator)
#codes['gf_intervention'] = codes['gf_intervention'].str.translate(translator)
#
#print(training_data.head())
#print(codes.head())
#
#training_data_merge = pd.merge(training_data, codes, on=['gf_module', 'gf_intervention', 'disease'], how='outer')
#training_data_merge = training_data_merge.rename(index=str, columns = {'sda_activity':'activity_description'})
#training_data_merge.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample_feb2019.csv")

#
#---------------------------------------------------------
# #Work on validating two error cases - where we have codes 
#   we'd like our model to map to but they aren't in the training data, and 
#   where we're not getting a full merge between the training data module/intervention 
#   and the codes file. 
#---------------------------------------------------------
#codes_only = training_data_merge[training_data_merge.activity_description.isnull()]
#data_only = training_data_merge[training_data_merge.code.isnull()]









   
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
#dataset = pd.read_csv(url, names=names)
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