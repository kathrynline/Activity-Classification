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
import os
import sys

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

from googletrans import Translator
translator = Translator()
translator.translate('buenos dias', dest='en')
translation = translator.translate('안녕하세요.')
# <Translated src=ko dest=en text=Good evening. pronunciation=Good evening.>
translator.translate('안녕하세요.', dest='ja')
# <Translated src=ko dest=ja text=こんにちは。 pronunciation=Kon'nichiwa.>
translator.translate('veritas lux mea', src='la')
# <Translated src=la dest=en text=The truth is my light pronunciation=The truth is my light>

if sys.platform == "win32":
    j = "J:/"
else:
    j = "homes/j/"
    
os.chdir(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data")
orig_stdout = sys.stdout
f = open('model_testing.txt', 'w')
sys.stdout = f

#---------------------------------------------------------
# To-do list for this code: 
#   verify that you have the same model fitting for all languages
#   Add disease as an independent variable in the model. 
#   Run some descriptive statistics on the types and counts of codes that are in the training data right now. 
#   Make an array of the vectorization of activity description. We really only want this as one variable. 
#---------------------------------------------------------

budgetpudr_all = pd.read_csv(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_all.csv", encoding = "latin-1")
budgetpudr_eng = pd.read_csv(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_english.csv", encoding = "latin-1")
budgetpudr_esp = pd.read_csv(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_spanish.csv", encoding = "latin-1")
budgetpudr_fr = pd.read_csv(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_french.csv", encoding = "latin-1")
handcoded_all = pd.read_csv(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_handcoded_all.csv", encoding = "latin-1")

handcoded_all = handcoded_all.rename(index = str, columns={"gf_module":"abbrev_module", "gf_intervention":"abbrev_intervention"})

#dataset = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample_feb2019.csv")

training_datasets = []
training_datasets.append(("All budgets/pudrs", budgetpudr_all))
training_datasets.append(("Budgets/pudrs English only", budgetpudr_eng))
training_datasets.append(("Budgets/pudrs Spanish only", budgetpudr_esp))
training_datasets.append(("Budgets/pudrs French only", budgetpudr_fr))
training_datasets.append(("Hand-coded data, all languages", handcoded_all))

for label, dataset in training_datasets:
    #-------------------------------------------------------------------------
    # Common natural language processing prep before running machine learning 
    #-------------------------------------------------------------------------
    dataset = dataset[['abbrev_module', 'abbrev_intervention', 'code', 'activity_description', 'disease']]
    dataset = dataset.applymap(str)
    print("Number of observations for dataset " + label + str(dataset.shape))
    print("")
    activities = list(dataset['activity_description'])
    
    #Fix acronyms and diacritical marks (?)
    
    #Remove numbers and punctuation
    new_activities = []
    translator1 = str.maketrans('', '', string.digits)
    translator2 = str.maketrans('', '', string.punctuation)
    for activity in activities:
        activity = activity.translate(translator1)
        activity = activity.translate(translator2)
        #print(activity)
        new_activities.append(activity)
    
    activities = new_activities 
    
    #Remove common words (is, the, a) that don't give phrase any meaning 
    stopWords = list(stopwords.words('french')) #This is a built-in list, but you could reformat. 
    #print(stopWords)
    
    #Tokenize - split strings into sentences or words and vectorize data (store words as numbers so it's easier for a computer to understand) using Bag-Of-Words 
    vectorizer = CountVectorizer()
    X = vectorizer.fit(activities) 
    
    #What do your vocabulary and dictionary look like? 
    #print(vectorizer.vocabulary_) 
    #print(vectorizer.get_feature_names())
    
    #For each row in activity description, create a vector and append it to the dataset. 
    activity_vectors = []
    for activity in activities:
        #print(activity)
        vector = vectorizer.transform([activity])
        vector = vector.toarray()
        activity_vectors.append(vector)
    
    #-------------------------------------------------------------------------
    # From vectorized data, run some analysis to make sure it's still comparable
    #   to training data, and store dictionary for reference.     
    #-------------------------------------------------------------------------
    #Store the dictionary 
    dictionary = vectorizer.get_feature_names()
    #print(np.array(activity_vectors).shape) #Assign your reshape with these dimensions. 
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
    models.append(('SVM', SVC(gamma='auto'))) 
    # evaluate each model in turn
    results = []
    names = []
    print("Results for training dataset " + label)
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=5, random_state=seed) #Set up 5-fold cross-validation with n_splits here. 
    	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
    
    print("")

sys.stdout = orig_stdout
f.close()      

#-----------------------------------------------------------------------------------
# Which model did you end up picking? - set as the 'model' variable below. 
#   Run a confusion matrix, accuracy score, and classification score to see what 
#   we can improve. 
#------------------------------------------------------------------------------------
#Run a confusion matrix to see what types of errors the model is creating! Anything off of the diagonal we need to verify. 
#model = DecisionTreeClassifier() #Accuracy on french dataset: 95.5%
#model.fit(X_train, Y_train)
#predictions = model.predict(X_validation)
#print(accuracy_score(Y_validation, predictions)) #96.1% accuracy on french data, 98.4% accuracy on English data 
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))   
    
    
    





   
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
#models.append(('LR', LogisticRegression(solver='liblinebigar', multi_class='ovr')))
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