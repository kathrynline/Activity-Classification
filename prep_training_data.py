# -*- coding: utf-8 -*-
"""
AUTHOR: Emily Linebarger 
PURPOSE: Prep different training datasets to feed into machine learning models for activity classification. 
DATE: February 2019 
"""

import pandas as pd
import string
import sys
import os

if sys.platform == "win32":
    j = "J:/"
else:
    j = "homes/j/"
    
os.chdir(j + "Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data")

# TO DO LIST FOR THIS CODE: 
# - Remap post modular-framework data using new map 

#------------------------------------------------------------------
# 1. GF Budgets and PU/DRs after 2016, all languages combined
#------------------------------------------------------------------ 
#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate. 
#training_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/budget_pudr_iterations.csv", encoding = "latin-1")
raw_cod_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/cod/prepped/raw_bound_gf_files.csv", encoding = "latin-1")
raw_gtm_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/gtm/prepped/raw_bound_gf_files.csv", encoding = "latin-1")
raw_uga_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/uga/prepped/raw_bound_gf_files.csv", encoding = "latin-1")

training_data = raw_cod_data.append(raw_gtm_data)
training_data = training_data.append(raw_uga_data)

#Subset to only files that have already been formatted in the modular framework 
cod_filelist = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/cod/grants/cod_budget_filelist.csv", encoding = "latin-1")
gtm_filelist = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/gtm/grants/gtm_budget_filelist.csv", encoding = "latin-1")
uga_filelist = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/uga/grants/uga_budget_filelist.csv", encoding = "latin-1")

cod_filelist = cod_filelist.loc[cod_filelist['mod_framework_format'] == True, ['file_name']]
gtm_filelist = gtm_filelist.loc[gtm_filelist['mod_framework_format'] == True, ['file_name']]
uga_filelist = uga_filelist.loc[uga_filelist['mod_framework_format'] == True, ['file_name']]

mf_files = cod_filelist.append(gtm_filelist)
mf_files = mf_files.append(uga_filelist)

training_data = training_data[training_data['fileName'].isin(mf_files.file_name)]
print(training_data.fileName.unique()) # - Verify that all of these files follow the modular framework, and they should all 

#Remap these data with new post-2017 map
translator = str.maketrans('', '', string.punctuation)
training_data['module'] = training_data['module'].str.lower()
training_data['intervention'] = training_data['intervention'].str.lower()
training_data['module'] = training_data['module'].str.replace(" ", "")
training_data['intervention'] = training_data['intervention'].str.replace(" ", "")
training_data['module'] = training_data['module'].str.translate(translator)
training_data['intervention'] = training_data['intervention'].str.translate(translator)


post_2017_map = pd.read_csv("J:/Project/Evaluation/GF/mapping/multi_country/intervention_categories/post_2017_map.csv", encoding = "latin-1")
training_data = pd.merge(training_data, post_2017_map, on=['module', 'intervention', 'disease'], how='left')

print(training_data.shape)
training_data = training_data[training_data.activity_description != "All"]
training_data = training_data[training_data.activity_description != "all"]
print(training_data.shape) #It's a little weird that we have some rows dropping here. 

training_data.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_all.csv")


#---------------------------------------------------------
# #2. GF Budgets and PU/DRs after 2016, in each language
#---------------------------------------------------------
french_data = training_data[training_data['lang'] == 'fr']
spanish_data = training_data[training_data['lang'] == 'esp']
english_data = training_data[training_data['lang'] == 'eng']

french_data.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_french.csv")
spanish_data.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_spanish.csv")
english_data.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_budgetpudr_english.csv")

#
## What is the representation of all codes in the training data? 
#dataset = spanish_data #Make it easy to switch between languages.
#dataset.code.unique()
#
##---------------------------------------------------------
## Split the data by language, and prep the inputs for the model.
##---------------------------------------------------------
#dataset = spanish_data #Make it easy to switch between languages.
#print(dataset.shape) # 17,201 rows of data here. 
#
##Remove all rows with NA in activity description. We won't be able to classify these by this method. 
#print(dataset.shape) #17,201 french, 2,920 spanish
#dataset = dataset.dropna(subset=['activity_description'])
#print(dataset.shape) #17,127
#dataset = dataset.dropna(subset=['code'])
#print(dataset.shape) #17,127


#------------------------------------------------------
# 3. Training data based on observations we've coded by hand. 
#--------------------------------------------------------
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

iteration5 = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration5/iteration5.csv", encoding = "latin-1")
iteration5 = iteration5[['corrected_module', 'corrected_intervention', 'sda_activity', 'disease_lang_concat']]

#iterations = [iteration2, iteration3, iteration4, iteration5]
#training_data = initial_training.append(iterations, ignore_index = True)

training_data = iteration5

training_data = training_data.rename(index = str, columns={"corrected_module":"gf_module", "corrected_intervention":"gf_intervention"})

#Separate out the language and disease variables 
training_data.disease_lang_concat = training_data.disease_lang_concat.astype(str)
training_data['disease']= training_data.disease_lang_concat.str.slice(0, 1)
training_data['disease']= training_data['disease'].map({'h': 'hiv', 't': 'tb', 'm':'malaria', 'r':'rssh'})

training_data['language']= training_data.disease_lang_concat.str[-1:]
training_data['language']= training_data['language'].map({'p': 'spanish', 'g': 'english', 'r':'french'})

#Pull in code using module, intervention, and disease 
codes = pd.read_csv("J:/Project/Evaluation/GF/mapping/multi_country/intervention_categories/all_interventions.csv", encoding = "latin-1")
codes = codes.rename(index = str, columns={"Module":"gf_module", "Intervention":"gf_intervention", "Code":"code"})

#Correct data before trying to merge. 
rssh_mods = ['Health management information system and monitoring and evaluation', 'Community responses and systems', 'Human resources for health, including community health workers', 
                               'Procurement and supply chain management systems']
for mod in rssh_mods:
    training_data.loc[training_data.gf_module == mod, 'disease'] = 'rssh'

codes['gf_module'] = codes['gf_module'].str.lower()
codes['gf_intervention'] = codes['gf_intervention'].str.lower()
codes['gf_module'] = codes['gf_module'].str.replace(" ", "")
codes['gf_intervention'] = codes['gf_intervention'].str.replace(" ", "")
codes['gf_module'] = codes['gf_module'].str.translate(translator)
codes['gf_intervention'] = codes['gf_intervention'].str.translate(translator)

print(training_data.head())
print(codes.head())

training_data_merge = pd.merge(training_data, codes, on=['gf_module', 'gf_intervention', 'disease'], how='outer')
training_data_merge = training_data_merge.rename(index=str, columns = {'sda_activity':'activity_description'})
training_data_merge.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_handcoded_all.csv")

#
#---------------------------------------------------------
# #Work on validating two error cases - where we have codes 
#   we'd like our model to map to but they aren't in the training data, and 
#   where we're not getting a full merge between the training data module/intervention 
#   and the codes file. 
#---------------------------------------------------------
#codes_only = training_data_merge[training_data_merge.activity_description.isnull()]
#data_only = training_data_merge[training_data_merge.code.isnull()]



#Once we pick a dataset - pick the file you'd like to keep. 
#training_data.to_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/nlp_training_sample_feb2019.csv")
