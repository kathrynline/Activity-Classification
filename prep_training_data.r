rm(list=ls())
library(data.table)

#------------------------------------------------------------------
# 1. GF Budgets and PU/DRs after 2016, all languages combined
#------------------------------------------------------------------ 
#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate. 
#training_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/budget_pudr_iterations.csv", encoding = "latin-1")
raw_cod_data = read_csv("J:/Project/Evaluation/GF/resource_tracking/cod/prepped/raw_bound_gf_files.csv", encoding = "latin-1")
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


#Analysis of current iteration of 
iteration5 = fread("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration5/iteration5.csv")
iteration5 = iteration5[order(-confidence_translated)]
