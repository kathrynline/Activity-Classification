rm(list=ls())
library(data.table)

user= "elineb"
j = ifelse(Sys.info()[1]=='Windows','J:','/home/j')
dir = paste0(j, '/Project/Evaluation/GF/')
code_loc = ifelse(Sys.info()[1]=='Windows', paste0("C:/Users/", user, "/Documents/gf/"), paste0('/homes/', user, '/gf/'))

source(paste0(code_loc, 'resource_tracking/prep/_common/set_up_r.R'), encoding = "UTF-8")

#------------------------------------------------------------------
# 1. GF Budgets and PU/DRs after 2016, all languages combined
#------------------------------------------------------------------ 
#I need the current module map split by language. Can I cleave this off from the data we've already mapped? The most recent files should be the most accurate.
#training_data = pd.read_csv("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/budget_pudr_iterations.csv", encoding = "latin-1")
raw_cod_data = fread(paste0(dir, "_gf_files_gos/cod/prepped_data/raw_bound_gf_files.csv"), encoding = "Latin-1")
raw_gtm_data = fread(paste0(dir, "_gf_files_gos/gtm/prepped_data/raw_bound_gf_files.csv"), encoding = "Latin-1")
raw_uga_data = fread(paste0(dir, "_gf_files_gos/uga/prepped_data/raw_bound_gf_files.csv"), encoding = "Latin-1")

all_raw_data = list(raw_cod_data, raw_gtm_data, raw_uga_data)
training_data = rbindlist(all_raw_data, use.names = TRUE, fill = TRUE)

#Subset to only files that have already been formatted in the modular framework
cod_filelist = fread(paste0(dir, "_gf_files_gos/cod/raw_data/cod_budget_filelist.csv"), encoding = "Latin-1")
gtm_filelist = fread(paste0(dir, "_gf_files_gos/gtm/raw_data/gtm_budget_filelist.csv"), encoding = "Latin-1")
uga_filelist = fread(paste0(dir, "_gf_files_gos/uga/raw_data/uga_budget_filelist.csv"), encoding = "Latin-1")

all_files = list(cod_filelist, gtm_filelist, uga_filelist)
filelist = rbindlist(all_files, use.names = TRUE, fill = TRUE)
filelist = filelist[mod_framework_format == TRUE]

training_data = training_data[fileName%in%filelist$file_name]
print(unique(training_data$fileName)) # - Verify that all of these files follow the modular framework, and they should all

#Remap these data with new post-2017 map
training_data = strip_chars(training_data)
training_data[, module:=fix_diacritics(module)]
training_data[, intervention:=fix_diacritics(intervention)]

training_data = training_data[, .(module, intervention, disease, sda_activity, loc_name, lang)]
training_data[disease == 'tb/hiv', disease:='hiv/tb']

post_2017_map = fread(paste0(mapping_dir, "post_2017_map.csv"), encoding = "Latin-1")
post_2017_map = post_2017_map[, .(module, intervention, disease, code)]

training_data1 = merge(training_data, post_2017_map, by=c('module', 'intervention', 'disease'), all.x = TRUE, allow.cartesian = TRUE)
training_data1 = training_data1[!is.na(module) & !is.na(intervention)]

training_data1[, .N]
training_data1 = training_data1[sda_activity != "All"]
training_data1 = training_data1[sda_activity != "all"]
training_data1[, .N] #It's weird that we have activities of 'all' here. 

check_nas = training_data1[is.na(code), .(module, intervention, disease)]
check_nas = unique(check_nas)

write.csv(training_data1, paste0(mapping_dir, "nlp/nlp_training_budgetpudr_all.csv"), row.names=FALSE)

#---------------------------------------------------------
# #2. GF Budgets and PU/DRs after 2016, in each language
#---------------------------------------------------------
training_data2_fr = training_data1[lang == 'fr']
training_data2_esp = training_data1[lang == 'esp']
training_data2_eng = training_data1[lang == 'eng']

write.csv(training_data2_fr, paste0(mapping_dir, "nlp/nlp_training_budgetpudr_french.csv"), row.names=FALSE)
write.csv(training_data2_esp, paste0(mapping_dir, "nlp/nlp_training_budgetpudr_spanish.csv"), row.names=FALSE)
write.csv(training_data2_eng, paste0(mapping_dir, "nlp/nlp_training_budgetpudr_english.csv"), row.names=FALSE)

#------------------------------------------------------
# 3. Training data based on observations we've coded by hand. 
#--------------------------------------------------------
initial_training = fread(paste0(mapping_dir, "nlp/nlp_training_sample.csv"), encoding = "Latin-1")
initial_training = initial_training[, .(gf_module, gf_intervention, sda_activity, disease_lang_concat)]
setnames(initial_training, old=c('gf_module', 'gf_intervention'), new =c('corrected_module', 'corrected_intervention'))
initial_training$iteration="iteration1"
         
iteration2 = fread(paste0(mapping_dir, "nlp/model_outputs/iteration2/iteration2.csv"), encoding="Latin-1")
iteration2 = iteration2[, .(corrected_module, corrected_intervention, sda_activity, disease_lang_concat)]
iteration2$iteration="iteration2"

iteration3 = fread(paste0(mapping_dir, "nlp/model_outputs/iteration3/iteration3.csv"), encoding = "Latin-1")
iteration3 = iteration3[, .(corrected_module, corrected_intervention, sda_activity, disease_lang_concat)]
iteration3$iteration="iteration3"

iteration4 = fread(paste0(mapping_dir, "nlp/model_outputs/iteration4/iteration4.csv"), encoding = "Latin-1")
iteration4 = iteration4[, .(corrected_module, corrected_intervention, sda_activity, disease_lang_concat)]
iteration4$iteration="iteration4"

# iteration5 = fread(paste0(dir, "resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration5/iteration5.csv"), fileEncoding = "Latin-1")
# iteration5 = iteration5[, .(corrected_module, corrected_intervention, sda_activity, disease_lang_concat)] #Not adding this one because it hasn't been validated yet.

iterations = list(initial_training, iteration2, iteration3, iteration4)
training_data = rbindlist(iterations, use.names = TRUE, fill = TRUE)

setnames(training_data, old=c('corrected_module', 'corrected_intervention'), new=c('module', 'intervention'))
training_data[, .N]

#Separate out the language and disease variables 
training_data[, lang:="english"]
training_data[, lang:=ifelse(grepl("esp", disease_lang_concat), "spanish", lang)]
training_data[, lang:=ifelse(grepl("fr", disease_lang_concat), "french", lang)]

training_data[, disease:='hiv'] #Note that we don't have any RSSH here. 
training_data[, disease:=ifelse(grepl("tb", disease_lang_concat), "tb", disease)]
training_data[, disease:=ifelse(grepl("malaria", disease_lang_concat), "malaria", disease)]

#Pull in code using module, intervention, and disease 
codes = fread(paste0(mapping_dir, "all_interventions.csv"), encoding = "Latin-1")
setnames(codes, old=c('module_eng', 'intervention_eng'), new=c('module', 'intervention'))
codes = codes[, .(module, intervention, code, disease)]

#Merge with clean map of codes 
training_data = strip_chars(training_data)
training_data[, module:=fix_diacritics(module)]
training_data[, intervention:=fix_diacritics(intervention)]

codes = strip_chars(codes)
codes[, module:=fix_diacritics(module)]
codes[, intervention:=fix_diacritics(intervention)]

#Correct data before trying to merge. - can we do this above too? 
rssh_mods = c('healthmanagementinformationsystemandmonitoringandevaluation', 'communityresponsesandsystems', 'humanresourcesforhealthincludingcommunityhealthworkers',
             'procurementandsupplychainmanagementsystems', 'financialmanagementsystems', 'integratedservicedeliveryandqualityimprovement', 'nationalhealthstrategies')

training_data[module%in%rssh_mods, disease:='rssh']

#Edit a few of these because they don't match to the modular framework. 
training_data[module == "communityresponsesandsystems" & intervention == "otherothercommunityresponsesandsystemsinterventions", intervention:="othercommunityresponsesandsystemsinterventions"]
training_data[module == 'multidrugresistanttb' & (intervention == 'communitymdrtbcaredelivery' | intervention == 'treatmentmdrtb'), disease:='tb']
training_data[module == 'programstoreducehumanrightsrelatedbarrierstohivservices' & intervention == 'legalliteracyknowyourrights', disease:='hiv']
training_data[module == 'tbcareandprevention' & intervention == 'keypopulationstbcareandpreventionaaprisoners', intervention:='keypopulationstbcareandpreventionprisoners']

#Merge the data with the codes. 
training_data3 = merge(training_data, codes, by=c('module', 'intervention', 'disease'), all.x = TRUE)
check_nas = training_data3[is.na(code), .(module, intervention, disease)] #These don't map 1:1 to the modular framework? 
check_nas = unique(check_nas)
check_nas #These don't map easily to the modular framework, so leaving them out at this time so the mapping is really clean. 

#"module"       "intervention" "disease"      "sda_activity" "loc_name"     "lang"         "code" 
training_data3 = training_data3[, .(module, intervention, disease, sda_activity, lang, code, iteration)] #We don't have loc_name here - we could guess but let's leave it for now. 
training_data3 = training_data3[!is.na(code)] #Drop these out for now- N = 1
write.csv(training_data3, paste0(mapping_dir, "nlp/nlp_training_handcoded_all.csv"), row.names = FALSE)

#-------------------------------------------------------------------------------------------------------
#Make one final training dataset - budget/pudr data + codes that are only represented in hand-coded data 
#-------------------------------------------------------------------------------------------------------
training_data4 = training_data3[!code%in%training_data1$code]
training_data4 = rbind(training_data4, training_data1, use.names = TRUE, fill = TRUE)

write.csv(training_data4, paste0(mapping_dir, "nlp/budgetpudr_with_extra_codes.csv"), row.names=FALSE)

#------------------------------------------------------
# Review the different training datasets 
#--------------------------------------------------------

#What is the size of each of the datasets? 
all_training = list(training_data1, training_data2_fr, training_data2_esp, training_data2_eng, training_data3, training_data4)
total_codes = length(codes$code)

for(dt in all_training) {
  #What's the size of each dataset? 
  print(paste0("Dataset size: ", dt[, .N]))
  
  #What's the 'coverage' of all codes in the modular framework for each dataset? 
  print(paste0("% of total codes represented in this dataset: ", nrow(unique(dt[, .(code)]))/total_codes))
  print("")
}


#----------------------------------------------------------------
# Review the handcoded training data specifically. 
#----------------------------------------------------------------

gf_map = fread(paste0(mapping_dir, "gf_mapping.csv"))
#How many of these codes got changed from what they were originally? 
merge1 = training_data3[, .(module, intervention, code, disease, sda_activity)]
merge2 = gf_map[, .(module, intervention, code, disease)]

handcoded_merge = merge(merge1, merge2, by=c('module', 'intervention', 'disease'), all.x = TRUE)

relabeled = handcoded_merge[code.x != code.y]
nrow(relabeled)
