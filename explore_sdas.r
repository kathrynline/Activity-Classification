# ----------------------------------------------
# AUTHOR: Emily Linebarger 
# PURPOSE: Comparing GOS data with training data for NLP  
# DATE: Last updated March 2019
# ----------------------------------------------
rm(list=ls())

user = "elineb" #Change to your username 
code_dir = ifelse(Sys.info()[1]=='Windows', paste0("C:/Users/", user, "/Documents/gf/"), paste0('/homes/', user, '/gf/'))
source(paste0(code_dir, "resource_tracking/prep/_common/set_up_r.R"), encoding="UTF-8")
nlp_dir = paste0(dir, "modular_framework_mapping/nlp/")
#--------------------------------------------------------
# Read in datasets 
#--------------------------------------------------------
nlp_training = fread(paste0(nlp_dir, "nlp_training_handcoded_all.csv"), encoding="Latin-1")
final_map = fread(paste0(mapping_dir, "gf_mapping.csv"))
final_map = final_map[, .(code, module, intervention, disease, coefficient)]

#We have a lot of activity descriptions here that are encoded correctly - these output files need to be saved in "latin-1". 
gms_data <- data.table(read_excel(paste0(gos_raw, 'Expenditures from GMS and GOS for selected countries.xlsx'),
                                              sheet=as.character('GMS SDAs - extract')))
names(gms_data) = tolower(names(gms_data))
setnames(gms_data, old=c('grant number', 'program start date', 'program end date', 'financial reporting period start date', 'financial reporting period end date',
                         'service delivery area', 'associated standard sda','total budget amount (usd equ)', 'total expenditure amount (usd equ)'),
         new=c('grant_number', 'program_start_date', 'program_end_date', 'period_start_date', 'period_end_date', 'service_delivery_area', 'standard_sda', 'budget', 'expenditure'))
#--------------------------------------------------------------------
# Explore the SDAs that are in the GOS. 
#--------------------------------------------------------------------

unique(gms_data$service_delivery_area) #211 observations for these 8 countries 

#--------------------------------------------------------------------
# Have we already mapped all of the categories in GOS using our map? 
#--------------------------------------------------------------------
service_delivery_areas = gms_data$standard_sda
service_delivery_areas = tolower(service_delivery_areas)
service_delivery_areas = gsub("[[:punct:]]", "", service_delivery_areas)
service_delivery_areas

service_delivery_areas = replace_eng_acronyms(service_delivery_areas)
unique(service_delivery_areas)
#[!service_delivery_areas%in%final_map$module]) #They're not all mapped yet, but we can probably make a


replace_eng_acronyms = function(x) {
  x = gsub(' fortgs ', 'for transgender people', x)
  x = gsub(' pwid ', 'people who inject drugs', x)
  x = gsub(' msm ', 'men who have sex with men', x)
  x = gsub(' stis ', 'sexually transmitted infections', x)
  x = gsub(' acsm ', 'advocacy communication and social mobilization', x)
  x = gsub(' me ', 'monitoring and evaluation', x)
  x = gsub(' itns ', 'insecticide treated nets', x)
  x = gsub(' hss ', 'health system strengthening', x)
  x = gsub(' cbtc ', 'community tb care', x)
  x = gsub(' dots ', 'directly observed therapy short course', x)
  x = gsub(' tb ', 'tuberculosis', x)
  x = gsub(' mdrtb ', 'multidrug resistant tuberculosis', x)
  x = gsub(' css ', 'community responses and systems', x)
  
  return(x)
}

#What are the chances of 1:m matches between activity description and all budget files? 
all_budgets = readRDS(paste0(combined_output_dir, "budget_pudr_iterations.rds"))
#dups = all_budgets[duplicated(code), .(code, gf_module, gf_intervention, orig_module, orig_intervention, cost_category, activity_description)]
dups = all_budgets[coefficient ==1]
dups = unique(dups[, .(code, activity_description)])

dups[duplicated(activity_description), dup:=1]
dups = dups[dup==1]

dups = merge(dups, all_budgets, all.x = TRUE)
dups = dups[, .(code, orig_module, orig_intervention, activity_description, gf_module, gf_intervention)]
dups = unique(dups)
