rm(list=ls())
library(data.table)
library(ggplot2)
library(translateR)

user= "elineb"
j = ifelse(Sys.info()[1]=='Windows','J:','/home/j')
dir = paste0(j, '/Project/Evaluation/GF/')
code_loc = ifelse(Sys.info()[1]=='Windows', paste0("C:/Users/", user, "/Documents/gf/"), paste0('/homes/', user, '/gf/'))

source(paste0(code_loc, 'resource_tracking/prep/_common/set_up_r.R'), encoding = "UTF-8")
training = fread(paste0(mapping_dir, "nlp/nlp_training_handcoded_all.csv"))
all_interventions = fread(paste0(mapping_dir, "all_interventions.csv"))
budgets = readRDS(paste0(combined_output_dir, "budget_pudr_iterations.rds"))
budgets = budgets[data_source=="fpm"]

#-------------------------------------------------------------------------------
# RUN BASIC STATISTICS ON THE TRAINING DATA
#-------------------------------------------------------------------------------
code_counts = training[, .(code_count=.N), by=c('code')][order(code_count)]
count_plot = ggplot(code_counts, aes(x=reorder(code, code_count), y=code_count, fill=code_count)) + geom_bar(stat="identity")+
  theme_bw() + labs(x="Code", y="Number of observations in each dataset", title="Count of unique codes in full training data") + 
  scale_y_discrete(breaks=c(0, 45, by=5))
count_plot

ggsave(paste0(mapping_dir, "nlp/analysis/code_count.png"), count_plot)

code_counts2 = training[, .(code_count=.N), by=c('code', 'iteration')][order(code_count)]
count_plot2 = ggplot(code_counts2[iteration=='iteration1'], aes(x=reorder(code, code_count), y=code_count, fill=code_count)) + geom_bar(stat="identity")+
  theme_bw() + labs(x="Code", y="Number of observations in each dataset", title="Count of unique codes in initial training data") + 
  scale_y_discrete(breaks=c(0, 45, by=5))
count_plot2

ggsave(paste0(mapping_dir, "nlp/analysis/code_count2.png"), count_plot2)

count_plot3 = ggplot(code_counts2, aes(x=reorder(code, code_count), y=code_count, fill=code_count)) + geom_bar(stat="identity")+
  theme_bw() + labs(x="Code", y="Number of observations in each dataset", title="Count of unique codes in initial training data") + 
 facet_wrap(~iteration)
count_plot3

ggsave(paste0(mapping_dir, "nlp/analysis/code_count3.png"), count_plot3)

print(code_counts[code_count>15]) #There are 12 codes that have greater than 15 observations. 
print("Percentage of codes have 10 or less observations")
print(nrow(code_counts[code_count<=10])/nrow(code_counts))
print("Percentage of codes that have 5 or less observations")
print(nrow(code_counts[code_count<=5])/nrow(code_counts))

print(code_counts2[code_count>15]) #There are 12 codes that have greater than 15 observations. 
print("Percentage of codes have 10 or less observations")
print(nrow(code_counts2[code_count<=10])/nrow(code_counts2))
print("Percentage of codes that have 5 or less observations")
print(nrow(code_counts2[code_count<=5])/nrow(code_counts2))
print("Percentage of codes that have 1 observation")
print(nrow(code_counts2[code_count==1])/nrow(code_counts2))


print(training[, .(obs=.N), by='iteration'][order(iteration)]) #Find the number of counts you've pulled for each iteration. 

training[, .N, by='disease']
training[, .N, by='lang']
#-------------------------------------------------------------------------------
# ARE ALL CODES REPRESENTED IN THE TRAINING DATA, AND WHAT IS THE DISTRIBUTION? 
#-------------------------------------------------------------------------------
unique(training$code)
unique(all_interventions$code)

print("These module/intervention pairs are currently not in the training data at all.")
unique(all_interventions[!code%in%training$code, .(module_eng, intervention_eng, disease, code)][order(code)])

#-------------------------------------------------------------------------------
# DOES THE MODULE/INTERVENTION IN THE TRAINING DATA REPRESENT THE ORIGINAL FILE, 
# OR HAS IT BEEN RUN THROUGH A VERSION OF OUR INTERNAL MAP? 
#-------------------------------------------------------------------------------
training_mods = sort(unique(training$module))
all_mods = unique(tolower(all_interventions$module_eng))
all_mods = gsub(" |/|-|,", "", all_mods)
all_mods = sort(all_mods)

training_mods[!training_mods%in%all_mods] #Every single one of the modules in the training data matches the 

#-------------------------------------------------------------------------------
# HOW BIG IS YOUR SAMPLE RELATIVE TO THE FINAL BUDGETS IN 2018-2020? 
# ARE THERE CODES IN FINAL BUDGETS THAT WE DON'T HAVE IN TRAINING DATA? 
#-------------------------------------------------------------------------------
nrow(budgets) #There are 36,768 observations in the final 2018-2020 budgets. 
nrow(training)/nrow(budgets) #Our training dataset contains 3.1% of this file. 
nrow(unique(training))/nrow(budgets)

#-------------------------------------------------------------------------------
# EXAMINE UNIQUE ACTIVITY DESCRIPTIONS - WILL WE BE ABLE TO COMPARE BETWEEN THEM?
#-------------------------------------------------------------------------------
fr = unique(budgets[language=='fr', .(activity_description, loc_name)])
esp = unique(budgets[language=='esp', .(activity_description, loc_name)])
eng = unique(budgets[language=='eng', .(activity_description, loc_name)])

write.xlsx(fr, paste0(mapping_dir, "nlp/analysis/french_budget_activities.xlsx"))
write.xlsx(esp, paste0(mapping_dir, "nlp/analysis/spanish_budget_activities.xlsx"))
write.xlsx(eng, paste0(mapping_dir, "nlp/analysis/english_budget_activities.xlsx"))

fr_trans = read.xlsx(paste0(mapping_dir, "nlp/analysis/french_budget_activities_trans.xlsx"))
esp_trans = read.xlsx(paste0(mapping_dir, "nlp/analysis/spanish_budget_activities_trans.xlsx"))

budgets_trans = rbind(fr_trans, esp_trans, eng)

count_act_country = unique(budgets_trans[, .(.N), by=c('activity_description', 'loc_name')])
act_uga = unique(budgets_trans[loc_name=="uga", .(loc_name, activity_description)])
act_cod = unique(budgets_trans[loc_name=="cod", .(loc_name, activity_description)])
act_gtm = unique(budgets_trans[loc_name=="gtm", .(loc_name, activity_description)])

merge_uga_cod = merge(act_uga, act_cod, by='activity_description')
merge_gtm_cod = merge(act_gtm, act_cod, by='activity_description')
merge_uga_gtm = merge(act_gtm, act_uga, by='activity_description')


