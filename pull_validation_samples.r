

#rm(list=ls())

library(data.table)

nlp <- fread("J:/Project/Evaluation/GF/resource_tracking/multi_country/mapping/nlp_data/model_outputs/iteration5/test_output.csv")

#Pull observations with low-confidence 
low_confidence <- nlp[confidence_translated <.25]
low_confidence$rand <- runif(nrow(low_confidence), 1, 100)
low_confidence = low_confidence[order(rand)]
low_confidence = low_confidence[, .(module, intervention, gf_module, gf_intervention, disease_lang_concat, predicted_3_module_lang, 
                                    predicted_3_intervention_lang, confidence_3_lang, predicted_3_module_translated, predicted_3_intervention_translated, 
                                    confidence_3_translated, predicted_module_lang, predicted_intervention_lang, rand)]

fr_low_confidence <- low_confidence[disease_lang_concat %in% c('malariafr', 'hivfr', 'tbfr')]
esp_low_confidence <- low_confidence[disease_lang_concat %in% c('malariaesp', 'hivesp', 'tbesp')]
eng_low_confidence <- low_confidence[disease_lang_concat %in% c('malariaeng', 'hiveng', 'tbeng')]

fr_low_confidence <- fr_low_confidence[gf_intervention != predicted_intervention_lang] # N = 272
esp_low_confidence <- esp_low_confidence[gf_intervention != predicted_intervention_lang] # N = 272
eng_low_confidence <- eng_low_confidence[gf_intervention != predicted_intervention_lang] # N = 128

#Pull observations with medium confidence
med_confidence <- nlp[confidence_translated >=.25 & confidence_translated <=.75]
med_confidence$rand <- runif(nrow(med_confidence), 1, 100)
med_confidence = med_confidence[order(rand)]
med_confidence = med_confidence[, .(module, intervention, gf_module, gf_intervention, disease_lang_concat, predicted_3_module_lang, 
                                    predicted_3_intervention_lang, confidence_3_lang, predicted_3_module_translated, predicted_3_intervention_translated, 
                                    confidence_3_translated, predicted_module_lang, predicted_intervention_lang, rand)]

fr_med_confidence <- med_confidence[disease_lang_concat %in% c('malariafr', 'hivfr', 'tbfr')]
esp_med_confidence <- med_confidence[disease_lang_concat %in% c('malariaesp', 'hivesp', 'tbesp')]
eng_med_confidence <- med_confidence[disease_lang_concat %in% c('malariaeng', 'hiveng', 'tbeng')]

fr_med_confidence <- fr_med_confidence[gf_intervention != predicted_intervention_lang] # N = 2071
esp_med_confidence <- esp_med_confidence[gf_intervention != predicted_intervention_lang] # N = 1709
eng_med_confidence <- eng_med_confidence[gf_intervention != predicted_intervention_lang] # N = 540


#For french and spanish, pulling 50 observations from low-end of spectrum 
fr_sample = rbind(fr_low_confidence[1:30, ], fr_med_confidence[1:20, ])
esp_sample = rbind(esp_low_confidence[1:30, ], esp_med_confidence[1:20, ])
eng_sample = rbind(eng_low_confidence[1:30, ], eng_med_confidence[1:20, ])

write.csv(fr_sample, "C:/Users/elineb/Desktop/NLP Validation/nlp_french_12.20.18.csv", row.names = FALSE)
write.csv(esp_sample, "C:/Users/elineb/Desktop/NLP Validation/nlp_spanish_12.20.18.csv", row.names = FALSE)
write.csv(eng_sample, "C:/Users/elineb/Desktop/NLP Validation/nlp_english_12.20.18.csv", row.names = FALSE)



#Overall how is the model doing so far? 
