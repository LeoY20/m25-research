#install.packages(c("usethis", "gitcreds"))
#gitcreds::gitcreds_set()
#install.packages("haven")
#haven for the .dta file

#* MAINTAIN THIS: Want to retain 100% of the survey responses 
#for primary residences and as much as we can for the ResStock data
library(tidyverse)
library(haven)

#read_csv to get the actual data out
az_data <- read_csv("AZ_baseline_metadata_and_annual_results.csv")
resstock <- read_dta("RET_2017_part1.dta")

#removing all the gas/propane dryers
#do you want this to be NULL? or not?
az_data <- az_data |> filter(in.clothes_dryer != "Propane")

#combine electric induction and resist categories
az_data <- az_data |> mutate(in.cooking_range = fct_collapse(in.cooking_range, 
                  Electric = c("Electric Induction", "Electric Resistance")))

unique(az_data$in.cooking_range)
