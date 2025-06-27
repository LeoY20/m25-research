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
rename

#Setting all propane dryers to null
az_data <- az_data |> mutate(in.clothes_dryer = na_if(in.clothes_dryer, "Propane"))

#combine electric induction and resist categories
az_data <- az_data |> mutate(in.cooking_range = fct_collapse(in.cooking_range, 
                  Electric = c("Electric Induction", "Electric Resistance")))

#changing single-unit to multi-unit
az_data <- az_data |> mutate(in.geometry_building_type_acs = case_when(
                              str_detect(in.geometry_building_type_acs, "Unit") ~ "Multi-Unit",
                              TRUE ~ in.geometry_building_type_acs))

#categories from 1 story, 2 stories, 2+
az_data <- az_data |> mutate(in.geometry_stories = case_when(
                              in.geometry_stories > 2 ~ "2+ Stories",
                              in.geometry_stories == 2 ~ "2 Stories",
                              TRUE ~ "1 Story"))

#Combine categories into Under 1000 sq ft, 1000 - 1499 sq ft, 
#1500 - 1999 sq ft, 2000 - 2999 sq ft, 3000 - 3999 sq ft, 4000 or more sq ft
#question: why are ticks not uniform?
#this works cuz it's like a switch statement, if one is false you know it's false for any below too
#TRUE is the fall through case, it'll always be true at the end.

az_data <- az_data |> mutate(in.geometry_floor_area = case_when(
                            in.geometry_floor_area < 1000 ~ "Under 1000 sq ft",
                            in.geometry_floor_area < 1500 ~ "1000 - 1499 sq ft",
                            in.geometry_floor_area < 2000 ~ "1500 - 1999 sq ft",
                            in.geometry_floor_area < 3000 ~ "2000 - 2999 sq ft",
                            in.geometry_floor_area < 4000 ~ "3000 - 3999 sq ft",
                            TRUE ~ "4000 or more sq ft"))

az_data <- az_data |> mutate(in.heating_fuel = fct_collapse(in.heating_fuel,
                                                Other = c("Propane", "Other Fuel", "None", "Fuel Oil"))) |>
                            mutate(in.heating_fuel = case_when(
                                  in.heating_fuel == "Natural Gas" ~ "Gas",
                                  TRUE ~ in.heating_fuel))















