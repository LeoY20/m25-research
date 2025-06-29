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

#Make categories electric, gas, or other
az_data <- az_data |> mutate(in.heating_fuel = fct_collapse(in.heating_fuel,
                                                Other = c("Propane", "Other Fuel", "None", "Fuel Oil"))) |>
                            mutate(in.heating_fuel = case_when(
                                  in.heating_fuel == "Natural Gas" ~ "Gas",
                                  TRUE ~ in.heating_fuel))

#Keep only integers from SEER values and make everything else zero
az_data <- az_data |> mutate(in.hvac_cooling_efficiency = case_when(
                            str_detect(in.hvac_cooling_efficiency, "\\d$") ~ str_extract(in.hvac_cooling_efficiency, "\\d$"),
                            TRUE ~ as.character(0)))
az_data$in.hvac_cooling_efficiency <- as.numeric(az_data$in.hvac_cooling_efficiency)


#converting in.hvac_heating_type_and_fuel to VHOMEHEAT, VHEATEQUIP, VHOMEHEATV1 
#don't even worry about this
hvac_argsct <- length(unique(az_data$in.hvac_heating_type_and_fuel)) #length
hvac_htf_other <- vector(mode = "character", length = hvac_argsct) #to store all variables
hvac_tftable <- str_detect(unique(az_data$in.hvac_heating_type_and_fuel), "Other Fuel")
for(i in 1:hvac_argsct) {
  if(hvac_tftable[i]) {
    hvac_htf_other[i] = unique(az_data$in.hvac_heating_type_and_fuel)[i]
  }
}
#why does this work? I have no idea
hvac_htf_other <- hvac_htf_other[hvac_htf_other != ""]
hvac_htf_other
hvac_variables <- unique(az_data$in.hvac_heating_type_and_fuel)
hvac_electric <- hvac_variables[!(hvac_variables %in% c("Electricity MSHP", "Electricity ASHP")) &
                              str_detect(hvac_variables, "Electricity")]
hvac_electric <- hvac_electric[hvac_electric != ""]
hvac_natgas <- hvac_variables[str_detect(hvac_variables, "Natural Gas")]
hvac_natgas <- hvac_natgas[hvac_natgas != ""]
hvac_propane <- hvac_variables[str_detect(hvac_variables, "Propane")]
hvac_propane <- hvac_propane[hvac_propane != ""]


az_data <- az_data |> mutate(in.hvac_heating_type_and_fuel = fct_collapse(
                             in.hvac_heating_type_and_fuel, "Electric Heat Pump" = c("Electricity MSHP", "Electricity ASHP"),
                             Other = hvac_htf_other,
                             Electric = hvac_electric,
                             Gas = hvac_natgas,
                             Propane = hvac_propane))

#TODO: make ggplot and find which distribution it fits the most, then set probablistic rule based on that
#distribution.
#need to find national distribution or at the very least distribution of wealthy in arizona to split into
#categories

#in.misc_pool same as resstock

az_data <- az_data |> mutate(in.misc_pool = case_when(
                              in.misc_pool == "Has Pool" ~ "Yes",
                              TRUE ~ "No"))
#does not have empty string case


#VHOUSEHOLD same as in.occupants
resstock <- resstock |> mutate(VHOUSEHOLD = case_when(
                              VHOUSEHOLD >= 10 ~ "10+",
                              TRUE ~ as.character(VHOUSEHOLD)
  
))

#merge together?
#unique(resstock$VWATERHEAT)
#unique(resstock$VTANKTYPE)

#merging in.water_heater_efficiency, in.water_heater_fuel into in.water_heater_efficiency_and_fuel
#want to remove fuel type? seems like redundant variable
az_data <- az_data |>
  unite("in.water_heater_efficiency_and_fuel",in.water_heater_efficiency, in.water_heater_fuel, sep = ", ")

#merging levels of factor for above variable
az_data <- az_data |>
  mutate(in.water_heater_efficiency_and_fuel = case_when(
            str_detect(in.water_heater_efficiency_and_fuel, "(?i)Electric Standard|Electric Premium") ~ "Electric Tank",
            str_detect(in.water_heater_efficiency_and_fuel, "(?i)Electric Heat Pump, 50 gal, 3.45 UEF") ~ "Electric, Heat Pump",
            str_detect(in.water_heater_efficiency_and_fuel, "(?i)Natural Gas Standard|Natural Gas Premium") ~ "Gas Tank",
            str_detect(in.water_heater_efficiency_and_fuel, "(?i)Natural Gas Tankless") ~ "Gas Tankless",
            str_detect(in.water_heater_efficiency_and_fuel, "(?i)Electric Tankless") ~ "Electric Tankless",
            TRUE ~ "Other Fuel"
  ))

#creating standardized variable above for resstock too
resstock <- resstock |>
  unite("VWATERHEAT_TANKTYPE", VWATERHEAT, VTANKTYPE, sep = " ")

#combining apartment, condo class into multi-unit
resstock <- resstock |>
  mutate(VRESTYPE = fct_collapse(VRESTYPE, "Multi-Unit" = c("Apartment", "Condo")))



























