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

#merging levels of factor for above variable
az_data <- az_data |>
  mutate(in.water_heater_efficiency_and_fuel = case_when(
            str_detect(in.water_heater_efficiency, "(?i)Electric Standard|Electric Premium") ~ "Electric Tank",
            str_detect(in.water_heater_efficiency, "(?i)Electric Heat Pump, 50 gal, 3.45 UEF") ~ "Electric, Heat Pump",
            str_detect(in.water_heater_efficiency, "(?i)Natural Gas Standard|Natural Gas Premium") ~ "Gas Tank",
            str_detect(in.water_heater_efficiency, "(?i)Natural Gas Tankless") ~ "Gas Tankless",
            str_detect(in.water_heater_efficiency, "(?i)Electric Tankless") ~ "Electric Tankless",
            TRUE ~ "Other Fuel"
  ))

#combining apartment, condo class into multi-unit
resstock <- resstock |>
  mutate(VRESTYPE = fct_collapse(VRESTYPE, "Multi-Unit" = c("Apartment", "Condo")))

#making VAIRCOND, VACTYPE, VACUNITS, VROOMAC like in.hvac_cooling_type
resstock <- resstock |>
  mutate(VCOOLINGTYPE = case_when(
          str_detect(VACTYPE, "(?i)gas") ~ "Central AC",
          VAIRCOND == "Yes" & str_detect(VACTYPE, "(?i)Heat pump") ~ "Ducted Heat Pump",
          VAIRCOND == "No" & str_detect(VACTYPE, "(?i)Heat pump") ~ "Non-Ducted Heat Pump",
          VAIRCOND == "No" & VROOMAC >= 1 ~ "Room AC",
          VAIRCOND == "No" & VROOMAC < 1 ~ "None",
          str_detect(VACTYPE, "(?i)separate") ~ "Central AC",
          TRUE ~ "")) #don't know case set to empty string rn

#removing useless columns
resstock <- resstock |>
  select(-VAIRCOND, -VACTYPE, -VACUNITS, -VROOMAC)

#regex matching for heating
resstock <- resstock |> mutate(VHEATTYPEFUEL = case_when(
  str_detect(VHOMEHEATV1, "(?i)f ?i ?r ?e ?p ?l ?a ?c ?e") ~ "Other",
  str_detect(VHOMEHEAT, "Other|") & str_detect(VHOMEHEATV1, "(?i)propane") ~ "Propane",
  str_detect(VHEATEQUIP, "(?i)Gas furnace|Gas pack") ~ "Gas",
  str_detect(VHEATEQUIP, "(?i)heat pump") ~ "Electric Heat Pump",
  str_detect(VHEATEQUIP, "(?i)individual") ~ "Electric",
  TRUE ~ ""
))

#removing useless columns
resstock <- resstock |>
  select(-VHOMEHEATV1, -VHOMEHEAT, -VHEATEQUIP)


#lightbulbs


#fridges - assuming that I should make this a duplicate and call it vfridges-1
#do you want me to extract the numbers from here too?
resstock <- resstock |>
  mutate("VFRIDGES-1" = case_when(
            VFRIDGES == "None" | VFRIDGES == "" ~ "Zero",
            TRUE ~ VFRIDGES))

resstock <- resstock |>
  select(-VFRIDGES)

#pool indicator - adjusting to be true indicator variable, with assumptions for non-valid data
az_data <- az_data |>
  mutate(in.misc_pool = case_when(
          in.misc_pool == "Yes" ~ 1,
          TRUE ~ 0
  ))

resstock <- resstock |>
  mutate(VPOOL = case_when(
          VPOOL == "Yes" ~ 1,
          TRUE ~ 0
  ))

#turning landlord, empty string to "Don't know"
#combine seemingly unnecessary? variables are quite disjoint as far as I know
#on other set of variables, seems borderline useless to evne have the other variable because 
#information can be spliced from first variable to a bijective correspondence.
resstock <- resstock |>
  mutate(VWATERHEAT = case_when(
    str_detect(VWATERHEAT, "(?i)landlord") | VWATERHEAT == "" ~ "Don't Know",
    str_detect(VWATERHEAT, "(?i)(?=.*solar)(?=.*back-up)") ~ "Other",
    TRUE ~ VWATERHEAT
  ))

#I assume the data is from 2017 based on the name of the dataset, and the data dictionary

resstock <- resstock |>
  mutate(VRESDECADE = case_when(
          is.na(VRESAGE) ~ NA_integer_,
          TRUE ~ ((2017 - VRESAGE) - ((2017 - VRESAGE) %% 10))
  ))

#vresage kept for more precision if needed

#lighting is just take whichever of the 3 ranges is the largest and assign to the variable in the lighting
#one in az data

#strat: index into individual value, see which one is max, then assign index based on that
#in even of tiebreaker, will default to incandescent
#NAs will default to 
lights <- vector(mode = "character", length = nrow(resstock))
lightnames <- c("Mostly Incandescent", "Mostly CFL", "Mostly LED")
for(i in 1:nrow(resstock)) {
  temp <- vector(mode = "numeric", length = 3)
  vinc_curr <- as.numeric(stringr::str_extract(resstock$VINCANDA[i], "^\\s*\\d+"))
  vcfl_curr <- as.numeric(stringr::str_extract(resstock$VCFLA[i], "^\\s*\\d+"))
  vled_curr <- as.numeric(stringr::str_extract(resstock$VLEDA[i], "^\\s*\\d+"))
  temp[1] <- ifelse(is.na(vinc_curr), 0, vinc_curr)
  temp[2] <- ifelse(is.na(vcfl_curr), 0, vcfl_curr)
  temp[3] <- ifelse(is.na(vled_curr), 0, vled_curr)
  tie_indices <- which(temp == max(temp)) #pulls the indices that match max
  if (sum(temp) < 9) { #because 9 is the min value
    lights[i] <- NA
  } else if(length(tie_indices) > 1) { #tiebreak
    lights[i] <- lightnames[sample(tie_indices, size = 1)]
  } else {
    lights[i] <- lightnames[which.max(temp)]
  }
}








