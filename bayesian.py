import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import optuna
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
from scipy.stats import kstest
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder

torch.manual_seed(42)

az_data = pd.read_csv('AZ_data_cleaned.csv', low_memory=False)
az_data = az_data[az_data['out.electricity.cooling.energy_consumption.kwh'] != 'NA']

# data cleaning - drop vars
X_df = az_data.drop(az_data.filter(regex='^out.').columns, axis=1)

X_df.replace([np.inf, -np.inf], np.nan, inplace=True) #replace inf vars with nan - could be error from parsing
X_df.dropna(axis=1, how='all', inplace=True) #drop columns that are completely empty
X_uniquecols = X_df.nunique() #drop cols that only have 1 unique value
zero_cols = []
for col, num in X_uniquecols.items():
    if num == 1:
        zero_cols.append(col)
X_df = X_df.drop(columns = zero_cols)

# ordinal factorization:

# # is indicator - ordinal
geom_floorarea_ord = ['Under 1000 sq ft', '1000 - 1499 sq ft', '1500 - 1999 sq ft',
                    '2000 - 2999 sq ft', '3000 - 3999 sq ft', '4000 or more sq ft']
geom_flarea_ordinal_encoder = OrdinalEncoder(categories=[geom_floorarea_ord])
X_df['in.geometry_floor_area'] = geom_flarea_ordinal_encoder.fit_transform(X_df[['in.geometry_floor_area']])

# in.duct_leakage_and_insulation - ordinal
duct_ord = ['0','30% Leakage to Outside, Uninsulated','30% Leakage to Outside, R-4',
                '30% Leakage to Outside, R-6','30% Leakage to Outside, R-8',
                '20% Leakage to Outside, Uninsulated','20% Leakage to Outside, R-4',
                '20% Leakage to Outside, R-6','20% Leakage to Outside, R-8',
                '10% Leakage to Outside, Uninsulated','10% Leakage to Outside, R-4',
                '10% Leakage to Outside, R-6','10% Leakage to Outside, R-8',
                '0% Leakage to Outside, Uninsulated']
duct_ord_encoder = OrdinalEncoder(categories=[duct_ord])
X_df['in.duct_leakage_and_insulation'] = X_df['in.duct_leakage_and_insulation'].fillna('0') # temp
X_df['in.duct_leakage_and_insulation'] = duct_ord_encoder.fit_transform(X_df[['in.duct_leakage_and_insulation']])
# restore after for knn imputation
X_df['in.duct_leakage_and_insulation'] = X_df['in.duct_leakage_and_insulation'].replace('0', np.nan)

# in.geometry_floor_area_bin - ordinal - vif???
geom_floor_area_bin_ord = ['0-1499', '1500-2499', '2500-3999', '4000+']
geom_flarea_bin_ordinal_encoder = OrdinalEncoder(categories=[geom_floor_area_bin_ord])
X_df['in.geometry_floor_area_bin'] = geom_flarea_bin_ordinal_encoder.fit_transform(X_df[['in.geometry_floor_area_bin']])

# factorizing categorical cols - literally anything not a number.
categorical_cols = X_df.select_dtypes(exclude='number').columns
for col in categorical_cols:
    X_df[col] = pd.factorize(X_df[col], use_na_sentinel = False)[0] + 1 

# KNN-impute values - knn imputer with 2 nearest values
imputer = KNNImputer(n_neighbors = 2)
X_df = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

corr_vals = []
for col in X_df.columns:
    corr, p_val = pearsonr(X_df[col].to_numpy(), az_data['out.electricity.cooling.energy_consumption.kwh'])
    if p_val < 0.05:
        corr_vals.append((col, round(abs(corr), 2)))

# sorting by correlation coefficient
corr_vals.sort(key=lambda item: item[1], reverse=True)

for val in corr_vals:
    print(val)

# note: need to take care of replacing NA values within the response variable, change that to cooling
X_df['in.sqft'] = np.log(X_df['in.sqft'] + 1) #transforming right skew
X_df['in.sqft'] = (X_df['in.sqft'] - X_df['in.sqft'].mean()) / X_df['in.sqft'].std()

X_df['in.bedrooms'] = (X_df['in.bedrooms'] - X_df['in.bedrooms'].mean()) / X_df['in.bedrooms'].std() # scaling to z-distribution

X_df['in.representative_income'] = np.sqrt(X_df['in.representative_income']) #transforming right skew
X_df['in.representative_income'] = (X_df['in.representative_income'] - X_df['in.representative_income'].mean()) / X_df['in.representative_income'].std() # scaling to z-distribution

X_df['in.heating_setpoint'] = (X_df['in.heating_setpoint']) ** 2 #transforming left skew
X_df['in.heating_setpoint'] = (X_df['in.heating_setpoint'] - X_df['in.heating_setpoint'].mean()) / X_df['in.heating_setpoint'].std()

X_df['in.occupants'] = np.sqrt(X_df['in.occupants']) #transforming left skew
X_df['in.occupants'] = (X_df['in.occupants'] - X_df['in.occupants'].mean()) / X_df['in.occupants'].std()

cols_keep = ['in.has_pv','in.insulation_ceiling', 'in.sqft', 'in.bedrooms', 'in.representative_income', 'in.duct_leakage_and_insulation', 'in.misc_pool', 'in.dishwasher', 'in.roof_material', 'in.heating_setpoint', 'in.misc_pool_heater', 'in.occupants', 'in.geometry_floor_area', 'in.has_pv', 'in.misc_hot_tub_spa']
X_df = X_df[cols_keep]


# # X_df_vif = add_constant(X_df) # updates with constnat to use for VIF, only using for this case
# # for i in range(1, len(X_df_vif.columns)):
# #     print(f"{X_df_vif.columns[i]} vif: {variance_inflation_factor(X_df_vif.values, i)}")

# # # # print(list(X_df.columns))
y_series = az_data['out.electricity.cooling.energy_consumption.kwh']
y_series = np.sqrt(y_series) # don't scale
# plt.hist(y_series)
# plt.show()

# # # Convert to NumPy arrays
X = X_df.to_numpy(dtype=np.float32)
y = y_series.values

from sklearn.preprocessing import MinMaxScaler

# Check for MPS availability and set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

# new variables to use after I uh...found out that I forgot to absolute value the magnitude.
# in.sqft (0.56), in.geometry_floor_area_bin (0.56),  in.bedrooms(0.48), in.geometry_building_type_height (0.43)
# in.corridor (0.39), in.geometry_building_type_acs (0.39), in.geometry_building_type_recs (0.39), in.plug_loads (0.39)
# in.ahs_region (0.36), in.geometry_building_level_mf (0.36), in.weather_file_city (0.36), in.representative_income (0.35)
# in.ashrae_iecc_climate_zone_2004 (0.35), in.ashrae_iecc_climate_zone_2004_2_a_split (0.35), 
# in.energystar_climate_zone_2023 (0.35), in.neighbors (0.35), in.geometry_building_horizontal_location_mf (0.33)
# in.water_heater_location (0.33), in.county (0.32), in.county_name (0.32), in.hvac_has_ducts (0.32), 
# in.building_america_climate_zone (0.3)

# need to scale these