import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import optuna
import sys
import copy
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
az_data = az_data[az_data['in.neighbors'] != 'Left/Right at 15ft'] # don't know what this means

# data cleaning - drop vars
X_df = az_data.drop(az_data.filter(regex='^out.').columns, axis=1)

# X_df.replace([np.inf, -np.inf], np.nan, inplace=True) #replace inf vars with nan - could be error from parsing
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
duct_ord = ['30% Leakage to Outside, Uninsulated','30% Leakage to Outside, R-4',
                '30% Leakage to Outside, R-6','30% Leakage to Outside, R-8',
                '20% Leakage to Outside, Uninsulated','20% Leakage to Outside, R-4',
                '20% Leakage to Outside, R-6','20% Leakage to Outside, R-8',
                '10% Leakage to Outside, Uninsulated','10% Leakage to Outside, R-4',
                '10% Leakage to Outside, R-6','10% Leakage to Outside, R-8',
                '0% Leakage to Outside, Uninsulated']
duct_ord_encoder = OrdinalEncoder(categories=[duct_ord], handle_unknown='use_encoded_value', unknown_value=np.nan)
X_df['in.duct_leakage_and_insulation'] = duct_ord_encoder.fit_transform(X_df[['in.duct_leakage_and_insulation']])

# in.geometry_floor_area_bin - ordinal - vif???
geom_floor_area_bin_ord = ['0-1499', '1500-2499', '2500-3999', '4000+']
geom_flarea_bin_ordinal_encoder = OrdinalEncoder(categories=[geom_floor_area_bin_ord])
X_df['in.geometry_floor_area_bin'] = geom_flarea_bin_ordinal_encoder.fit_transform(X_df[['in.geometry_floor_area_bin']])

geom_btype_height_ord = ['Mobile Home', 
                         'Single-Family Detached', 
                         'Single-Family Attached',
                         'Multifamily with 2-4 Units',
                         'Multifamily with 5+ units, 1-3 stories',
                         'Multifamily with 5+ units, 4-7 stories',
                         'Multifamily with 5+ units, 8+ stories'
                         ]
geom_bth_ordinal_encoder = OrdinalEncoder(categories=[geom_btype_height_ord])
X_df['in.geometry_building_type_height'] = geom_bth_ordinal_encoder.fit_transform(X_df[['in.geometry_building_type_height']])

in_plug_ord = ['78%', '84%', '103%', '106%', '166%']
in_plug_ord_encoder = OrdinalEncoder(categories=[in_plug_ord])
X_df['in.plug_loads'] = in_plug_ord_encoder.fit_transform(X_df[['in.plug_loads']])

blevel_ord = ['Bottom', 'Middle', 'Top']
blevel_ord_encoder = OrdinalEncoder(categories=[blevel_ord], handle_unknown='use_encoded_value', unknown_value=np.nan)
X_df['in.geometry_building_level_mf'] = blevel_ord_encoder.fit_transform(X_df[['in.geometry_building_level_mf']])

ashr_2004_ord = ['2B', '3B', '4B', '5B']
ashr_2004_ord_encoder = OrdinalEncoder(categories = [ashr_2004_ord])
X_df['in.ashrae_iecc_climate_zone_2004'] = ashr_2004_ord_encoder.fit_transform(X_df[['in.ashrae_iecc_climate_zone_2004']])

neighbors_ord = ['2', '4', '7', '12', '27']
neighbors_ord_encoder = OrdinalEncoder(categories = [neighbors_ord], handle_unknown='use_encoded_value', unknown_value=np.nan)
X_df['in.neighbors'] = neighbors_ord_encoder.fit_transform(X_df[['in.neighbors']])





# factorizing categorical cols - literally anything not a number.
categorical_cols = X_df.select_dtypes(exclude='number').columns
for col in categorical_cols:
    X_df[col] = pd.factorize(X_df[col], use_na_sentinel = False)[0] + 1 

# KNN-impute values - knn imputer with 10 nearest values
imputer = KNNImputer(n_neighbors = 10)
X_df = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

# corr_vals = []
# for col in X_df.columns:
#     corr, p_val = pearsonr(X_df[col].to_numpy(), az_data['out.electricity.cooling.energy_consumption.kwh'])
#     if p_val < 0.05:
#         corr_vals.append((col, round(abs(corr), 2)))

# # sorting by correlation coefficient
# corr_vals.sort(key=lambda item: item[1], reverse=True)

# for val in corr_vals:
#     print(val)

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






cols_keep = ['in.sqft', 'in.geometry_floor_area_bin',
             'in.bedrooms', 'in.geometry_building_type_height',
             'in.corridor', 'in.plug_loads',
             'in.ahs_region', 'in.geometry_building_level_mf',
             'in.weather_file_city', 'in.representative_income',
             'in.ashrae_iecc_climate_zone_2004', 'in.energystar_climate_zone_2023',
             'in.neighbors', 'in.geometry_building_horizontal_location_mf',
             'in.water_heater_location', 'in.county_name',
             'in.hvac_has_ducts']
X_df = X_df[cols_keep]
# , 'in.building_america_climate_zone' 'in.county_name', have

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

X_df_vif = add_constant(X_df) # updates with constnat to use for VIF, only using for this case
for i in range(1, len(X_df_vif.columns)):
    print(f"{X_df_vif.columns[i]} vif: {variance_inflation_factor(X_df_vif.values, i)}")

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

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 42) 


X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_val = torch.FloatTensor(y_val).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# moving to macos gpu to free up cpu usage
X_train = X_train.to(device)
X_val = X_val.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)


# This is the function Optuna will optimize.
# It takes a 'trial' object, suggests hyperparameters, builds and trains a model,
# and returns the best validation score for that trial.
def objective(trial):
    # --- 1. Suggest Hyperparameters ---
    n_layers = trial.suggest_int("n_layers", 2, 7) # Number of hidden layers
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5) # Dropout rate
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) # Learning rate
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    # --- 2. Dynamically Build the Model ---
    layers = []
    in_features = X_train.shape[1]

    for i in range(n_layers):
        # Suggest the number of neurons for this hidden layer
        # The range of neurons decreases for deeper layers
        out_features = trial.suggest_int(f"n_units_l{i}", 64, 4096)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate)) # Add dropout for regularization
        in_features = out_features # The new input is the output of this layer

    layers.append(nn.Linear(in_features, 1)) # Final output layer
    
    model = nn.Sequential(*layers).to(device)

    # --- 3. Setup Optimizer and Criterion ---
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Use fewer epochs for hyperparameter search to save time
    epochs = 300 
    
    best_mse = np.inf
    best_weights = None
    
    # --- 4. Training and Validation Loop ---
    for i in range(epochs):
        model.train()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation phase with early stopping for this trial
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            
            if val_loss.item() < best_mse:
                best_mse = val_loss.item()
                best_weights = copy.deepcopy(model.state_dict())
    
    # Return the best validation MSE found during this trial.
    # Optuna will try to minimize this value.
    return best_mse

# --- 5. Run the Optimization Study ---
# The study object manages the optimization process.
study = optuna.create_study(direction="minimize")

# Start the optimization. Optuna will call the 'objective' function n_trials times.
# More trials will likely find better results but will take longer.
study.optimize(objective, n_trials=50)


# --- 6. Print the Best Results ---
print("\n" + "="*40)
print("Optuna Study Finished")
print(f"Number of finished trials: {len(study.trials)}")

print("\nBest trial:")
best_trial = study.best_trial

print(f"  Value (minimized validation loss): {best_trial.value:.5f}")

print("\n  Best Hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
print("="*40)

# After finding the best hyperparameters, you can build the final model
# with these parameters and train it on the full training data for more epochs.