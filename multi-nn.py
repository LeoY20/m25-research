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
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
from scipy.stats import kstest
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFromModel

torch.manual_seed(42)

az_data = pd.read_csv('AZ_data_cleaned.csv', low_memory=False)
az_data = az_data[az_data['out.electricity.cooling.energy_consumption.kwh'] != 'NA'] # annual data, so this should be hella easy lmao
az_data = az_data[az_data['out.electricity.cooling.energy_consumption.kwh'] >= 0]

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
# X_df['in.geometry_floor_area_below'] = np.where(X_df['in.geometry_floor_area_bin'] < 2, 1, 0)


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

# this is presence of neighbors
neighbors_ord = ['2', '4', '7', '12', 'Left/Right at 15ft', '27']
neighbors_ord_encoder = OrdinalEncoder(categories = [neighbors_ord], handle_unknown='use_encoded_value', unknown_value=np.nan)
X_df['in.neighbors'] = neighbors_ord_encoder.fit_transform(X_df[['in.neighbors']])

# factorizing categorical cols - literally anything not a number.
categorical_cols = X_df.select_dtypes(exclude='number').columns
for col in categorical_cols:
    X_df[col] = pd.factorize(X_df[col], use_na_sentinel = False)[0] + 1 

# KNN-impute values - knn imputer with 10 nearest values
imputer = KNNImputer(n_neighbors=10)
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

# , 'in.energystar_climate_zone_2023','in.ashrae_iecc_climate_zone_2004',
# 'in.water_heater_location', 'in.county_name', , 'in.geometry_building_horizontal_location_mf'
# cols_keep = ['in.sqft',
#              'in.geometry_floor_area_bin',
#              'in.bedrooms', 'in.geometry_building_type_height',
#              'in.corridor', 'in.plug_loads',
#              'in.ahs_region', 'in.geometry_building_level_mf',
#              'in.weather_file_city', 'in.representative_income',
#              'in.neighbors', 'in.geometry_building_horizontal_location_mf',
#              'in.hvac_has_ducts',
#              'in.heating_setpoint', 'in.occupants']
# X_df = X_df[cols_keep]

#one-hot encoding: 'in.weather_file_city', 'in.energystar_climate_zone_2023'
#'in.geometry_building_horizontal_location_mf', 'in.water_heater_location'.
# 'in.county_name', 'in.hvac_has_ducts'
X_df = pd.get_dummies(X_df, columns = ['in.weather_file_city'])
# X_df = pd.get_dummies(X_df, columns = ['in.energystar_climate_zone_2023'])
X_df = pd.get_dummies(X_df, columns = ['in.geometry_building_horizontal_location_mf'])
# X_df = pd.get_dummies(X_df, columns = ['in.water_heater_location'])
# X_df = pd.get_dummies(X_df, columns = ['in.county_name'])
X_df = pd.get_dummies(X_df, columns = ['in.hvac_has_ducts'])

# # # # print(list(X_df.columns))
y_series = az_data['out.electricity.cooling.energy_consumption.kwh']
y_series = np.sqrt(y_series) # don't scale

# scaler = MinMaxScaler() - cannot have minmaxscaler because we are now using different thing
# X = scaler.fit_transform(X_df)



# Check for MPS availability and set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

class Model(nn.Module):
    #init defines the properties of the object, we're defining fc1, fc2, out as layers
    # , h5 = 256, h6 = 128, h7 = 64
    def __init__(self, in_features, h1 = 1000, h2 = 500, h3 = 250, h4 = 125, out_features = 1): #pass in itself, 4 features due to petal width, petal length, etc.
        super().__init__() #instantiate our nn.module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) #fc1 is fully connected neural networks, linear model
        self.fc2 = nn.Linear(h1, h2) #basically you are moving forward
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        # self.fc5 = nn.Linear(h4, h5)
        # self.fc6 = nn.Linear(h5, h6)
        # self.fc7 = nn.Linear(h6, h7)
        self.out = nn.Linear(h4, out_features)
    
    def forward(self, x): #relu = rectified linear unit
        #do something, if output < 0, we call 0, otherwise use the output
        x = F.relu(self.fc1(x)) #basically coding to move it thru it
        x = F.relu(self.fc2(x)) #basically you reassign every single time
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = self.out(x)
        return x

models_saved = {} # dictionary to store all models
analyzed_category = 'in.geometry_floor_area_bin'
size_categories = X_df[analyzed_category].unique() # need to drop this category when training
print(X_df[analyzed_category].value_counts())

# training loop - for each category
for category in size_categories:
    print(f"Training for {category}...")
    # getting dataset for category
    category_mask = X_df[analyzed_category] == category # bool mask for getting right rows
    X_category_df = X_df[category_mask] # train dataset
    X_category_df = X_category_df.drop(columns = [analyzed_category]) # drop redundant column, all constant now
    y_category_df = y_series[category_mask]

    # Convert to NumPy arrays
    X_category_df_np = X_category_df.to_numpy(dtype=np.float32)
    y_category_df_np = y_category_df.values

    scaler = MinMaxScaler()
    X_category_df_np = scaler.fit_transform(X_category_df_np)

    # train test split for this specific dataset - 70/30
    X_train, X_test, y_train, y_test = train_test_split(X_category_df_np, y_category_df_np, test_size=0.3, random_state=42)
    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    lgbm = LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    selector = SelectFromModel(lgbm, prefit=True, threshold="median")

    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    # conversion to pytorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_train = torch.FloatTensor(y_train).view(-1, 1).to(device)
    y_val = torch.FloatTensor(y_val).view(-1, 1).to(device)

    # model initialization:
    # shape returns (row, col), and [1] indexes to get cols
    input_size = X_train.shape[1]  # gets all the parameters
    model = Model(in_features=input_size)
    model.to(device)

    # loss function, optimizer, lr scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005) #model.parameters basically just gets the parameters from object model
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', 
                                                       factor=0.1, 
                                                       patience=10)

    # training specific model
    epochs = 1000 #loop guard
    train_loss = 0
    test_loss = 0
    val_loss = 0
    best_mse = np.inf   # init to infinity
    best_weights = None

    # training loops
    for i in range(epochs):
        model.train() # need so that pytorch can do nn training right
        y_pred = model.forward(X_train) #sending training data forward
        train_loss = criterion(y_pred, y_train) # because we square rooted it.
        optimizer.zero_grad() # gradient descent
        train_loss.backward() #back propogation
        optimizer.step() #step thru`

        # eval phase
        model.eval() # Set the model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            if(val_loss < best_mse):
                best_mse = val_loss
                best_weights = copy.deepcopy(model.state_dict())
        # adjust LR based on validation loss
        scheduler.step(val_loss)

    models_saved[category] = {
        'model': model,
        'X_test': X_test, # test dataset for category so I don't lose it
        'y_test': y_test
    }
    

# at this point, X_test is a pytorch tensor
# size_categories = X_test['in.geometry_floor_area_bin'].unique()
# run mse on each individual test
category_errors = []
# X_test = torch.FloatTensor(X_test).to(device)
# y_test = torch.FloatTensor(y_test).view(-1, 1).to(device)
for category in size_categories:
    # retreiving test datasets
    temp_ds_x = models_saved[category]['X_test']
    temp_ds_y = models_saved[category]['y_test']

    if temp_ds_x.size == 0:
        category_errors.append([category, [0]])
        continue

    # tensor conversion
    temp_ds_x = torch.FloatTensor(temp_ds_x).to(device)
    temp_ds_y = torch.FloatTensor(temp_ds_y).view(-1, 1).to(device)
    model_to_use = models_saved[category]['model'] # getting the right model

    model_to_use.eval()
    # pred calculations
    with torch.no_grad():
        y_test_pred = model_to_use(temp_ds_x).squeeze()
    y_test_pred_untrans = y_test_pred ** 2
    resp_y_untrans = temp_ds_y ** 2

    # numpy array conversion
    y_true_np = resp_y_untrans.cpu().numpy() # back to cpu
    y_pred_np = y_test_pred_untrans.cpu().detach().numpy()

    # array of squared errors
    individual_squared_errors = (y_true_np - y_pred_np)**2
    category_errors.append([category, individual_squared_errors])

labels = []
indiv_mse = []
for category in category_errors:
    labels.append(category[0])
    indiv_mse.append(np.mean(category[1]))

plt.bar(labels, indiv_mse)
plt.title('MSE of Each Model')
plt.xlabel('House Size')
plt.ylabel('MSE')

plt.show()

# # mse eval
# model.eval()
# with torch.no_grad():
#     train_output = model(X_train)
#     test_output = model(X_test)

#     untrans_train_pred = train_output ** 2 # pred output for training data
#     untrans_test_pred = test_output ** 2 # pred output for test data

#     resp_train = y_train ** 2 # actual output for training data
#     resp_test = y_test ** 2 # actual output for test data

#     in_sample_mse = F.mse_loss(untrans_train_pred, resp_train)
#     oos_mse = F.mse_loss(untrans_test_pred, resp_test)



# print(f'In-sample Final Loss: {in_sample_mse}')
# print(f'OOS Final Loss: {oos_mse}')