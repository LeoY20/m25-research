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

# is indicator - ordinal
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
        corr_vals.append((col, round(corr, 2)))

# sorting by correlation coefficient
corr_vals.sort(key=lambda item: item[1], reverse=True)

# not very practical - k-s test will say my minor deviations which accumulate over large dataset go boom
# #k-s test stuff - analyze which ones you need to normalize
# nta = []
# for i in range(15):
#     statistic, p_value = kstest(X_df[corr_vals[i][0]], 'norm')
#     if p_value < 0.05: # H0 = var1 follows normal distribution
#         nta.append((corr_vals[i][0]))

# note: need to take care of replacing NA values within the response variable, change that to cooling
X_df['in.sqft'] = np.log(X_df['in.sqft'] + 1) #transforming right skew
X_df['in.sqft'] = (X_df['in.sqft'] - X_df['in.sqft'].mean()) / X_df['in.sqft'].std()

# scaling in.bedrooms - numerical
X_df['in.bedrooms'] = (X_df['in.bedrooms'] - X_df['in.bedrooms'].mean()) / X_df['in.bedrooms'].std() # scaling to z-distribution

# scaling in.representative_income - numerical
X_df['in.representative_income'] = np.sqrt(X_df['in.representative_income']) #transforming right skew
X_df['in.representative_income'] = (X_df['in.representative_income'] - X_df['in.representative_income'].mean()) / X_df['in.representative_income'].std() # scaling to z-distribution

# scaling in.heating_setpoint - is numerical, even though it seems categorical
X_df['in.heating_setpoint'] = (X_df['in.heating_setpoint']) ** 2 #transforming left skew
X_df['in.heating_setpoint'] = (X_df['in.heating_setpoint'] - X_df['in.heating_setpoint'].mean()) / X_df['in.heating_setpoint'].std()

# do not scale in.misc_pool_heater due to indicators

# in.occupants - numerical
X_df['in.occupants'] = np.sqrt(X_df['in.occupants']) #transforming left skew
X_df['in.occupants'] = (X_df['in.occupants'] - X_df['in.occupants'].mean()) / X_df['in.occupants'].std()

# do not scale in.insulation_ceiling - categorical

# do not scale in.misc_pool, in.dishwasher - is indicator

# do not scale roof material - categorical


# don't scale has pv, in_misc_hot_tub_spa because both are indicators

# plt.hist(X_df['in.misc_hot_tub_spa'])
# plt.show()
# # fig = sm.qqplot(X_df['in.geometry_floor_area'], line='45') # '45' adds a 45-degree reference line
# # plt.title("Q-Q Plot")
# # plt.show()

# excluding , 'in.geometry_floor_area' because seems to be a carbon copy of another variable in here.
# somehow in.geometry_floor_area has less correlation
cols_keep = ['in.has_pv','in.insulation_ceiling', 'in.sqft', 'in.bedrooms', 'in.representative_income', 'in.duct_leakage_and_insulation', 'in.misc_pool', 'in.dishwasher', 'in.roof_material', 'in.heating_setpoint', 'in.misc_pool_heater', 'in.occupants', 'in.geometry_floor_area_bin', 'in.has_pv', 'in.misc_hot_tub_spa']
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

# # 1. Split the data BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # 2. Initialize scalers
# x_scaler = MinMaxScaler()
# y_scaler = MinMaxScaler()

# # 3. Fit on the training data and transform it
# X_train_scaled = x_scaler.fit_transform(X_train)
# y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)) # Reshape y for the scaler

# # 4. Use the FITTED scaler to transform the test data
# X_test_scaled = x_scaler.transform(X_test)
# y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

# # 5. Convert the SCALED data to PyTorch Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# moving to macos gpu to free up cpu usage
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

class Model(nn.Module):
    #init defines the properties of the object, we're defining fc1, fc2, out as layers
    def __init__(self, in_features, h1 = 4096, h2 = 2048, h3 = 1024, h4 = 512, h5 = 256, h6 = 128, h7 = 64, out_features = 1): #pass in itself, 4 features due to petal width, petal length, etc.
        super().__init__() #instantiate our nn.module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) #fc1 is fully connected neural networks, linear model
        self.fc2 = nn.Linear(h1, h2) #basically you are moving forward
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.fc6 = nn.Linear(h5, h6)
        self.fc7 = nn.Linear(h6, h7)
        self.out = nn.Linear(h7, out_features)
    
    def forward(self, x): #relu = rectified linear unit
        #do something, if output < 0, we call 0, otherwise use the output
        x = F.relu(self.fc1(x)) #basically coding to move it thru it
        x = F.relu(self.fc2(x)) #basically you reassign every single time
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.out(x)
        return x
    
input_size = X_df.shape[1] 
model = Model(in_features=input_size)
model.to(device)

#set criterion of model to measure error, how far off predictions are from data
criterion = nn.MSELoss()
# choose adam optimizer (other ones exist), lr = learning rate (if error doesn't go down as we learn)
# also called epochs, we prob want to lower our learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001) #model.parameters basically just gets the parameters from object model
# variable learning rate in model
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', 
                                                       factor=0.1, 
                                                       patience=10)

epochs = 1000 #loop guard
# losses = [] #empty list of losses
train_loss = 0
test_loss = 0
print("------------starting training------------")
for i in range(epochs): #for each epoch want to send shit forward
    # Go forward and get prediction
    model.train() # need so that pytorch can do nn training right
    y_pred = model.forward(X_train) #sending training data forward
    # get predicted results
    train_loss = criterion(y_pred, y_train) # because we square rooted it.
    optimizer.zero_grad() # gradient descent
    train_loss.backward() #back propogation
    optimizer.step() #step thru

    # eval phase
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
    # adjust LR based on validation loss
    scheduler.step(test_loss)

# getting an output - uncompressing response variable.
print("------------training complete------------")
with torch.no_grad():
    train_output = model(X_train)
    test_output = model(X_test)

    untrans_train_pred = train_output ** 2 # pred output for training data
    untrans_test_pred = test_output ** 2 # pred output for test data

    resp_train = y_train ** 2 # actual output for training data
    resp_test = y_test ** 2 # actual output for test data

    in_sample_mse = F.mse_loss(untrans_train_pred, resp_train)
    oos_mse = F.mse_loss(untrans_test_pred, resp_test)



print(f'In-sample Final Loss: {in_sample_mse}')
print(f'OOS Final Loss: {oos_mse}')