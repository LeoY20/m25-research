import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
from scipy.stats import kstest

torch.manual_seed(42)

az_data = pd.read_csv('AZ_data_cleaned.csv', low_memory=False)

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

# factorizing categorical cols - literally anything not a number.
categorical_cols = X_df.select_dtypes(exclude='number').columns
for col in categorical_cols:
    X_df[col] = pd.factorize(X_df[col], use_na_sentinel = False)[0] + 1 

# KNN-impute values - knn imputer with 2 nearest values
imputer = KNNImputer(n_neighbors = 2)
X_df = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

# checking if categorical vars were converted / if na values were imputed successfully - seems like so!
for col in X_df.columns:
    if not is_numeric_dtype(X_df[col]):
        print("fuck")
    if X_df[col].isna().any():
        print("shit na still there") 
    if (X_df[col] == -1).any():
        print("column failed: {col}")

corr_vals = []
for col in X_df.columns:
    corr, p_val = pearsonr(X_df[col].to_numpy(), az_data['out.electricity.cooling.energy_consumption.kwh'])
    if p_val < 0.05:
        corr_vals.append((col, round(corr, 2)))

# sorting by correlation coefficient
corr_vals.sort(key=lambda item: item[1], reverse=True)

#k-s test stuff - analyze which ones you need to normalize
nta = []
for i in range(15):
    statistic, p_value = kstest(X_df[corr_vals[i][0]], 'norm')
    if p_value < 0.05: # H0 = var1 follows normal distribution
        nta.append((corr_vals[i][0]))

# print(nta) # lol every single one is not normal data. fuck.
# note: need to take care of replacing NA values within the response variable, change that to cooling



# X_df_vif = add_constant(X_df) # updates with constnat to use for VIF, only using for this case
# for i in range(1, len(X_df_vif.columns)):
#     print(f"{X_df_vif.columns[i]} vif: {variance_inflation_factor(X_df_vif.values, i)}")

# # print(list(X_df.columns))
y_series = az_data['out.electricity.cooling.energy_consumption.kwh']

# # # Fill any resulting NaNs
# # # New, more robust line
# # X_encoded_df = X_encoded_df.fillna(0)

# # Convert to NumPy arrays
X = X_df.to_numpy(dtype=np.float32)
y = y_series.values

from sklearn.preprocessing import MinMaxScaler

# # ... after you create X and y NumPy arrays ...
# X = X_encoded_df.to_numpy(dtype=np.float32)
# y = y_series.values

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
    def __init__(self, in_features, h1 = 1000, h2 = 2500, h3 = 1000, out_features = 1): #pass in itself, 4 features due to petal width, petal length, etc.
        super().__init__() #instantiate our nn.module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) #fc1 is fully connected neural networks, linear model
        self.fc2 = nn.Linear(h1, h2) #basically you are moving forward
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)
    
    def forward(self, x): #relu = rectified linear unit
        #do something, if output < 0, we call 0, otherwise use the output
        x = F.relu(self.fc1(x)) #basically coding to move it thru it
        x = F.relu(self.fc2(x)) #basically you reassign every single time
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
    
input_size = X_df.shape[1] 
model = Model(in_features=input_size)
model.to(device)

#set criterion of model to measure error, how far off predictions are from data
criterion = nn.MSELoss()
# choose adam optimizer (other ones exist), lr = learning rate (if error doesn't go down as we learn)
# also called epochs, we prob want to lower our learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005) #model.parameters basically just gets the parameters from object model

epochs = 10000 #loop guard
# losses = [] #empty list of losses
loss = 0
for i in range(epochs): #for each epoch want to send shit forward
    # Go forward and get prediction
    y_pred = model.forward(X_train) #sending training data forward
    # get predicted results
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad() # gradient descent
    loss.backward() #back propogation
    optimizer.step() #step thru

print(f'Final Loss: {loss}')