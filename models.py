import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split

## TODO: remove bad data - identified problem as NAN
## perform scaling, look at teammates data and do scaling based on 
## k-s test
## 

torch.manual_seed(42)

az_data = pd.read_csv('AZ_data_cleaned.csv', low_memory=False)


# 1. Start with your original DataFrame after dropping initial columns
X_df = az_data.drop(az_data.filter(regex='^out\.').columns, axis=1)
# so python does not implicitly fill in columns = x....some bs icl
# select_dtypes returns a dataframe of the selected columns of certain type
# Dropping the non-numerical columns for now until I can convert them
X_df = X_df.drop(columns = X_df.select_dtypes(include = 'object').columns) 
X_df = X_df.drop(columns=['bldg_id', 'applicability']) #dropping specific columns

#replace INF values, somehow you don't need to reassign for these?
X_df.replace([np.inf, -np.inf], np.nan, inplace=True)

#drop columns that are empty
X_df.dropna(axis=1, how='all', inplace=True)

#replace NAN values with mean of the column
X_df.fillna(X_df.mean(), inplace=True)

# removing all columns with only 1 unique value
X_uniquecols = X_df.nunique()

#creating list
zero_cols = []
for col, num in X_uniquecols.items():
    if num == 1:
        zero_cols.append(col)

# print(zero_cols)

#dropping the columns with only 1 value
X_df = X_df.drop(columns = zero_cols)

# dropping perfectly predicting variables - which should happen because I DIDN'T DELETE LMAO
# corr_matrix = X_df.corr().abs()

# perfect_corr = []

# #row-major ordering
# for i in range(len(corr_matrix.index)):
#     for j in range(len(corr_matrix.columns)):
#         if corr_matrix.iloc[i, j] > 0.9 and i != j:
#             row = corr_matrix.index[i]
#             col = corr_matrix.columns[j]
#             perfect_corr.append((row, col, round(corr_matrix.iloc[i, j], 3)))

# print(perfect_corr)

# X_df_vif = add_constant(X_df) # updates with constnat to use for VIF, only using for this case
# for i in range(1, len(X_df_vif.columns)):
#     print(f"{X_df_vif.columns[i]} vif: {variance_inflation_factor(X_df_vif.values, i)}")

# komologorov-smirnov test


# print(f"Shape after cleaning NaNs/Infs: {X_df.shape}")

# # 2. **CRITICAL DIAGNOSTIC STEP:** Check the data types
# print("--- Initial Data Types (first 10 columns) ---")
# print(X_df.info(verbose=False, show_counts=True)) # Use .info() to see types and non-null counts


# print(list(X_df.columns))
y_series = az_data['out.site_energy.total.energy_consumption.kwh']

# # One-Hot Encode using dtype=int
# categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
# X_encoded_df = pd.get_dummies(X_df, columns=categorical_cols, dtype=int)

# # Force all columns to numeric
# X_encoded_df = X_encoded_df.apply(pd.to_numeric, errors='coerce')

# # Fill any resulting NaNs
# # New, more robust line
# X_encoded_df = X_encoded_df.fillna(0)

# # Convert to NumPy arrays
X = X_df.to_numpy(dtype=np.float32)
y = y_series.values

from sklearn.preprocessing import MinMaxScaler

# # ... after you create X and y NumPy arrays ...
# X = X_encoded_df.to_numpy(dtype=np.float32)
# y = y_series.values

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
y_train = torch.FloatTensor(y_train).view(-1, 1) # .view(-1, 1) is already done by reshape
y_test = torch.FloatTensor(y_test).view(-1, 1)

class Model(nn.Module):
    #init defines the properties of the object, we're defining fc1, fc2, out as layers
    def __init__(self, in_features, h1 = 15, h2 = 15, out_features = 1): #pass in itself, 4 features due to petal width, petal length, etc.
        super().__init__() #instantiate our nn.module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) #fc1 is fully connected neural networks, linear model
        self.fc2 = nn.Linear(h1, h2) #basically you are moving forward
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x): #relu = rectified linear unit
        #do something, if output < 0, we call 0, otherwise use the output
        x = F.relu(self.fc1(x)) #basically coding to move it thru it
        x = F.relu(self.fc2(x)) #basically you reassign every single time
        x = self.out(x)
        return x
    
input_size = X_df.shape[1] 
model = Model(in_features=input_size)

#set criterion of model to measure error, how far off predictions are from data
criterion = nn.MSELoss()
# choose adam optimizer (other ones exist), lr = learning rate (if error doesn't go down as we learn)
# also called epochs, we prob want to lower our learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) #model.parameters basically just gets the parameters from object model

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