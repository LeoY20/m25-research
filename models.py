import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

az_data = pd.read_csv('AZ_data_cleaned.csv', low_memory=False)

# Define Features (X) and Target (y)
X_df = az_data.drop(az_data.filter(regex='^out\.').columns, axis=1)
X_df = X_df.drop(columns=['bldg_id'])
print(X_df.columns)
y_series = az_data['out.site_energy.total.energy_consumption.kwh']

# One-Hot Encode using dtype=int
categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
X_encoded_df = pd.get_dummies(X_df, columns=categorical_cols, dtype=int)

# Force all columns to numeric
X_encoded_df = X_encoded_df.apply(pd.to_numeric, errors='coerce')

# Fill any resulting NaNs
# New, more robust line
X_encoded_df = X_encoded_df.fillna(0)

# Convert to NumPy arrays
X = X_encoded_df.to_numpy(dtype=np.float32)
y = y_series.values

from sklearn.preprocessing import MinMaxScaler

# ... after you create X and y NumPy arrays ...
X = X_encoded_df.to_numpy(dtype=np.float32)
y = y_series.values

# 1. Split the data BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize scalers
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# 3. Fit on the training data and transform it
X_train_scaled = x_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)) # Reshape y for the scaler

# 4. Use the FITTED scaler to transform the test data
X_test_scaled = x_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

# 5. Convert the SCALED data to PyTorch Tensors
X_train = torch.FloatTensor(X_train_scaled)
X_test = torch.FloatTensor(X_test_scaled)
y_train = torch.FloatTensor(y_train_scaled) # .view(-1, 1) is already done by reshape
y_test = torch.FloatTensor(y_test_scaled)

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
    
input_size = X_encoded_df.shape[1] 
model = Model(in_features=input_size)

#set criterion of model to measure error, how far off predictions are from data
criterion = nn.MSELoss()
# choose adam optimizer (other ones exist), lr = learning rate (if error doesn't go down as we learn)
# also called epochs, we prob want to lower our learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) #model.parameters basically just gets the parameters from object model

epochs = 1000 #loop guard
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