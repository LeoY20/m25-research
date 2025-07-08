import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

#seed for randomization
torch.manual_seed(41)

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    #init defines the properties of the object, we're defining fc1, fc2, out as layers
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3): #pass in itself, 4 features due to petal width, petal length, etc.
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
    
model = Model()
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# change last column from strings to integer indicators
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)

# print(my_df.head())

# train test split, x, y

X = my_df.drop('species', axis = 1) #dropping the last cuz first 4 are features but last is variety, the outcome
y = my_df['species']

#conversion to numpy array

X = X.values
y = y.values

from sklearn.model_selection import train_test_split

#Train Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

# convert X features to float tensors
X_train = torch.FloatTensor(X_train) #why float tensor? cuz all the input is floats
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#set criterion of model to measure error, how far off predictions are from data
criterion = nn.CrossEntropyLoss()
# choose adam optimizer (other ones exist), lr = learning rate (if error doesn't go down as we learn)
# also called epochs, we prob want to lower our learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) #model.parameters basically just gets the parameters from object model
# print(model.parameters)

#train our model
#epochs = run thru all of training data in network
epochs = 100 #loop guard
losses = [] #empty list of losses
for i in range(epochs): #for each epoch want to send shit forward
    # Go forward and get prediction
    y_pred = model.forward(X_train) #sending training data forward
    # get predicted results

    # measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train)

    # keep track of our losses
    losses.append(loss.detach().numpy()) #append loss onto the array

    # print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # do backpropogation: take error rate moving forwards (forward propogation)
    # feed it back thru network to fine-tune the weights of each node

    optimizer.zero_grad() # gradient descent
    loss.backward() #back propogation
    optimizer.step() #step thru

# plt.plot(range(epochs), losses)
# plt.ylabel("loss/error")
# plt.xlabel("epoch")
# plt.show()

with torch.no_grad(): #turns off backpropogation, just want to send the data right thru
    y_eval = model.forward(X_test) #x_test are features from test set
    #y_eval are the predictions
    loss = criterion(y_eval, y_test) #find loss, error
    print(loss)

correct = 0
with torch.no_grad():
    #need this loop guard as X_test is now a tensor
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_test[i] == 0:
            x = "Setosa"
        elif y_test[i] == 1:
            x = "Versicolor"
        else:
            x = "Virginica"
        #will tell us what type our network thinks it is
        print(f'{i+1}.) {str(y_val)} \t {x} \t {y_val.argmax().item()}')
        #highest number in tensor is what our network thinks the flower is
        #y_val.argmax is prediction
        if y_val.argmax().item() == y_test[i]:
            correct+=1

print(f'we got {correct} correct')