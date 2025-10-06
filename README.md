# Deep-Learning-Exp5

DL-Implement a Recurrent Neural Network model for stock price prediction.

**AIM**

To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

**THEORY**

A Recurrent Neural Network (RNN) is a type of deep learning model designed to handle sequential data, making it well-suited for stock price prediction, a classic time-series forecasting problem. Unlike traditional feed-forward neural networks, RNNs possess internal memory that allows them to use information from previous data points to influence the current output, enabling them to recognize patterns and trends over time

**Neural Network Model**

![deep_neural_network_mobile](https://github.com/user-attachments/assets/cdae107f-16dc-4a76-956f-fafe5b6eba5b)


**DESIGN STEPS**

STEP 1: Read the csv file and create the Data frame using pandas.

STEP 2: Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.

STEP 3: Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

STEP 4: Create a model with the desired number of neurons and one output neuron.

STEP 5: Follow the same steps to create the Test data. But make sure you combine the training data with the test data.

STEP 6: Make Predictions and plot the graph with the Actual and Predicted values.

**PROGRAM**

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('/content/trainset.csv')
dataset_train.columns
dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape
length = 60
n_features = 1

model = Sequential([layers.SimpleRNN(40,input_shape=(60,1)),
                    layers.Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(X_train1,y_train,epochs=25, batch_size=64)
model.summary()

dataset_test = pd.read_csv('/content/testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,len(inputs)),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(len(dataset_train), len(dataset_total)),predicted_stock_price[60:], color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
y_test = test_set
mse_value = mse(y_test,predicted_stock_price[60:])
print("Mean squared Error : ",mse_value)


```

**Name:** Madhan kumar J 

**Register Number:** 2305001016


**OUTPUT**

Training Loss Over Epochs Plot

<img width="691" height="498" alt="image" src="https://github.com/user-attachments/assets/46c85278-dcee-4d8b-a1ce-689dd9be7266" />


**True Stock Price, Predicted Stock Price vs time**

<img width="792" height="598" alt="image" src="https://github.com/user-attachments/assets/0667fac4-69f0-4f3d-934b-c83359e34541" />


**Mean Square Error:**

<img width="425" height="49" alt="image" src="https://github.com/user-attachments/assets/35927125-0333-4e9d-95fe-912276a0081d" />


**RESULT**

Thus the stock price is predicted using Recurrent Neural Networks successfully.
