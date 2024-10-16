## Developed By: Prasannalakshmi G
## Reg No: 212222240075
## Date: 


# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.
5. Fit an AutoRegressive (AR) model with 13 lags
6. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
7. Make predictions using the AR model.
8. Compare the predictions with the test data
9. Calculate Mean Squared Error (MSE).
10. Plot the test data and predictions.
   
### PROGRAM :
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Goodreads_books.csv')

# Check for missing values and drop if necessary
data = data.dropna(subset=['average_rating', 'publication_date'])

# Convert 'publication_date' to datetime and extract year
data['publication_date'] = pd.to_datetime(data['publication_date'], errors='coerce')
data['year'] = data['publication_date'].dt.year

# Group by 'year' and calculate the average rating per year
time_series_data = data.groupby('year')['average_rating'].mean().dropna()

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test
result = adfuller(time_series_data)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = time_series_data.iloc[:int(0.8 * len(time_series_data))]
test_data = time_series_data.iloc[int(0.8 * len(time_series_data)):]

# Define the lag order for the AutoRegressive model (adjusted from ACF/PACF)
lag_order = 5  # Adjust based on your ACF/PACF plots
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(time_series_data, lags=20, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Average Ratings')
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(time_series_data, lags=20, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Average Ratings')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for predictions
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

# Plot Test Data vs Predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Test Data - Average Ratings', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Average Ratings', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Average Ratings')
plt.title('AR Model Predictions vs Test Data (Average Ratings)')
plt.legend()
plt.grid(True)
plt.show()


```
## OUTPUT:

## GIVEN DATA :

![{A5C1859D-5F56-48EC-BD22-BF335EFC3E9B}](https://github.com/user-attachments/assets/ba5de968-3cbf-4fff-9b08-04389d6f0013)


## Augmented Dickey-Fuller test :

![{AACCF198-F7AE-4AC1-859A-6E7D91A48EDA}](https://github.com/user-attachments/assets/2be1af5d-fb8b-4103-8fe5-78d438131bfd)


## PACF - ACF : 

![{EDCB1C6D-A279-4DED-9C13-4F8853EC5299}](https://github.com/user-attachments/assets/18bb62f3-96c4-4959-bf27-8e1f2e5bb9a0)

![{202FB8EF-15BA-40F7-AC29-F9858B9E4D95}](https://github.com/user-attachments/assets/de542394-d0bc-49fb-83de-8e5969491b93)



## Mean Squared Error : 

![{633ABB62-D726-4EC9-A184-198572563B7F}](https://github.com/user-attachments/assets/c8bc2638-bd60-4c9f-aa30-938a691f310a)


## FINAL PREDICTION :

![{FD8B0F14-8692-4F4A-8A29-F4ABE8BBFD09}](https://github.com/user-attachments/assets/b6b7b0a6-9158-47c0-b6e2-c7091acfa141)


### RESULT:
Thus, the auto regression function using python is successfully implemented.
