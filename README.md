# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 18-05-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
data = pd.read_csv('wb_commodity_price_intelligence_1960_2026.csv')

# Use Close column
X = data['price_nominal_usd']

# Make stationary
X = X.diff().dropna()

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Plot original data
plt.plot(X)
plt.title('Stationary Data')
plt.show()

# ACF and PACF
plt.subplot(2,1,1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('ACF')

plt.subplot(2,1,2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1)
arma11_model = ARIMA(X, order=(1,0,1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# ARMA(2,2)
arma22_model = ARIMA(X, order=(2,0,2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']

theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```
## OUTPUT:
#### DATA:
<img width="767" height="403" alt="image" src="https://github.com/user-attachments/assets/848289e4-9a22-42be-aa54-31778b7bf630" />

#### ACF:
<img width="772" height="190" alt="image" src="https://github.com/user-attachments/assets/c2c3076e-25aa-4ce5-a904-fd297c4398c9" />


#### PACF:

<img width="776" height="195" alt="image" src="https://github.com/user-attachments/assets/dd46be7a-a05e-42c0-882b-0fec0cda5151" />


#### SIMULATED ARMA(1,1) PROCESS:


<img width="772" height="414" alt="image" src="https://github.com/user-attachments/assets/6bb8163f-cecb-4822-9785-561cca305ce2" />


#### Partial Autocorrelation

<img width="767" height="402" alt="image" src="https://github.com/user-attachments/assets/71b45b55-8392-4064-8fa4-0f68fc59f522" />

#### Autocorrelation


<img width="774" height="406" alt="image" src="https://github.com/user-attachments/assets/42e85fa7-b802-4354-87fb-493db375859a" />


#### SIMULATED ARMA(2,2) PROCESS:

<img width="780" height="416" alt="image" src="https://github.com/user-attachments/assets/11fc469d-cf85-4f26-be22-6f263fab8cd7" />

#### Partial Autocorrelation


<img width="774" height="406" alt="image" src="https://github.com/user-attachments/assets/e510f68d-1313-41e2-a59b-edc7a7d2af13" />


#### Autocorrelation

<img width="777" height="414" alt="image" src="https://github.com/user-attachments/assets/fceb1466-3523-482f-a265-2febc9759fec" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
