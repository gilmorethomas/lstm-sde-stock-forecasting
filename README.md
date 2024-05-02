# lstm-sde-stock-forecasting
LSTM SDE Stock Forecasting, comparing against tradition LSTM and GBM models. 

## Project Overview

This study explores the integration of stochastic modeling and
advanced machine learning techniques, focusing specifically on 
recurrent neural networks (RNNs) to forecast stock prices. 
This will use a combination of models, including 

1. Geometric Brownian Motion
2. Long Short-Term Memory (LSTM) (subclass of RNN) 
3. LSTM model using stochastic differential equations (SDEs)

We aim to compare performance using metrics such as R-CWCE, RMSE, R2, and BIC.  
This project performs univariate, stochastic, time-dependent modeling principles and 
evaluates performance of regressors across variable-length time horizons. 

The study conducts experiments using a dataset comprised of 10 stocks within the technology 
industry from 2009 to 2024, evaluating performance over 5-day, 1-month, 6-month, 1-year, and 5-year periods. 


## MAC 
To activate virtual environment run source env/bin/activate
## Windows 
To activate virtual environment, run env/Scripts/activate.bat //In CMD

## Getting started 
perform a '''pip install -e .''' at this level to register the package and begin w/ development.


## Known Issues 
If there are 

1. Currently no package management
2. Descriptive analytics not complete
3. Univariate analysis only supported
4. Cross-library randomness not currently enforced
5. Mac users may experience issues using torch's newest release. The working version for this code is 1.13.1 https://stackoverflow.com/questions/72556150/there-appear-to-be-1-leaked-semaphore-objects-to-clean-up-at-shutdown4


