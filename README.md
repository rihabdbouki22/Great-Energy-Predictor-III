# Great-Energy-Predictor-III
kaggle competition

 # Abstract:
We are using a dataset related to ASHRAE – Great Energy Predictor III (How much energy will a building consume?). The goal is to develop models from ASHRAE’s 2016 data in order to better understand metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a one-year timeframe. The method chosen to solve the problem is LSTM(Long Short Time Memory).

- Kaggle competition @https://www.kaggle.com/c/ashrae-energy-prediction/overview

# Used libraries
import numpy as np  
import pandas as pd  
import matplotlib  
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import missingno as msno  
import seaborn as sns  
import warnings  
import gc

from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from tensorflow.keras import backend as K  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import LSTM  
from tensorflow.keras.layers import Activation  
from tensorflow.keras.layers import Dropout  
from keras.callbacks import EarlyStopping  
from tensorflow.keras.optimizers import Adam

# Data visualization
The data visualisation section of this work helps us to have a better understanding of the different datasets as well as their feautures. For this purpose we used several data visualisation techniques.
The dataset presented in (Kaggle ASHRAE Energy Predictor III dataset) of Energy Predictor has Five datasets:
*  train.csv with (202116100, 3) data
* building_meta.csv with (1449,6) data
* weather_[train/test].csv with (139773, 8) / (277243, 8) data
* test.csv with (41697600, 3) data

# Data preparation
This part represents the most important part of the project.
To facilitate our task we start by **merging all tables in two datasets train and test **thanks to the foreign keys (building_id, site_id).
Then we used other data preartion techniques such as **dealing with missing values**, **Datetime Features** and **remove redundant columns**.
# Feature engineering
Before we go any further, we need to deal with pesky categorical variables. A machine learning model unfortunately cannot deal with categorical variables.
Therefore, we have to find to encode primary_use variable using **label encoding**.
Also there are 4 types of meters: 0 = electricity, 1 = chilledwater, 2 = steam, 3 = hotwater. We used the **one hot encoding** for this 4 feature.
After identifying outliers we opt for **scaling** very skewed features.
Concerning the target variable meter_reading we opt for the **log tranformation**.

# Data Modeling(LSTM)
For the modeling we used an LSTM architechture composed of one input layes, 3 hidden layes and one output layes with respectively 512, 256, 128, 64, 32 and 1 units. To avoid overfitting we used dropout=0,2 and early stopping.

# Result
After training our model with 20 epochs it gives 0.9959 loss and 0.9963 root_mean_square_error. 

### Submission score
After submitting this project on Kaggle we had 1,764 score.

