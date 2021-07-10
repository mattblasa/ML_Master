import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, classification_report, roc_curve, auc


import numpy as np
import pandas as pd
from pandas import DataFrame

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#statistics
import pylab
import statsmodels.api as sm
import statistics
from scipy import stats
from statsmodels.formula.api import ols

#Linear Regression 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score

############# ML Utilities ####################################################

def return_shape(df, features, target): #This is only a single column. Will add multiple columns in future update
    '''
    Parameters 
    df- takes in dataframe 
    features 

    Returns 
    Returns correlation heatmap 

    '''
    Y = df[['Churn']]
    X = df[['Yearly_equip_failure','Courteous', 'Listening','MonthlyCharge', 'Bandwidth_GB_Year', 'Income']]
    before_split_shp = X.shape, Y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state =1)
    split_shape = X_train.shape, X_test.shape, y_train.shape, y_test.return_shape

    return before_split_shp, split_shape

# decorate this from return shape 
def Log_Regression(df, features, target, test_sz):
    Y = df[[target]]
    X = df[[features]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= test_sz, random_state =1)

    reg = LogisticRegression()
    reg.fit(X_train, y_train)

    coef = reg.coef_
    intercept = reg.intercept_

    return coef, intercept