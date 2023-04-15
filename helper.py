import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import metrics


class Utils:

    def __init__(self) -> None:
        pass

    def read_file(self, raw_data):
        
        return pd.read_csv(raw_data)

    def drop_columns(self, data, selected_columns):

        return data.drop(selected_columns, axis=1)

    def target_column(self, data, selected_column):

        y = data[selected_column]
        x = data.drop(selected_column, axis = 1)

        return x, y
    

    def train_test_split(self, X, y, test_size):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test

        


class ApplyLinearRegression:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def Reg_Models_Evaluation_Metrics (self, model,X_train,y_train,X_test,y_test,y_pred, metrics_selector):

        cv_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
        
        if "MAE" in metrics_selector:
            mae = round(metrics.mean_absolute_error(y_test, y_pred), 2)
        else:
            mae = None
        
        if "MSE" in metrics_selector:
            mse = round(metrics.mean_squared_error(y_test, y_pred), 2)
        else:
            mse = None
        
        if "R2" in metrics_selector:
            # Calculating Adjusted R-squared
            r2 = round(model.score(X_test, y_test), 2)
        else:
            r2 = None
        
        if "Adjusted R2" in metrics_selector:
            r2_ = model.score(X_test, y_test)
            n = X_test.shape[0]
            p = X_test.shape[1]
            adjusted_r2 = round(1-(1-r2_)*(n-1)/(n-p-1), 2)
        else:
            adjusted_r2 = None
        
        if "RMSE" in metrics_selector:
            RMSE = round(np.sqrt(mse), 2)
        else:
            RMSE = None
        
        if "Cross Validated R2" in metrics_selector:
            CV_R2 = round(cv_score.mean(), 2)
        else:
            CV_R2 = None
        

        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
            "RMSE": RMSE,
            "Cross Validated R2": CV_R2
        }

        return metrics_dict


    def apply_model(self, metrics_selector):

        regression=LinearRegression()
        regression.fit(self.X_train,self.y_train)
        y_pred = regression.predict(self.X_test)

        metrics = self.Reg_Models_Evaluation_Metrics(regression,self.X_train,self.y_train,self.X_test,self.y_test,y_pred, metrics_selector)
    
        return metrics

        