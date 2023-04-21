import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split


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

    
    def describe(self, data):
        num_category = [feature for feature in data.columns if data[feature].dtypes != "O"]
        str_category = [feature for feature in data.columns if data[feature].dtypes == "O"]
        column_with_null_values = data.columns[data.isnull().any()]
        return data.describe(), data.shape, data.columns, num_category, str_category, data.isnull().sum(),data.dtypes.astype("str"), data.nunique(), str_category, column_with_null_values


    def empty_liner_dict(self):

        metrics_dict = {
            "MAE": "Not Selected",
            "MSE": "Not Selected",
            "R2": "Not Selected",
            "Adjusted R2": "Not Selected",
            "RMSE": "Not Selected",
            "Cross Validated R2": "Not Selected"
        }

        return metrics_dict