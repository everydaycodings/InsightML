import pandas as pd
import os
import streamlit as st
import datetime

directory = "temp_dir"
filename = "data.csv"
file_path = os.path.join(directory, filename)


class DataManagment:

    def __init__(self):
        pass
        
    def load_data(self, raw_data=None):

        if os.path.isfile(file_path):
            data = pd.read_csv(file_path)

        else:
            with open(file_path, 'w') as f:
                f.write(raw_data.to_csv(index=False))
            data = pd.read_csv(file_path)
        
        return data


    def update_data(self, raw_data):
        with open(file_path, 'w') as f:
            f.write(raw_data.to_csv(index=False))
        data = pd.read_csv(file_path)
        return data


    def remove_file(self):
        try:
            os.remove(file_path)
        except:
            pass
    
    
    def download_csv(self):

        if os.path.isfile(file_path):
            now = datetime.datetime.now()
            with open(file_path) as f:
                st.sidebar.download_button('Download Updated CSV', f, file_name="{}-updated-csv.csv".format(now))

class DropColumns:

    def __init__(self):
        pass

    def remove_column(self, data, selected_columns):

        data = data.drop(selected_columns, axis=1)
        DataManagment().update_data(raw_data=data)

        col1, col2 = st.columns(2)
        with col1:
            st.text("Columns Present in the dataset")
            st.dataframe(data.columns)
        with col2:
            st.text("Columns that has been droped")
            st.dataframe(selected_columns)

        st.text("Dataset Top View")
        st.dataframe(data.head())



class RemoveMissingValues:

    def __init__(self) -> None:
        pass

    
    def remove_num_values_using_mean_median(self, data, selected_column, method_type):

        data_dict = {}

        if method_type == "Mean":
            for col in selected_column:
                mean_data = data[col].mean()
                data[col] = data[col].fillna(mean_data)
                data_dict[col] = mean_data

            st.success("All the null values in the columns  has been replace by the Mean values")
            st.code(data_dict)
            
        if method_type == "Median":
            for col in selected_column:
                median_data = data[col].median()
                data[col] = data[col].fillna(median_data)
                data_dict[col] = median_data

            st.success("All the null values in the columns has been replace by the Median values")
            st.code(data_dict)

        DataManagment().update_data(raw_data=data)
    

    def drop_nan_value(self,data):

        data = data.dropna()
        st.success("Removed all the NaN value")
        DataManagment().update_data(raw_data=data)
    
    def random_imputed_value(self, data, selected_column):

        for col in selected_column:
            data[col] = data[col]
            data[col][data[col].isnull()] = data[col].dropna().sample(data[col].isnull().sum()).values

        DataManagment().update_data(raw_data=data)
        st.success("Replaced all the NaN values with random Imputed values")

        