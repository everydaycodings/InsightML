import streamlit as st
from helper import Utils, ApplyLinearRegression


models_list = ["Linear Regression", "Logistic Regression"]
linear_regression_model_metrics = ["MAE", "MSE", "RMSE", "R2", "Adjusted R2", "Cross Validated R2"]

st.sidebar.title("Welcome to InsightML")

with st.sidebar.expander("Upload Dataset"):
    file_upload = st.file_uploader("Upload your Data: ")
    

if file_upload is not None:

    data = Utils().read_file(file_upload)
    
    if st.sidebar.checkbox("Display Data"):
        st.subheader("Dataset")
        st.dataframe(data)
    
    with st.sidebar.expander("General Settings"):
        
        drop_columns_selector = st.multiselect("Select Columns Which you want to drop(Optional): ", options=data.columns)
        if len(drop_columns_selector) != 0:
            data = Utils().drop_columns(data, drop_columns_selector)
            
        independent_column_selector = st.selectbox("Select your target Column: ", options=data.columns)
        x, y = Utils().target_column(data, independent_column_selector)

        test_size_selector = st.slider("Select your Test Size: ", min_value=1, max_value=100, value=42, step=1)
        test_size_value = test_size_selector/100
        st.text("Your Test Size is: {}".format(test_size_value))
        X_train, X_test, y_train, y_test = Utils().train_test_split(x, y, float(test_size_value))
    
    
    with st.sidebar.expander("Select your Ml Model"):

        model_selector = st.selectbox("Select Your Model: ", options=models_list)

    
    with st.sidebar.expander("Choose Metrics To show"):

        if model_selector == "Linear Regression":
            metrics_selector = st.multiselect("Select the metrics to be ploted: ", options=linear_regression_model_metrics, default=linear_regression_model_metrics)

    if st.sidebar.button("Evaluate"):

        if model_selector == "Linear Regression":

            metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_model(metrics_selector)
            st.subheader("Metrics Result: ")
            for key, value in metrics_dict.items():
                if value != None:
                    st.text("{}: {}".format(key, value))