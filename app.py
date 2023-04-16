import streamlit as st
import pandas as pd
from helper import Utils, ApplyLinearRegression, ApplyLogisticRegression


models_list = ["Linear Regression", "Logistic Regression"]
linear_regression_model_metrics = ["MAE", "MSE", "RMSE", "R2", "Adjusted R2", "Cross Validated R2"]
logistic_regression_model_metrics = ["Confusion Metrics", "ROC Curve", "Precision Recall Curve"]

st.sidebar.title("Welcome to InsightML")

with st.sidebar.expander("Upload Dataset"):
    file_upload = st.file_uploader("Upload your Data: ")
    

if file_upload is not None:

    data = Utils().read_file(file_upload)
    
    if st.sidebar.checkbox("Display Data", value=True):
        st.title("Uploaded Dataset")
        st.dataframe(data)
    
    with st.sidebar.expander("ML General Settings"):
        
        drop_columns_selector = st.multiselect("Select Columns Which you want to drop(Optional): ", options=data.columns)
        if len(drop_columns_selector) != 0:
            data = Utils().drop_columns(data, drop_columns_selector)
            
        independent_column_selector = st.selectbox("Select your target(y) Column: ", options=data.columns)
        x, y = Utils().target_column(data, independent_column_selector)

        test_size_selector = st.slider("Select your Test Size: ", min_value=1, max_value=100, value=42, step=1)
        test_size_value = test_size_selector/100
        st.text("Your Test Size is: {}".format(test_size_value))
        X_train, X_test, y_train, y_test = Utils().train_test_split(x, y, float(test_size_value))
    
    
    with st.sidebar.expander("Select your Ml Model"):

        model_selector = st.selectbox("Select Your Model: ", options=models_list)

        if model_selector == "Linear Regression":
            linear_model_selector = st.multiselect("Select Your Linear Model(Optional): ", options=[None, "Ridge Regression", "Lasso Regression", "ElasticNet Regression"])
    

    with st.sidebar.expander("Hyperparameters"):

        if model_selector == "Linear Regression":
            if None not in linear_model_selector:
                    apha_value_selector = st.number_input("Enter Your Alpha Value: ", min_value=0.0, value=1.0)
                    l1_ratio_selector = st.number_input("L1 ratio for ElasticNet(optional): ", min_value=0.1, max_value=1.0, value=0.5)

        if model_selector == "Logistic Regression":
            penalty_selector = st.selectbox("Select the penalty: ", options=["l1", "l2", "elasticnet", None], index=1)


    with st.sidebar.expander("Choose Metrics To show"):

        if model_selector == "Linear Regression":
            metrics_selector = st.multiselect("Select the metrics to be ploted: ", options=linear_regression_model_metrics, default=linear_regression_model_metrics)

        elif model_selector == "Logistic Regression":
            metrics_selector = st.multiselect("Select the metrics to be ploted: ", options=logistic_regression_model_metrics, default=logistic_regression_model_metrics)

    if st.sidebar.button("Evaluate"):

        if model_selector == "Linear Regression":

            linear_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_model(metrics_selector)
            st.title("{} Result: ".format(model_selector))
            
            if None not in linear_model_selector:

                if "Ridge Regression" in linear_model_selector:
                    ridge_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_liner_model(metrics_selector, "Ridge Regression", apha_value_selector)
                else:
                    ridge_metrics_dict = Utils().empty_liner_dict()
                if "Lasso Regression" in linear_model_selector:
                    lasso_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_liner_model(metrics_selector, "Lasso Regression", apha_value_selector)
                else:
                    lasso_metrics_dict = Utils().empty_liner_dict()
                if "ElasticNet Regression" in linear_model_selector:
                    elatic_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_liner_model(metrics_selector, "Lasso Regression", apha_value_selector, l1_ratio=l1_ratio_selector)
                else:
                    elatic_metrics_dict = Utils().empty_liner_dict()

                algo_name = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression"]
                metrics_dict = {
                    algo_name[0]: linear_metrics_dict,
                    algo_name[1]: ridge_metrics_dict,
                    algo_name[2]: lasso_metrics_dict,
                    algo_name[3]: elatic_metrics_dict,
                }
                df = pd.DataFrame.from_dict(metrics_dict, orient="index")
                st.dataframe(df)

            else:
                for key, value in linear_metrics_dict.items():
                    if value != None:
                        st.text("{}: {}".format(key, value))
        

        elif model_selector == "Logistic Regression":

            accuracy, recall, precision, f1, auc, y_pred, regression_ = ApplyLogisticRegression(X_train, X_test, y_train, y_test).apply_model(penalty_selector)
            st.title("{} Result: ".format(model_selector))
            st.write("Accuracy: ", accuracy)
            st.write("Recall: ", recall)
            st.write("Precision: ", precision)
            st.write("F1 Score: ", f1)
            st.write("AUC Score: ", auc)
            
            st.title("Plot Metrics")
            ApplyLogisticRegression(X_train, X_test, y_train, y_test).plot_metrics(y_pred,regression_, metrics_selector)