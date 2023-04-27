import streamlit as st
import pandas as pd
from helpers.utils import Utils
from helpers.apply_model import RegressionHandler, ClassifierHandler
from helpers.feature_engineering import DataManagment, DropColumns, RemoveMissingValues, RemoveCategoricalValues, HandleOutliers

models_list = ["Linear Regression", "Logistic Regression", "Decision Tree Regression", "Decision Tree Classifier"]
linear_regression_model_metrics = ["MAE", "MSE", "RMSE", "R2", "Adjusted R2", "Cross Validated R2"]
logistic_regression_model_metrics = ["Confusion Metrics", "ROC Curve", "Precision Recall Curve"]
problem_selector_options = ["Regression Problem", "Classifier Problem"]

regression_multi_model_problem = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Polynomial Regression",
                                  "Decision Tree", "Random Forest", "XGBoost",
                                  "Support Vector Regression", 
                                  ]

classifier_multi_model_problem = ["Lasso Regression","SGDClassifier",
                                   "Decision Tree", "Random Forest", "XGBoost",
                                   "AdaBoost",
                                   "K-Neighbors",
                                   "Support Vector Classification",
                                   "Naive Bayes",
                                   "Perceptron", "Multi-Layer Perceptron"
                                ]



st.sidebar.title("Welcome to InsightML")

with st.sidebar.expander("Upload Dataset"):
    file_upload = st.file_uploader("Upload your Data: ")
    

if file_upload is not None:

    data = Utils().read_file(file_upload)
    
    if st.sidebar.checkbox("Display Dataset Info", value=True):
        describe, shape, columns, num_category, str_category, null_values, dtypes, unique, str_category, column_with_null_values = Utils().describe(data)


        st.title("Dataset Preview")
        st.dataframe(data)

        st.subheader("Dataset Description")
        st.write(describe)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.text("Basic Information")
            st.write("Dataset Name")
            st.text(file_upload.name)

            st.write("Dataset Size(MB)")
            number = round((file_upload.size*0.000977)*0.000977,2)
            st.write(number)

            st.write("Dataset Shape")
            st.write(shape)
            
        with col2:
            st.text("Dataset Columns")
            st.write(columns)
        
        with col3:
            st.text("Numeric Columns")
            st.dataframe(num_category)
        
        with col4:
            st.text("String Columns")
            st.dataframe(str_category)
            

        col5, col6, col7= st.columns(3)

        with col5:
            st.text("Columns Data-Type")
            st.dataframe(dtypes)
        
        with col6:
            st.text("Counted Unique Values")
            st.write(unique)
        
        with col7:
            st.write("Counted Null Values")
            st.dataframe(null_values)
    
    
    work_type = st.sidebar.selectbox("Select Your Work Type: ", options=["Feature Engineering", "Model Evaluating"], index=0)


    if work_type == "Model Evaluating":
        with st.sidebar.expander("ML General Settings"):
            
            drop_columns_selector = st.multiselect("Select Columns Which you want to drop(Optional): ", options=data.columns)
            if len(drop_columns_selector) != 0:
                data = Utils().drop_columns(data, drop_columns_selector)
                
            independent_column_selector = st.selectbox("Select your target(y) Column: ", options=data.columns)
            x, y = Utils().target_column(data, independent_column_selector)


            problem_selector = st.selectbox("Select Your Problem Type: ", options=problem_selector_options, index=0)

            test_size_selector = st.slider("Select your Test Size: ", min_value=1, max_value=100, value=42, step=1)
            test_size_value = test_size_selector/100
            st.text("Your Test Size is: {}".format(test_size_value))

            is_random = st.selectbox("Do you want to set Random state for Train-Test-Split?", options=["No", "Yes"], index=0)
            if is_random == "Yes":
                is_random_num = st.number_input("Select your Random State for Train-Test-Split: ", min_value=0, step=1)
            else:
                is_random_num = None

            X_train, X_test, y_train, y_test = Utils().train_test_split(x, y, float(test_size_value), random_state=is_random_num)


            
        if problem_selector == "Regression Problem":

            with st.sidebar.expander("Selet Your Models"):

                regression_model_selected =st.multiselect("Select Your Regression Models: ", options=regression_multi_model_problem, default=regression_multi_model_problem)
            

            with st.sidebar.expander("Hyperparameters"):

                if "Ridge Regression" in regression_model_selected or "Lasso Regression" in regression_model_selected or "ElasticNet Regression" in regression_model_selected:
                    apha_value_selector = st.number_input("Enter Your Alpha Value (lasso, ridge, elastic): ", min_value=0.0, value=1.0)
                
                if "ElasticNet Regression" in regression_model_selected:
                    l1_ratio_selector = st.number_input("L1 ratio for ElasticNet(optional): ", min_value=0.1, max_value=1.0, value=0.5)                
                
                if "Polynomial Regression" in regression_model_selected:
                    poly_degree = st.number_input("Enter the degree for Polynomial Regression: ", min_value=0, step=1, value=2)

                if "Random Forest" in regression_model_selected or "XGBoost" in regression_model_selected:
                    n_estimators = st.number_input("The number of trees in the Random forest: ", min_value=1, value=100, step=1)
                    bootstrap = st.selectbox("Choose Weather To Bootstrap (Random Forest): ", options=[True, False], index=0)
                
                if "Support Vector Regression" in regression_model_selected:
                        svr_kernal = st.selectbox("Select the kernal for SVR: ", options=["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
                        svr_degree = st.number_input("Input The Degree for SVR: ", min_value=0, step=1, value=3)
                        svr_gama = st.selectbox("Select the Gama for SVR: ", options=["scale", "auto"], index=0)

                
                if "XGBoost" in regression_model_selected:
                    eta = round(st.number_input("Choose Your Learning rate for XGBoost: ", min_value=0.0, max_value=1.0, value=0.3, step=0.001),3)
                    st.text("Your Learning rate is : {}".format(eta))

                if "Random Forest" in regression_model_selected or "Decision Tree" in regression_model_selected:
                    criterion = st.selectbox(
                        'Criterion',
                        ('squared_error', 'friedman_mse', 'absolute_error', 'poisson')
                    )


                    splitter = st.selectbox('Splitter',['best', 'random'], index=0)

                    max_depth = int(st.number_input('Max Depth'))

                    min_samples_split = st.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

                    min_samples_leaf = st.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

                    max_leaf_nodes = int(st.number_input(label='Max Leaf Nodes', min_value=0, value=0, step=1))

                    min_impurity_decrease = st.number_input('Min Impurity Decrease')

                    if max_depth == 0:
                        max_depth = None

                    if max_leaf_nodes == 0:
                        max_leaf_nodes = None

                
            if st.sidebar.button("Evaluate"):

                with st.spinner("Evaluating Model..."):
                    regression_model = RegressionHandler(X_train, X_test, y_train, y_test)

                    metrics_dict = {}

                    for algo in regression_model_selected:
                        if algo == "Linear Regression":
                            linear_regression_result = regression_model.apply_linear_regression(model_name="Linear Regression")
                            metrics_dict[algo] = linear_regression_result
                        if algo == "Lasso Regression":
                            lasso_regression_result = regression_model.apply_linear_regression(model_name="Lasso Regression", alpha=apha_value_selector)
                            metrics_dict[algo] = lasso_regression_result
                        if algo == "Ridge Regression":
                            ridge_regression_result = regression_model.apply_linear_regression(model_name="Ridge Regression", alpha=apha_value_selector)
                            metrics_dict[algo] = ridge_regression_result
                        if algo == "ElasticNet Regression":
                            elastic_regression_result = regression_model.apply_linear_regression(model_name="ElasticNet Regression", alpha=apha_value_selector, l1_ratio=l1_ratio_selector)
                            metrics_dict[algo] = elastic_regression_result
                        if algo == "Decision Tree":
                            decision_tree_regression_result = regression_model.apply_decision_tree(model_name="Decision Tree",criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, splitter=splitter)
                            metrics_dict[algo] = decision_tree_regression_result
                        if algo == "Random Forest":
                            random_forest_regression_result = regression_model.apply_decision_tree(model_name="Random Forest",criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, n_estimators=n_estimators, bootstrap=bootstrap)
                            metrics_dict[algo] = random_forest_regression_result
                        if algo == "Support Vector Regression":
                            svr_regression_result = regression_model.apply_svr(kernal=svr_kernal, degree=svr_degree, gamma=svr_gama)
                            metrics_dict[algo] = svr_regression_result
                        
                        if algo == "XGBoost":
                            xgboost_regression_result = regression_model.apply_decision_tree(model_name="XGBoost", n_estimators=n_estimators, max_depth=max_depth, eta=eta)
                            metrics_dict[algo] = xgboost_regression_result
                        
                        if algo == "Polynomial Regression":
                            polynomial_regression_result = regression_model.apply_linear_regression(model_name="Polynomial Regression", degree=poly_degree)
                            metrics_dict[algo] = polynomial_regression_result


                    metrics_dataframe = regression_model.apply_model(metrics_dict)
                    st.title("Regression Results: ")
                    st.dataframe(metrics_dataframe)
        
        
        
        if problem_selector == "Classifier Problem":
            
            with st.sidebar.expander("Selet Your Models"):

                classification_model_selected =st.multiselect("Select Your Classification Models: ", options=classifier_multi_model_problem, default=classifier_multi_model_problem)
                
            with st.sidebar.expander("Hyperparameters"):

                if "Lasso Regression" in classification_model_selected or "SGDClassifier" in classification_model_selected:
                    penalty = st.selectbox("Select the penalty: ", options=["l1", "l2", "elasticnet", None], index=1)
                
                if "SGDClassifier" in classification_model_selected:
                    sgd_loss = st.selectbox("Select Your Loss Function (SGDClassifier): ", options=["hinge", "log_loss", "log", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], index=0)
                    sgd_alpha = round(st.number_input("Select the alpha: ", min_value=0.00000, value=0.0001, step=0.00001),5)
                    st.text("SGD Alpha Value is: {}".format(sgd_alpha))

                if "Random Forest" in classification_model_selected or "XGBoost" in classification_model_selected:
                    n_estimators = st.number_input("The number of trees in the Random forest: ", min_value=1, value=100, step=1)
                    bootstrap = st.selectbox("Choose Weather To Bootstrap (Random Forest): ", options=[True, False], index=0)

                if "XGBoost" in classification_model_selected or "Decision Tree" not in classification_model_selected:
                    eta = round(st.number_input("Choose Your Learning rate for XGBoost: ", min_value=0.0, max_value=1.0, value=0.3, step=0.001),3)
                    st.text("Your Learning rate is : {}".format(eta))
                    max_depth = int(st.number_input('Max Depth for XGBoost', key=111))
                    if max_depth == 0:
                        max_depth = None
                
                if "AdaBoost" in classification_model_selected:
                    ada_boost_n_estimators = st.number_input("Select Your estimators for AdaBoost: ", min_value=1, value=50, step=1)
                    ada_boost_learning_rate = st.number_input("Select Your Learning Rate for AdaBoost: ", min_value=0.1, value=1.0, step=0.1)
                    ada_boost_algorithm = st.selectbox("Select your algorithm for AdaBoost: ", options=["SAMME", "SAMME.R"], index=1)
                    ada_boost_random_state = st.number_input("Select your Random State for AdaBoost: ", value=282828)
                    st.text("282828 mean random state will be None")
                    if ada_boost_random_state == 282828:
                        ada_boost_random_state = None

                if "Support Vector Classification" in classification_model_selected:
                    svc_kernal = st.selectbox("Select the kernal for SVR: ", options=["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
                    svc_degree = st.number_input("Input The Degree for SVR: ", min_value=0, step=1, value=3)
                    svc_gama = st.selectbox("Select the Gama for SVR: ", options=["scale", "auto"], index=0)

                if "Random Forest" in classification_model_selected or "Decision Tree" in classification_model_selected:
                    criterion = st.selectbox(
                        'Criterion',
                        ["gini", "entropy", "log_loss"], index=0
                    )


                    splitter = st.selectbox('Splitter',['best', 'random'], index=0)
                    max_depth = int(st.number_input('Max Depth'))
                    max_features = st.slider('Max Features', 1, 2, len(X_train.columns))
                    min_samples_split = st.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)
                    min_samples_leaf = st.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)
                    max_leaf_nodes = int(st.number_input(label='Max Leaf Nodes', min_value=0, value=0, step=1))
                    min_impurity_decrease = st.number_input('Min Impurity Decrease')

                    if max_depth == 0:
                        max_depth = None

                    if max_leaf_nodes == 0:
                        max_leaf_nodes = None

            
            with st.sidebar.expander("Evaluation Metrics Hyperparameters"):
                is_multiclass = st.selectbox("Do you have mult-class Target: ", options=[True, False], index=1)

                if is_multiclass:
                    average = st.selectbox("Select your f1 score Average: ", options=["micro", "macro", "samples", "weighted", "binary", None], index=3)
                    multi_class = st.selectbox("Select your multi_class for ROC-AUC Score: ", options=["raise", "ovr", "ovo"], index=1)
                
                else:
                    average = "binary"
                    multi_class = "raise"

            
            if st.sidebar.button("Evaluate"):

                with st.spinner("Evaluating Model..."):

                    classifier_model = ClassifierHandler(X_train, X_test, y_train, y_test)

                    metrics_dict = {}

                    for algo in classification_model_selected:

                        if algo == "Lasso Regression":
                            lasso_regression_result, lasso_ypred = classifier_model.apply_linear(model_name="Logistic Regression", penalty=penalty, average=average, multi_class=multi_class)
                            metrics_dict[algo] = lasso_regression_result
                        elif algo == "SGDClassifier":
                            lasso_regression_result, sgd_ypred = classifier_model.apply_linear(model_name="SGDClassifier", penalty=penalty, average=average, multi_class=multi_class, loss=sgd_loss, alpha=sgd_alpha)
                            metrics_dict[algo] = lasso_regression_result
                        elif algo == "Decision Tree":
                            decision_tree_result, dt_ypred = classifier_model.apply_decision_tree(model_name="Decision Tree", criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, splitter=splitter, average=average, multi_class=multi_class)
                            metrics_dict[algo] = decision_tree_result
                        elif algo == "Random Forest":
                            random_forest_result, rf_ypred  = classifier_model.apply_decision_tree(model_name="Random Forest", criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, n_estimators=n_estimators, bootstrap=bootstrap, average=average, multi_class=multi_class)
                            metrics_dict[algo] = random_forest_result
                        elif algo == "XGBoost":
                            xgboost_result, xg_ypred  = classifier_model.apply_decision_tree(model_name="XGBoost", n_estimators=n_estimators, max_depth=max_depth, eta=eta, average=average, multi_class=multi_class)
                            metrics_dict[algo] =xgboost_result
                        elif algo == "Support Vector Classification":
                            svc_result, svc_ypred  = classifier_model.apply_svc(kernal=svc_kernal, degree=svc_degree, gamma=svc_gama, average=average, multi_class=multi_class)
                            metrics_dict[algo] =svc_result
                        elif algo == "Naive Bayes":
                            naive_bayes_result, nb_ypred = classifier_model.apply_naive_bayse(model_name="Naive Bayes", average=average, multi_class=multi_class)
                            metrics_dict[algo] = naive_bayes_result
                        elif algo == "AdaBoost":
                            adaboost_result, ad_ypred = classifier_model.apply_decision_tree_v2(model_name="AdaBoost", n_estimators=ada_boost_n_estimators, learning_rate=ada_boost_learning_rate, algorithm=ada_boost_algorithm, random_state=ada_boost_random_state, average=average, multi_class=multi_class)
                            metrics_dict[algo] = adaboost_result
                        elif algo == "Perceptron":
                            perceptron_result, p_ypred = classifier_model.apply_perceptron(model_name="Perceptron", average=average, multi_class=multi_class)
                            metrics_dict[algo] = perceptron_result
                        elif algo == "Multi-Layer Perceptron":
                            multi_layer_perceptron_result, mlp_ypred = classifier_model.apply_perceptron(model_name="Multi-Layer Perceptron", average=average, multi_class=multi_class)
                            metrics_dict[algo] = multi_layer_perceptron_result
                        elif algo == "K-Neighbors":
                            K_Neighbors_result, kn_ypred = classifier_model.appply_neighbors(model_name="K-Neighbors", average=average, multi_class=multi_class)
                            metrics_dict[algo] = K_Neighbors_result
                    

                    metrics_dataframe = classifier_model.apply_model(metrics_dict)
                    st.title("Classification Results: ")
                    st.dataframe(metrics_dataframe)
                    
                    st.title("Plot Evaluation Metrics")
                    for algo in classification_model_selected:

                        if algo == "Lasso Regression":
                            classifier_model.plot_evaluation_metrics(lasso_ypred, algo)
                        if algo == "SGDClassifier":
                            classifier_model.plot_evaluation_metrics(sgd_ypred, algo)
                        if algo == "Decision Tree":
                            classifier_model.plot_evaluation_metrics(dt_ypred, algo)
                        if algo == "Random Forest":
                            classifier_model.plot_evaluation_metrics(rf_ypred, algo)
                        if algo == "XGBoost":
                            classifier_model.plot_evaluation_metrics(xg_ypred, algo)
                        if algo == "Support Vector Classification":
                            classifier_model.plot_evaluation_metrics(svc_ypred, algo)
                        if algo == "Naive Bayes":
                            classifier_model.plot_evaluation_metrics(nb_ypred, algo)
                        if algo == "AdaBoost":
                            classifier_model.plot_evaluation_metrics(ad_ypred, algo)
                        if algo == "Perceptron":
                            classifier_model.plot_evaluation_metrics(p_ypred, algo)
                        if algo == "Multi-Layer Perceptron":
                            classifier_model.plot_evaluation_metrics(mlp_ypred, algo)
                        if algo == "K-Neighbors":
                            classifier_model.plot_evaluation_metrics(kn_ypred, algo)


    if work_type == "Feature Engineering":
        
        managment = DataManagment()
        feature_engineering_type = st.sidebar.selectbox("Select your Feature Engineering Type: ", options=["Drop Columns", "Handle Missing Values", "Removing Outliers"])

        if feature_engineering_type == "Drop Columns":
            st.title("Drop Column")
            data = managment.load_data(raw_data=data)
            select_column = st.multiselect("Select Your Column To Be Droped: ", options=data.columns)
            
            if st.button("Apply Changes"):
                DropColumns().remove_column(data=data, selected_columns=select_column)
        
        if feature_engineering_type == "Handle Missing Values":
            
            missing_value = RemoveMissingValues()

            st.title("Handle Missing Values")
            missing_value_type = st.selectbox("Choose Your Missing Value Type: ", options=["Handle Numerical Missing Value", "Handle Categorical Missing Value"])

            if missing_value_type == "Handle Numerical Missing Value":
                
                method_Select = st.selectbox("Select the method you want to apply: ", options=["Drop Missing Value", "Replace Missing Value with Mean", "Replace Missing Value with Median", "Replace Missing value with random Imputed Values"])

                if method_Select == "Drop Missing Value":
                    if st.button("Apply Changes"):
                        missing_value.drop_nan_value(managment.load_data(raw_data=data))

                if method_Select == "Replace Missing Value with Mean":

                    data = managment.load_data(raw_data=data)
                    num_category = [feature for feature in data.columns if data[feature].dtypes != "O"]
                    selected_column = st.multiselect("Select the column: ", options=num_category)
                    
                    if st.button("Apply Changes"):
                        missing_value.remove_num_values_using_mean_median(data=data, method_type="Mean", selected_column=selected_column)
                
                if method_Select == "Replace Missing Value with Median":

                    data = managment.load_data(raw_data=data)
                    num_category = [feature for feature in data.columns if data[feature].dtypes != "O"]
                    selected_column = st.multiselect("Select the column: ", options=num_category)
                    
                    if st.button("Apply Changes"):
                        missing_value.remove_num_values_using_mean_median(data=data, method_type="Median", selected_column=selected_column)


                if method_Select == "Replace Missing value with random Imputed Values":

                    data = managment.load_data(raw_data=data)
                    num_category = [feature for feature in data.columns if data[feature].dtypes != "O"]
                    selected_column = st.multiselect("Select the column: ", options=num_category)
                    if st.button("Apply Changes"):
                        missing_value.random_imputed_value(data, selected_column)
            
        
        
            if missing_value_type == "Handle Categorical Missing Value":
        
                method_Select = st.selectbox("Select the method you want to apply: ", options=["Drop Missing Value", "Replace Missing value with random Imputed Values", "Replace missing value to custom value"])

                remove_cat_values = RemoveCategoricalValues()

                if method_Select == "Drop Missing Value":
                    if st.button("Apply Changes"):
                        remove_cat_values.drop_nan_value(managment.load_data(raw_data=data))
                
                if method_Select == "Replace Missing value with random Imputed Values":
                    data = managment.load_data(raw_data=data)
                    cat_category = [feature for feature in data.columns if data[feature].dtypes == "O"]
                    selected_column = st.multiselect("Select the column: ", options=cat_category)
                    if st.button("Apply Changes"):
                        remove_cat_values.random_imputed_value(data, selected_column)
                
                if method_Select == "Replace missing value to custom value":
                    data = managment.load_data(raw_data=data)
                    cat_category = [feature for feature in data.columns if data[feature].dtypes == "O"]
                    selected_column = st.multiselect("Select the column: ", options=cat_category)
                    rword = st.text_input("Enter the word you want to replace NaN values with: ", placeholder="Missing")
                    if st.button("Apply Changes"):
                        remove_cat_values.impute_custom_values(data, selected_column, rword)


        if feature_engineering_type == "Removing Outliers":
            st.title("Remove Outliers")
            data = managment.load_data(raw_data=data)
            num_category = [feature for feature in data.columns if data[feature].dtypes != "O"]
            selected_column = st.multiselect("Select the column: ", options=num_category)
            if st.button("Apply Changes"):
                HandleOutliers().handle_outliers(data, selected_column)

                
        managment.download_csv()



else:
    DataManagment().remove_file()