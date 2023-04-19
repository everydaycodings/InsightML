import streamlit as st
import pandas as pd
from helpers.single_model import Utils, ApplyLinearRegression, ApplyLogisticRegression, ApplyDecisionTreeRegressor, ApplyDecisionTreeClassifier
from helpers.multi_model import RegressionHandler, ClassifierHandler

models_list = ["Linear Regression", "Logistic Regression", "Decision Tree Regression", "Decision Tree Classifier"]
linear_regression_model_metrics = ["MAE", "MSE", "RMSE", "R2", "Adjusted R2", "Cross Validated R2"]
logistic_regression_model_metrics = ["Confusion Metrics", "ROC Curve", "Precision Recall Curve"]
problem_selector_options = ["Regression Problem", "Classifier Problem"]

regression_multi_model_problem = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Decision Tree", "Random Forest", "Support Vector Regression", "XGBoost"]
classifier_multi_model_problem = ["Lasso Regression", "Decision Tree", "Random Forest", "XGBoost", "Support Vector Classification"]



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

        model_type = st.selectbox("Select Your Work Type: ", options=["Multi-Model Inspection", "Single-Model Inspection"], index=0)

        if model_type == "Multi-Model Inspection":

            problem_selector = st.selectbox("Select Your Problem Type: ", options=problem_selector_options, index=0)

        test_size_selector = st.slider("Select your Test Size: ", min_value=1, max_value=100, value=42, step=1)
        test_size_value = test_size_selector/100
        st.text("Your Test Size is: {}".format(test_size_value))
        X_train, X_test, y_train, y_test = Utils().train_test_split(x, y, float(test_size_value))



    if model_type == "Single-Model Inspection":
    
        with st.sidebar.expander("Select your Ml Model"):

            model_selector = st.selectbox("Select Your Model: ", options=models_list)

            if model_selector == "Linear Regression":
                multi_model_selector = st.multiselect("Select Your Linear Model(Optional): ", options=[None, "Ridge Regression", "Lasso Regression", "ElasticNet Regression"])
        
            if model_selector == "Decision Tree Regression":
                multi_model_selector = st.multiselect("Select Your Decision Tree Model(OPtional): ", options=[None, "Random Forest"])
            
            if model_selector == "Decision Tree Classifier":
                multi_model_selector = st.multiselect("Select Your Decision Tree Model(OPtional): ", options=[None, "Random Forest"])
        
        
        
        with st.sidebar.expander("Hyperparameters"):

            if model_selector == "Linear Regression":
                if None not in multi_model_selector:
                        apha_value_selector = st.number_input("Enter Your Alpha Value: ", min_value=0.0, value=1.0)
                        l1_ratio_selector = st.number_input("L1 ratio for ElasticNet(optional): ", min_value=0.1, max_value=1.0, value=0.5)

            if model_selector == "Logistic Regression":
                penalty_selector = st.selectbox("Select the penalty: ", options=["l1", "l2", "elasticnet", None], index=1)

            if model_selector == "Decision Tree Regression":

                if None not in multi_model_selector:
                    n_estimators = st.number_input("The number of trees in the forest: ", min_value=1, value=100, step=1)
                    bootstrap = st.selectbox("Choose Weather To Bootstrap: ", options=[True, False], index=0)
                
                criterion = st.selectbox(
                    'Criterion',
                    ('squared_error', 'friedman_mse', 'absolute_error', 'poisson')
                )

                splitter = st.selectbox(
                    'Splitter',
                    ('best', 'random')
                )

                max_depth = int(st.number_input('Max Depth'))

                min_samples_split = st.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

                min_samples_leaf = st.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

                max_leaf_nodes = int(st.number_input(label='Max Leaf Nodes', min_value=0, value=0, step=1))

                min_impurity_decrease = st.number_input('Min Impurity Decrease')

                if max_depth == 0:
                    max_depth = None

                if max_leaf_nodes == 0:
                    max_leaf_nodes = None


            if model_selector == "Decision Tree Classifier":
                
                if None not in multi_model_selector:
                    n_estimators = st.number_input("The number of trees in the forest: ", min_value=1, value=100, step=1)
                    bootstrap = st.selectbox("Choose Weather To Bootstrap: ", options=[True, False], index=0)

                criterion = st.selectbox(
                    'Criterion',
                    ('gini', 'entropy')
                )

                splitter = st.selectbox(
                    'Splitter',
                    ('best', 'random')
                )

                max_depth = int(st.number_input('Max Depth', step=1))

                min_samples_split = st.slider('Min Samples Split', 1, X_train.shape[0], 2)

                min_samples_leaf = st.slider('Min Samples Leaf', 1, X_train.shape[0], 1)

                max_features = st.slider('Max Features', 1, 2, len(X_train.columns))

                max_leaf_nodes = int(st.number_input(label='Max Leaf Nodes', min_value=0, value=0, step=1))

                min_impurity_decrease = st.number_input('Min Impurity Decrease')

                if max_depth == 0:
                    max_depth = None

                if max_leaf_nodes == 0:
                    max_leaf_nodes = None
        


        with st.sidebar.expander("Choose Metrics To Plot"):

            if model_selector == "Linear Regression" or model_selector == "Decision Tree Regression":
                metrics_selector = st.multiselect("Select the metrics to be ploted: ", options=linear_regression_model_metrics, default=linear_regression_model_metrics)

            elif model_selector == "Logistic Regression":
                metrics_selector = st.multiselect("Select the metrics to be ploted: ", options=logistic_regression_model_metrics, default=logistic_regression_model_metrics)

        
        
        
        if st.sidebar.button("Evaluate"):

            if model_selector == "Linear Regression":

                linear_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_model(metrics_selector)
                st.title("{} Result: ".format(model_selector))
                
                if None not in multi_model_selector:

                    if "Ridge Regression" in multi_model_selector:
                        ridge_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_liner_model(metrics_selector, "Ridge Regression", apha_value_selector)
                    else:
                        ridge_metrics_dict = Utils().empty_liner_dict()
                    if "Lasso Regression" in multi_model_selector:
                        lasso_metrics_dict = ApplyLinearRegression(X_train, X_test, y_train, y_test).apply_liner_model(metrics_selector, "Lasso Regression", apha_value_selector)
                    else:
                        lasso_metrics_dict = Utils().empty_liner_dict()
                    if "ElasticNet Regression" in multi_model_selector:
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
                            st.write(key, ":", value)
            

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
            
            elif model_selector == "Decision Tree Regression":

                st.title("{} Result: ".format(model_selector))
                
                decision_tree_metrics_dict = ApplyDecisionTreeRegressor(X_train, X_test, y_train, y_test).apply_model(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, metrics_selector)
                
                if None not in multi_model_selector:
                    
                    if "Random Forest" in multi_model_selector:
                        random_forest_metrics_dict = ApplyDecisionTreeRegressor(X_train, X_test, y_train, y_test).apply_decision_tree_models("Random Forest", n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, metrics_selector, bootstrap)
                    else:
                        random_forest_metrics_dict = Utils().empty_liner_dict()
                    
                    algo_name = ["Decision Tree", "Random Forest"]
                    metrics_dict = {
                        algo_name[0]: decision_tree_metrics_dict,
                        algo_name[1]: random_forest_metrics_dict,
                    }
                    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
                    st.dataframe(df)

                else:
                    for key, value in decision_tree_metrics_dict.items():
                            if value != None:
                                st.write(key, ":", value)
            
            
            elif model_selector == "Decision Tree Classifier":

                st.title("{} Result: ".format(model_selector))
            
                decision_tree_metrics_dict = ApplyDecisionTreeClassifier(X_train, X_test, y_train, y_test).apply_model(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease)
            
                if None not in multi_model_selector:
                    
                    if "Random Forest" in multi_model_selector:
                        random_forest_metrics_dict = ApplyDecisionTreeClassifier(X_train, X_test, y_train, y_test).apply_decision_tree_models("Random Forest", n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap)
                    else:
                        random_forest_metrics_dict = Utils().empty_liner_dict()
                    
                    algo_name = ["Decision Tree", "Random Forest"]
                    metrics_dict = {
                        algo_name[0]: decision_tree_metrics_dict,
                        algo_name[1]: random_forest_metrics_dict,
                    }
                    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
                    st.dataframe(df)

                else:
                    for key, value in decision_tree_metrics_dict.items():
                            if value != None:
                                st.write(key, ":", value)
        

    if model_type == "Multi-Model Inspection":
        
        if problem_selector == "Regression Problem":

            with st.sidebar.expander("Selet Your Models"):

                regression_model_selected =st.multiselect("Select Your Regression Models: ", options=regression_multi_model_problem, default=regression_multi_model_problem)
            

            with st.sidebar.expander("Hyperparameters"):

                if "Ridge Regression" in regression_model_selected or "Lasso Regression" in regression_model_selected or "ElasticNet Regression" in regression_model_selected:
                    apha_value_selector = st.number_input("Enter Your Alpha Value (lasso, ridge, elastic): ", min_value=0.0, value=1.0)
                
                if "ElasticNet Regression" in regression_model_selected:
                    l1_ratio_selector = st.number_input("L1 ratio for ElasticNet(optional): ", min_value=0.1, max_value=1.0, value=0.5)                
                
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
                    
                
                metrics_dataframe = regression_model.apply_model(metrics_dict)
                

                st.title("Regression Results: ")
                st.dataframe(metrics_dataframe)
        
        
        
        if problem_selector == "Classifier Problem":
            
            with st.sidebar.expander("Selet Your Models"):

                classification_model_selected =st.multiselect("Select Your Classification Models: ", options=classifier_multi_model_problem, default=classifier_multi_model_problem)
                
            with st.sidebar.expander("Hyperparameters"):

                if "Lasso Regression" in classification_model_selected:
                    penalty = st.selectbox("Select the penalty: ", options=["l1", "l2", "elasticnet", None], index=1)

                if "Random Forest" in classification_model_selected or "XGBoost" in classification_model_selected:
                    n_estimators = st.number_input("The number of trees in the Random forest: ", min_value=1, value=100, step=1)
                    bootstrap = st.selectbox("Choose Weather To Bootstrap (Random Forest): ", options=[True, False], index=0)

                if "XGBoost" in classification_model_selected or "Decision Tree" not in classification_model_selected or "XGBoost" not in classification_model_selected:
                    eta = round(st.number_input("Choose Your Learning rate for XGBoost: ", min_value=0.0, max_value=1.0, value=0.3, step=0.001),3)
                    st.text("Your Learning rate is : {}".format(eta))
                    max_depth = int(st.number_input('Max Depth', key=111))
                    if max_depth == 0:
                        max_depth = None
                
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
                    average = st.selectbox("Select your f1 score Average: ", options=["micro", "macro", "samples", "weighted", "binary", None], index=4)
                    multi_class = st.selectbox("Select your multi_class for ROC-AUC Score: ", options=["raise", "ovr", "ovo"], index=0)
                
                else:
                    average = "binary"
                    multi_class = "raise"

            
            if st.sidebar.button("Evaluate"):

                classifier_model = ClassifierHandler(X_train, X_test, y_train, y_test)

                metrics_dict = {}

                for algo in classification_model_selected:

                    if algo == "Lasso Regression":
                        lasso_regression_result = classifier_model.apply_logistic(penalty, average, multi_class)
                        metrics_dict[algo] = lasso_regression_result
                    elif algo == "Decision Tree":
                        decision_tree_result = classifier_model.apply_decision_tree(model_name="Decision Tree", criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, splitter=splitter, average=average, multi_class=multi_class)
                        metrics_dict[algo] = decision_tree_result
                    elif algo == "Random Forest":
                        random_forest_result = classifier_model.apply_decision_tree(model_name="Random Forest", criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, n_estimators=n_estimators, bootstrap=bootstrap, average=average, multi_class=multi_class)
                        metrics_dict[algo] = random_forest_result
                    elif algo == "XGBoost":
                        xgboost_result = classifier_model.apply_decision_tree(model_name="XGBoost", n_estimators=n_estimators, max_depth=max_depth, eta=eta, average=average, multi_class=multi_class)
                        metrics_dict[algo] =xgboost_result
                    elif algo == "Support Vector Classification":
                        svc_result = classifier_model.apply_svc(kernal=svc_kernal, degree=svc_degree, gamma=svc_gama, average=average, multi_class=multi_class)
                        metrics_dict[algo] =svc_result

                metrics_dataframe = classifier_model.apply_model(metrics_dict)
                st.title("Classification Results: ")
                st.dataframe(metrics_dataframe)