import streamlit as st
import pandas as pd
from helpers.single_model import Utils, ApplyLinearRegression, ApplyLogisticRegression, ApplyDecisionTreeRegressor, ApplyDecisionTreeClassifier
from helpers.multi_model import RegressionHandler, ClassifierHandler

models_list = ["Linear Regression", "Logistic Regression", "Decision Tree Regression", "Decision Tree Classifier"]
linear_regression_model_metrics = ["MAE", "MSE", "RMSE", "R2", "Adjusted R2", "Cross Validated R2"]
logistic_regression_model_metrics = ["Confusion Metrics", "ROC Curve", "Precision Recall Curve"]
problem_selector_options = ["Regression Problem", "Classifier Problem"]

regression_multi_model_problem = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Decision Tree", "Random Forest", "Support Vector Regression", "XGBoost"]
classifier_multi_model_problem = ["Lasso Regression","SGDClassifier",
                                   "Decision Tree", "Random Forest", "XGBoost",
                                   "AdaBoost",
                                   "Support Vector Classification",
                                   "Naive Bayes"
                                ]



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


        problem_selector = st.selectbox("Select Your Problem Type: ", options=problem_selector_options, index=0)

        test_size_selector = st.slider("Select your Test Size: ", min_value=1, max_value=100, value=42, step=1)
        test_size_value = test_size_selector/100
        st.text("Your Test Size is: {}".format(test_size_value))
        X_train, X_test, y_train, y_test = Utils().train_test_split(x, y, float(test_size_value))


        
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
                    lasso_regression_result = classifier_model.apply_linear(model_name="Logistic Regression", penalty=penalty, average=average, multi_class=multi_class)
                    metrics_dict[algo] = lasso_regression_result
                elif algo == "SGDClassifier":
                    lasso_regression_result = classifier_model.apply_linear(model_name="SGDClassifier", penalty=penalty, average=average, multi_class=multi_class, loss=sgd_loss, alpha=sgd_alpha)
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
                
                elif algo == "Naive Bayes":
                    naive_bayes_result = classifier_model.apply_naive_bayse(model_name="Naive Bayes", average=average, multi_class=multi_class)
                    metrics_dict[algo] = naive_bayes_result
                
                elif algo == "AdaBoost":
                    adaboost_result = classifier_model.apply_decision_tree_v2(model_name="AdaBoost", n_estimators=ada_boost_n_estimators, learning_rate=ada_boost_learning_rate, algorithm=ada_boost_algorithm, random_state=ada_boost_random_state, average=average, multi_class=multi_class)
                    metrics_dict[algo] = adaboost_result

            metrics_dataframe = classifier_model.apply_model(metrics_dict)
            st.title("Classification Results: ")
            st.dataframe(metrics_dataframe)