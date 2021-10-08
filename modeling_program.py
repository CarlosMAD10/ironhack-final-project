from helper_functions import *

def create_models(list_of_indexes=None, setup_data=None, silent=False):
        
        time0 = time()
        #We determine a series of setup options and the list of indexes that we will use. We will leave the 
        #GDAXI index out because we will use it for further testing.
        if not setup_data:
                setup_data = {"number_of_snippets":100, "previous_days":700, "snippet_size":30,
                 "projection_step":10, "scaling":"minmax", "n_clusters":10}

        if not list_of_indexes:
                list_of_indexes = ['NYA', 'IXIC', 'HSI', 'GSPTSE', 'NSEI',
                 'KS11', 'SSMI', 'TWII',  'N225', 'N100']

        initial_df = create_initial_df(list_of_indexes=list_of_indexes, setup_data=setup_data)
        initial_df = initial_df.drop(columns=["Pivot_date"])

        #CLUSTERING
        
        #Create cluster dataset and model
        X_cluster_train = initial_df.drop(columns=["Pivot_value", "Target_value"])
        clustering_model = KMeans(init="k-means++", n_clusters=setup_data["n_clusters"])

        #We fit the model to the training data and create the array with the predicted clusters
        clustering_model.fit(X_cluster_train)
        clusters = clustering_model.predict(X_cluster_train)

        #We incorporate the clusters to the dataframe, and cast them to objects
        df_clusters = initial_df.copy()
        df_clusters["Cluster"] = clusters
        df_clusters["Cluster"]= df_clusters["Cluster"].astype(object)

        #If the program is not running in silent mode, we show some of the calculated trends, grouped by clusters
        if not silent:
                visualise_clusters(df_clusters, n_per_cluster=5)

        #Save the model using a custom function
        save_clustering_model(clustering_model, path="clustering_model.pkl")

        #CLASSIFICATION

        #We create the classification dataframe, with the percentage of change between pivot and target values, 
        #and the custom function to cast them the percentages into labels (Class_category)
        df_class = df_clusters.copy()
        df_class["Class_category"] = df_class["Target_value"].apply(get_label)

        #Classification: first we setup the experiment and then we create the Random Forest model.
        X_train_class = df_class.drop(columns=["Pivot_value", "Target_value"])
        class_exp = pycaret.classification.setup(X_train_class, 
                target = "Class_category", remove_perfect_collinearity=False, silent=True)

        class_model = pycaret.classification.create_model("rf")

        #Tuning the model's parameters and finalizing the model (training it with all the data, not
        #just the training set)
        class_model = pycaret.classification.tune_model(class_model)
        class_model = pycaret.classification.finalize_model(class_model)

        #Save the model
        pycaret.classification.save_model(class_model, "classification_model")

        #REGRESSION

        #We create the regression dataframe, only with cluster data, classification category and pivot value.
        #We don't feed it the "trend data" (normalised prices and volumes). The target is the target value.
        #df_reg = df_class[["Cluster", "Class_category", "Target_value"]]
        df_reg = df_class.copy()
        df_reg = df_reg.drop(columns=["Pivot_value"])

        #We setup the experiment
        reg_exp = pycaret.regression.setup(df_reg, 
                target = "Target_value", remove_perfect_collinearity=False, silent=True)

        #We create and tune the model
        reg_model = pycaret.regression.create_model("et")
        reg_model = pycaret.regression.tune_model(reg_model)

        #If we are not running in silent mode, we show the regression model's evaluation
        if not silent:
                pycaret.regression.evaluate_model(reg_model)

        #Finalize the model
        reg_model = pycaret.regression.finalize_model(reg_model)

        #Save the model
        pycaret.regression.save_model(reg_model, "regression_model")

        time_1 = time()

        print(f"Time elapsed: {(time_1-time0):.1f} seconds.")

        return "Models created and saved successfully."


def load_models(path_cluster="clustering_model.pkl", path_class="classification_model",
        path_reg="regression_model"):
        

        cluster = load_clustering_model(path=path_cluster)

        classification = pycaret.classification.load_model(path_class)

        regression = pycaret.regression.load_model(path_reg)


        return cluster, classification, regression

def evaluation_ccr(clustering_model, classification_model, regression_model, setup_data=None, indexes=["GDAXI"]):

        if not setup_data:
                setup_data = {"number_of_snippets":30, "previous_days":300, "snippet_size":30,
                 "projection_step":10, "scaling":"minmax", "n_clusters":10}

                X = create_initial_df(setup_data=setup_data, list_of_indexes=indexes)
                y_test = X["Target_value"]
                dates = X["Pivot_date"]
                X = X.drop(columns=["Pivot_date", "Target_value"])
        else:
                X = create_initial_df(setup_data=setup_data, list_of_indexes=indexes)
                y_test = X["Target_value"]
                dates = X["Pivot_date"]
                X = X.drop(columns=["Pivot_date", "Target_value"])

        

        #We run the clustering model and add the clusters
        X_cluster = X.drop(columns=["Pivot_value"])
        clusters = clustering_model.predict(X_cluster)
        X["Cluster"] = clusters
        X["Cluster"] = X["Cluster"].astype(object)

        #Setup the classification experiment (for the encoding of the cluster labels) and predict the 
        #classification categories.
        X_class = X.drop(columns=["Pivot_value"])
        
        categories = classification_model.predict(X_class)

        X["Class_category"] = categories
        X["Class_category"] = X["Class_category"].astype(object)


        #We do the same thing with the regression model
        #X_reg = X[["Cluster", "Class_category"]]
        X_reg = X.copy()
        X_reg = X_reg.drop(columns=["Pivot_value"])

        y_pred = regression_model.predict(X_reg)

        r2 = r2_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        #print(f"The R2 score for the prediction is {r2:.3f}.")
        #print(f"Accuracy is {acc}")

        return y_test, y_pred, X


def prediction(pred_row, clustering_model, classification_model, regression_model):

        pivot_date = pred_row["Pivot_date"]
        row = pred_row.drop(columns=["Pivot_date"])
        cluster = clustering_model.predict(row.drop(columns=["Pivot_value"]))
        row["Cluster"] = cluster
        row["Cluster"] = row["Cluster"].astype(object)
        class_category = classification_model.predict(row.drop(columns=["Pivot_value"]))
        row["Class_category"] = class_category
        row["Class_category"] = row["Class_category"].astype(object)
        y_pred = regression_model.predict(row.drop(columns=["Pivot_value"]))

        return y_pred

def make_prediction(symbol, clustering_model, classification_model, regression_model, pivot_date=None, setup_data=None, apikey=None):
        stock_df = import_stock_data(symbol, apikey=apikey)

        if stock_df.empty:
                print(f"No stock found for symbol {symbol}.")
                return 0, 0, 0

        row = create_prediction_row(stock_df, pivot_date=pivot_date, setup_data=setup_data)
        if row.empty:
                print(f"No stock found for symbol {symbol}.")
                return 0, 0, 0

        y_pred = prediction(row, clustering_model, classification_model, regression_model)
        final_price = float(row["Pivot_value"])*(100 + float(y_pred))/100
        pivot_date = str(row["Pivot_date"][0])[:-9]

        print(f"The predicted value for the stock {symbol} is {final_price:.2f}.")
        print(f"It is a {float(y_pred):.2f} % change since {pivot_date}.")

        return y_pred, final_price, row




def run_program(setup_train=None, setup_test=None, train_indexes=None, test_indexes=None):

        if not setup_train:
                setup_train = {"number_of_snippets":100, "previous_days":700, "snippet_size":30,
                 "projection_step":10, "scaling":"minmax", "n_clusters":10}
        if not setup_test:
                setup_test = {"number_of_snippets":10, "previous_days":400, "snippet_size":30,
                 "projection_step":10, "scaling":"minmax", "n_clusters":10}
        if not train_indexes:
                train_indexes = ['NYA', 'IXIC', 'HSI', 'GSPTSE', 'NSEI',
                 'KS11', 'SSMI', 'TWII',  'N225', 'N100']
        if not test_indexes:
                test_indexes = ["GDAXI"]

        create_models(setup_data=setup_train, list_of_indexes=train_indexes)

        cluster_model, class_model, reg_model = load_models()

        y_test, y_predict, X = evaluation_ccr(cluster_model, class_model, reg_model, setup_data=setup_test, indexes=test_indexes)

        return y_test, y_predict, X, cluster_model, class_model, reg_model



if __name__=="__main__":
        
        print(1)



