import pandas as pd
import numpy as np
from time import time
from datetime import datetime
import pickle
import requests
import pycaret.classification
import pycaret.regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, r2_score
import matplotlib.pyplot as plt
import sys

def normalize_vector(data):
        norm = np.linalg.norm(data)
        return data/norm

def normalize_minmax(data):
        m = data.min()
        M = data.max()
        norm_list = [(x-m)/(M-m) for x in data]
        return np.array(norm_list)

def normalize_origin(data):
        bias = data.iloc[0]
        return data-bias


def import_index_data(symbol="GDAXI", path="./data/indexData.csv"):
        df = pd.read_csv(path)
        df = df[df["Index"]==symbol]
        df = df.fillna(method="ffill")
        #df["SMA20"] = df["Close"].rolling(20).mean()
        #df["Spread"] = df["High"] - df["Low"]
        df = df[["Date","Close", "Volume"]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.reset_index(drop=True)    

        return df

def import_stock_data(symbol, apikey=None):

        if not apikey:
                with open(r"C:\Users\carlo\OneDrive\Programming\alphavantage.txt", "r") as f:
                        apikey = f.readline().strip()

        datatype = "csv"  #json is the other option is full (20 plus years)
        outputsize = "compact" #returns only the 100 last data points. The other option is full

        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}\
&outputsize={outputsize}&apikey={apikey}&datatype={datatype}"
        
        df = pd.read_csv(url)

        if len(df) == 2 or len(df) < 30:
                df = pd.DataFrame(columns=["a"])
                return df

        df = df[["timestamp","close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.rename(columns={'timestamp':'Date', "close":"Close", "volume":"Volume"})
        df = df.iloc[::-1].reset_index(drop=True)  #Reverse the df
        

        return df    


def create_modeling_df(data_input, number_of_snippets, snippet_size=30, projection_step=10, scaling="minmax"):
        #Scale the data
        data = data_input.copy()
        #scaler = MinMaxScaler()
        #data.iloc[:,1:3] = scaler.fit_transform(data.iloc[:,1:3])

        #Create the columns that the df will have
        cols = [0]*snippet_size*2
        for i in range(snippet_size):
                cols[i] = f"Vol{i}"
                cols[snippet_size + i] = f"Close{i}"
        cols = cols + ["Pivot_date", "Pivot_value", "Target_value"]
        
        #Create the empty dataframe
        df = pd.DataFrame(columns=cols)

        #We fill the dataframe with snippets
        while True:
                #Randomly select an index number
                pivot = int(data.sample().index.values)

                #Check in order to avoid trying to index outside the dataframe
                if pivot >= len(data) - projection_step or pivot <= snippet_size:
                        continue
                else:
                        #For each snippet or row, we get the relevant info and add it to the df
                        prices = data.iloc[(pivot-snippet_size):pivot, 1]  # 1 is the column number for closing price
                        volumes = data.iloc[(pivot-snippet_size):pivot, 2]  # 2 is for the volume

                        #If more than 5 of the volumes are zero, we assume the index had no information on volumes 
                        #for the selected date and so we avoid working with this row
                        if len([" " for x in volumes if x==0]) >= 5:
                                continue

                        #Scaling of each row
                        if scaling=="minmax":
                                volumes = normalize_minmax(volumes)
                                prices = normalize_minmax(prices)
                        elif scaling=="vector":
                                volumes = normalize_vector(volumes)
                                prices = normalize_vector(prices)
                        elif scaling=="origin":
                                volumes = normalize_origin(volumes)
                                prices = normalize_origin(prices)
                        else:
                                print("No such scaling method.")
                                return 0
                        
                        #Create the metada for each row
                        date = data.iloc[pivot, 0]
                        pivot_value =  data.iloc[pivot, 1]
                        target_value = 100 * (data.iloc[pivot + projection_step, 1] - pivot_value)/pivot_value
                        meta_data = [date, pivot_value, target_value]
                        
                        #Create the row and add it to the dataframe
                        row = np.concatenate((volumes, prices, meta_data))
                        row = pd.Series(data=row, index=cols)
                        df = df.append(row, ignore_index=True)

                #We exit the loop when we have extracted enough "snippets"
                if len(df) == number_of_snippets:
                        break

        return df
                
def create_initial_df(list_of_indexes=None, setup_data=None):

        if not list_of_indexes:
                list_of_indexes = ['NYA', 'IXIC', 'HSI', 'GSPTSE', 'NSEI', 
                'GDAXI', 'KS11', 'SSMI', 'TWII',  'N225', 'N100']

        if not setup_data:
                setup_data = {"number_of_snippets":100, "previous_days":700, 
                "snippet_size":30, "projection_step":10, "scaling":"minmax"}

        dfs = []
        for index in list_of_indexes:
                a = import_index_data(symbol=index)
                a = a.iloc[-(setup_data["previous_days"]):,:]
                a = a.reset_index(drop=True)
                dfs.append(a)

        modeling_dfs=[]
        for df in dfs:
                modeling_dfs.append(create_modeling_df(df, 
                        number_of_snippets=setup_data["number_of_snippets"], 
                        snippet_size=setup_data["snippet_size"], 
                        projection_step=setup_data["projection_step"], 
                        scaling=setup_data["scaling"]))

        #Unify the training datasets into one
        initial_df = pd.concat(modeling_dfs)

        return initial_df

def create_prediction_row(stock_df, pivot_date=None, setup_data=None):
        if not setup_data:
                setup_data = {"number_of_snippets":100, "previous_days":700, 
                "snippet_size":30, "projection_step":10, "scaling":"minmax"}

        #Create the columns that the df will have
        cols = [0]*setup_data["snippet_size"]*2
        for i in range(setup_data["snippet_size"]):
                cols[i] = f"Vol{i}"
                cols[setup_data["snippet_size"] + i] = f"Close{i}"
        cols = cols + ["Pivot_date", "Pivot_value"]

        #We create the row, which is actually a Dataframe
        df = pd.DataFrame(columns=cols)

        if stock_df.empty or len(stock_df) < 30:
                return df

        if not pivot_date:
                pivot_date = stock_df.iloc[-1,0]
        else:
                pivot_date = datetime.strptime(pivot_date,"%Y-%m-%d").date()
                pivot_date = np.datetime64(pivot_date)

        index = stock_df[stock_df["Date"]==pivot_date].index.values

        if len(index) == 1:
                pivot = int(index)
        else:
                print("Pivot date passed unavailable for prediction.")
                return df

        prices = stock_df.iloc[(pivot-setup_data["snippet_size"]):pivot, 1]  # 1 is the column number for closing price
        volumes = stock_df.iloc[(pivot-setup_data["snippet_size"]):pivot, 2]  # 2 is for the volume

        #Scaling of each row
        if setup_data["scaling"]=="minmax":
                volumes = normalize_minmax(volumes)
                prices = normalize_minmax(prices)
        elif setup_data["scaling"]=="vector":
                volumes = normalize_vector(volumes)
                prices = normalize_vector(prices)
        elif setup_data["scaling"]=="origin":
                volumes = normalize_origin(volumes)
                prices = normalize_origin(prices)
        else:
                print("No such scaling method.")
                return 0
        
        #Create the metada for the row
        date = stock_df.iloc[pivot, 0]
        pivot_value =  stock_df.iloc[pivot, 1]
        meta_data = [date, pivot_value]
        
        #Check integrity
        if len(cols) != (len(volumes) + len(prices) + len(meta_data)):
                return df

        #Create the row and add it to the dataframe
        row = np.concatenate((volumes, prices, meta_data))
        row = pd.Series(data=row, index=cols)
        df = df.append(row, ignore_index=True)

        return df




def visualise_clustering_pca(df, init_algo="k-means++", n_clusters=10, n_init=4):

        reduced_data = PCA(n_components=2).fit_transform(df)
        kmeans = KMeans(init=init_algo, n_clusters=n_clusters, n_init=n_init, random_state=0)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape) 
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        plt.title("K-means clustering for index data\n"
                  "Centroids are marked with white cross\n"
                  f"Number of clusters used = {n_clusters}\n"
                  f"Inertia = {kmeans.inertia_:.2f}")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

def elbow_graph(df):
        K = range(2, 31, 2)
        inertia = []

        for k in K:
                time0 = time()
                kmeans = KMeans(n_clusters=k,
                            random_state=1234)
                kmeans.fit(df)
                time_trained = time()-time0
                inertia.append(kmeans.inertia_)
                #print(f"Trained a K-Means model with {k} neighbours! Time needed = {time_trained:.3f} seconds.")

        plt.figure(figsize=(16,8))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.xticks(np.arange(min(K), max(K)+1, 1.0))
        plt.title('Elbow Method showing the optimal k')
        plt.show()


def silhouette_graph(df):
        K = range(2, 31, 2)
        silhouette = []

        for k in K:

                time0 = time()
                kmeans = KMeans(n_clusters=k,
                            random_state=1234)
                kmeans.fit(df)
                silhouette.append(silhouette_score(df, kmeans.predict(df)))
                time_trained = time()- time0

                #print(f"Calculated silhouette with {k} neighbours! Time needed = {time_trained:.3f} seconds.")



        plt.figure(figsize=(16,8))
        plt.plot(K, silhouette, 'bx-')
        plt.xlabel('k')
        plt.ylabel('silhouette score')
        plt.xticks(np.arange(min(K), max(K)+1, 1.0))
        plt.title('Silhouette Method showing the optimal k')
        plt.show()

def visualise_clusters(df, n_per_cluster=5):
        clusters = []
        for i in df["Cluster"].unique():
                clusters.append(df[df["Cluster"] == i])

        for cluster in clusters:
                visualise_trend(cluster.sample(n_per_cluster))

def visualise_trend(df):
        n_plots = len(df)
        n_values = len(df.columns)
        prices = df.iloc[:, int((n_values - 3)/2):-3]
        
        fig, axs = plt.subplots(n_plots, 1, sharex=True)
        for index, row in enumerate(prices.iterrows()):
                axs[index].plot(row[1])
                axs[index].set_xticklabels(row[1].index, rotation="vertical")


def accuracy_score(y_pred, y_test, silent=False):
    """
    Input: the vectors with test and predicted labels.
    Output: a list with the accuracy, the R2 coefficient of determination, and the average error (absolute)
    """
    
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean(errors/abs(y_test))
    accuracy = 100 - mape
    
    return accuracy

def save_clustering_model(model, path="clustering_model.pkl"):

        with open(path, "wb") as f:
                pickle.dump(model,f)
        
        return 0

def load_clustering_model(path="clustering_model.pkl"):

        try:
                with open(path, "rb") as f:
                        model = pickle.load(f)
        except FileNotFoundError: 
                print("Model pickle not found!") 
        
        return model



def get_label(x):
        labels = ["Bad", "Neutral", "Good"]
        if x < -1:
                return labels[0]
        elif x < 3:
                return labels[1]
        else:
                return labels[2]




if __name__ == "__main__":
        run()