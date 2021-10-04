import pandas as pd
import numpy as np
from time import time
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, r2_score
import matplotlib.pyplot as plt

def normalize_vector(data):
        norm = np.linalg.norm(data)
        return data/norm

def normalize_minmax(data):
        m = data.min()
        M = data.max()
        norm_list = [(x-m)/(M-m) for x in data]
        return np.array(norm_list)

def normalize_origin(data):
        bias = data[0]
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

def create_modeling_df(data_input, number_of_snippets, snippet_size=20, projection_step=10, scaling="minmax"):
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
                        target_value = data.iloc[pivot + projection_step, 1]
                        meta_data = [date, pivot_value, target_value]
                        
                        #Create the row and add it to the dataframe
                        row = np.concatenate((volumes, prices, meta_data))
                        row = pd.Series(data=row, index=cols)
                        df = df.append(row, ignore_index=True)

                #We exit the loop when we have extracted enough "snippets"
                if len(df) == number_of_snippets:
                        break

        return df
                




"""
def _create_model(df, init_algo="k-means++", n_clusters=10, n_init=4):
        scaler = StandardScaler()
        kmeans = KMeans(init=init_algo, n_clusters=n_clusters, n_init=n_init, random_state=0)
        t0 = time()
        print("Initiating fit...")
        model = make_pipeline(scaler, kmeans).fit(df)
        fit_time = time() - t0
        print(f"Fit ended in {fit_time:.3f} seconds.")
        results = (model, model[-1].inertia_, fit_time)
        return results

def save_model(model, path="music_model.pkl"):
        kmeans_model = model[-1]
        save_text = f"Model saved - {kmeans_model}\nInertia = {kmeans_model.inertia_:.2f}\n"
        time_text = str(datetime.now())[:-10] + "h" + "\n"
        file_text = f"Filename: {path}\n"

        with open(path, "wb") as f:
                pickle.dump(model,f)

        with open("model_log.txt", "a") as f:
                f.write("--------------\n" + save_text + time_text + file_text)
        
        return 0

def load_model(path="music_model.pkl"):

        try:
                with open(path, "rb") as f:
                        model = pickle.load(f)
        except FileNotFoundError: 
                print("Model pickle not found!") 
        
        return model
"""

def cluster_song():
        return 0

def visualise_model(df, init_algo="k-means++", n_clusters=10, n_init=4):

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
        K = range(2, 41, 2)
        inertia = []

        for k in K:
                time0 = time()
                kmeans = KMeans(n_clusters=k,
                            random_state=1234)
                kmeans.fit(df)
                time_trained = time()-time0
                inertia.append(kmeans.inertia_)
                print(f"Trained a K-Means model with {k} neighbours! Time needed = {time_trained:.3f} seconds.")

        plt.figure(figsize=(16,8))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.xticks(np.arange(min(K), max(K)+1, 1.0))
        plt.title('Elbow Method showing the optimal k')
        plt.show()

        return 0

def silhouette_graph(df):
        K = range(2, 41, 2)
        silhouette = []

        for k in K:

                time0 = time()
                kmeans = KMeans(n_clusters=k,
                            random_state=1234)
                kmeans.fit(df)
                silhouette.append(silhouette_score(df, kmeans.predict(df)))
                time_trained = time()- time0

                print(f"Calculated silhouette with {k} neighbours! Time needed = {time_trained:.3f} seconds.")



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
                clusters.append(df[df["Cluster"] == i].drop("Cluster", axis=1))

        for cluster in clusters:
                visualise_trend(cluster.sample(n_per_cluster))

def visualise_trend(df, vol_data=True):
        n_plots = len(df)
        n_values = len(df.columns)

        if vol_data:
                prices = df.iloc[:, int((n_values -3)/2):-2]
        else:
                prices = df.iloc[:,:-2]

        fig, axs = plt.subplots(n_plots, 1, sharex=True)
        for index, row in enumerate(prices.iterrows()):
                axs[index].plot(row[1])
                axs[index].set_xticklabels(row[1].index, rotation="vertical")

        return 0

def evaluate(model, X_test, y_test, silent=False):
    """
    Input: a regressor, the matrix with test features, the vector with test labels.
    Output: a list with the accuracy, the R2 coefficient of determination, and the average error (absolute)
    """
    
    y_predict = model.predict(X_test)
    errors = abs(y_predict - y_test)
    mape = 100 * np.mean(errors / y_test)
    avg_error = np.mean(errors)
    accuracy = 100 - mape
    r2 = r2_score(y_test, y_predict)
    if not silent:
        print('---Model Performance---')
        print(model)
        #print(X_test.columns)
        print('\nAverage Absolute Error: {:0.1f} dollars.'.format(np.mean(errors)))
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_predict, squared=False))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f\n' % r2)
    
    return [accuracy, r2, avg_error]

def run():
        df = import_df(path="spotify_songs.csv")
        modeling_df = df.drop(columns=["song_name", "song_id", "artist_name", "artist_id"])
        
        #elbow_graph(modeling_df)
        #silhouette_graph(modeling_df)
        #visualise_model(modeling_df, n_clusters=15)

        model, inertia, fit_time = create_model(modeling_df, n_clusters=15)

        save_model(model, path="music_model.pkl")

        kmeans_model = model[-1]

        print(f"Results for {kmeans_model}: inertia = {inertia:.2f}; fit_time = {fit_time:.3f}")

        return 0

if __name__ == "__main__":
        run()