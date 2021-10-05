from helper_functions import *


setup_data = {"number_of_snippets":100, "previous_days":700, "snippet_size":30, "projection_step":20, "scaling":"minmax"}

indexes = ['NYA', 'IXIC', 'HSI', 'GSPTSE', 'NSEI', 'GDAXI', 'KS11', 'SSMI', 'TWII',  'N225', 'N100']

def main():
        dfs = []
        for index in indexes:
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
        final_df = pd.concat(modeling_dfs)

        #Clustering

        #Save Dataframe with clusters

        #Regression model

        #Classification model

        #Save models


if __name__=="__main__":
        main()



