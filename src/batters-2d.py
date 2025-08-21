### Importing Packages ####

import numpy as np
import pandas as pd
import requests
import warnings
from io import StringIO
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

from custom_funcs import run_clustering_evaluation, graph_6_plot, variance_plotting

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


### Importing + Preprocessing Data ###

batter_data_url = 'https://raw.githubusercontent.com/twainscott21/Baseball_Player_Architects/refs/heads/main/Data/MLB-Batters.csv'
response = requests.get(batter_data_url)

if response.status_code == 200:
    batters = pd.read_csv(StringIO(response.text))    
    print(batters)
else:
    raise FileNotFoundError(f"Data could not be imported: Code {response.status_code}")

drop_cols_bat = batters.columns[batters.isnull().sum() > 0.25*len(batters)]

batters = batters.drop(columns = drop_cols_bat, axis = 1)
batters = batters.dropna(axis = 0, how = "any")
batters_new = batters.drop(columns=["last_name, first_name", "player_id", "year"])

scaler_batter = MinMaxScaler()
scaler_batter.fit(batters_new)
batters_new_s = scaler_batter.fit_transform(batters_new)


### Testing Clustering Without Dimensional Reduction ###

eval_batters_nd = run_clustering_evaluation(batters_new_s)

top_6_sil_score_batters_nd = eval_batters_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_batters_nd = eval_batters_nd.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_batters_nd = eval_batters_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

top_sil_score_batters_nd = eval_batters_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_batters_nd = eval_batters_nd.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_batters_nd = eval_batters_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

print()
print('\nBest Silhouette Scores:',top_sil_score_batters_nd)
print('\nBest DB Scores:',top_db_score_batters_nd)
print('\nBest CH Scores:',top_ch_score_batters_nd)

### Performing DR and Clustering ###

## PCA ##

batters_new_s = np.array(batters_new_s)

pca_2d = PCA(n_components = 2, random_state = 42)
pca_2d.fit(batters_new_s)

batters_new_s_2d_pca = pca_2d.transform(batters_new_s)

eval_batters_2d_pca = run_clustering_evaluation(batters_new_s_2d_pca)

top_6_sil_score_batters_2d_pca = eval_batters_2d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_batters_2d_pca = eval_batters_2d_pca.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_batters_2d_pca = eval_batters_2d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

graph_6_plot(top_6_sil_score_batters_2d_pca, batters_new_s_2d_pca, 'silhouette_score', save_plots = False, disp_plots = True)
graph_6_plot(top_6_db_score_batters_2d_pca, batters_new_s_2d_pca, 'db_score', save_plots = False, disp_plots = True)
graph_6_plot(top_6_ch_score_batters_2d_pca, batters_new_s_2d_pca, 'ch_score', save_plots = False, disp_plots = True)

top_sil_score_batters_2d_pca = eval_batters_2d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_batters_2d_pca = eval_batters_2d_pca.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_batters_2d_pca = eval_batters_2d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

graph_6_plot(top_sil_score_batters_2d_pca, batters_new_s_2d_pca, 'silhouette_score', plot_all = True, dr_method = "PCA", group_class = "Batters", save_plots = False, disp_plots = True)
graph_6_plot(top_db_score_batters_2d_pca, batters_new_s_2d_pca, 'db_score', plot_all = True, dr_method = "PCA", group_class = "Batters", save_plots = False, disp_plots = True)
graph_6_plot(top_ch_score_batters_2d_pca, batters_new_s_2d_pca, 'ch_score', plot_all = True, dr_method = "PCA", group_class = "Batters", save_plots = False, disp_plots = True)
print("PCA Done")


## t-SNE ##

tsne_2d = TSNE(n_components = 2, random_state = 42)
tsne_2d.fit(batters_new_s)

batters_new_s_2d_tsne = tsne_2d.fit_transform(batters_new_s)

eval_batters_2d_tsne = run_clustering_evaluation(batters_new_s_2d_tsne)


top_6_sil_score_batters_2d_tsne = eval_batters_2d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_batters_2d_tsne = eval_batters_2d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_batters_2d_tsne = eval_batters_2d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

graph_6_plot(top_6_sil_score_batters_2d_tsne, batters_new_s_2d_tsne,'silhouette_score', save_plots = False, disp_plots = True)
graph_6_plot(top_6_db_score_batters_2d_tsne, batters_new_s_2d_tsne,'db_score', save_plots = False, disp_plots = True)
graph_6_plot(top_6_ch_score_batters_2d_tsne, batters_new_s_2d_tsne,'ch_score', save_plots = False, disp_plots = True)


top_sil_score_batters_2d_tsne = eval_batters_2d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_batters_2d_tsne = eval_batters_2d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_batters_2d_tsne = eval_batters_2d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

graph_6_plot(top_sil_score_batters_2d_tsne, batters_new_s_2d_tsne, 'silhouette_score', plot_all = True, dr_method = "t-SNE", group_class = "Batters", save_plots = False, disp_plots = True)
graph_6_plot(top_db_score_batters_2d_tsne, batters_new_s_2d_tsne, 'db_score', plot_all = True, dr_method = "t-SNE", group_class = "Batters", save_plots = False, disp_plots = True)
graph_6_plot(top_ch_score_batters_2d_tsne, batters_new_s_2d_tsne, 'ch_score', plot_all = True, dr_method = "t-SNE", group_class = "Batters", save_plots = False, disp_plots = True)
print("t-SNE Done")

## MDS ##

mds_2d = MDS(n_components = 2, random_state = 42)
mds_2d.fit(batters_new_s)

batters_new_s_2d_mds = mds_2d.fit_transform(batters_new_s)

eval_batters_2d_mds = run_clustering_evaluation(batters_new_s_2d_mds)

top_6_sil_score_batters_2d_mds = eval_batters_2d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_batters_2d_mds = eval_batters_2d_mds.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_batters_2d_mds = eval_batters_2d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

graph_6_plot(top_6_sil_score_batters_2d_mds, batters_new_s_2d_mds, 'silhouette_score', save_plots = False, disp_plots = True)
graph_6_plot(top_6_db_score_batters_2d_mds, batters_new_s_2d_mds, 'db_score', save_plots = False, disp_plots = True)
graph_6_plot(top_6_ch_score_batters_2d_mds, batters_new_s_2d_mds, 'ch_score', save_plots = False, disp_plots = True)

top_sil_score_batters_2d_mds = eval_batters_2d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_batters_2d_mds = eval_batters_2d_mds.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_batters_2d_mds = eval_batters_2d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

graph_6_plot(top_sil_score_batters_2d_mds, batters_new_s_2d_mds, 'silhouette_score', plot_all = True, dr_method = "MDS", group_class = "Batters", save_plots = False, disp_plots = True)
graph_6_plot(top_db_score_batters_2d_mds, batters_new_s_2d_mds, 'db_score', plot_all = True, dr_method = "MDS", group_class = "Batters", save_plots = False, disp_plots = True)
graph_6_plot(top_ch_score_batters_2d_mds, batters_new_s_2d_mds, 'ch_score', plot_all = True, dr_method = "MDS", group_class = "Batters", save_plots = False, disp_plots = True)

batter_feature_names = batters_new.columns.tolist()
batter_custom_features = ["k_percent", "bb_percent","whiff_percent","p_era","batting_avg","woba","p_out_ground","babip","xobp"]
print("MDS Done")

variance_plotting(batters_new_s_2d_mds, batters_new_s, n_clusters = 9, feature_names = batter_feature_names,
                  n_parallel_features = len(batter_custom_features), custom_features=batter_custom_features, is_2d = True)