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

pitcher_data_url = 'https://raw.githubusercontent.com/twainscott21/Baseball_Player_Architects/refs/heads/main/Data/MLB-Pitchers.csv'
response = requests.get(pitcher_data_url)

if response.status_code == 200:
    pitchers = pd.read_csv(StringIO(response.text))    
    print(pitchers)
else:
    raise FileNotFoundError(f"Data could not be imported: Code {response.status_code}")

drop_cols_pit = pitchers.columns[pitchers.isnull().sum() > 0.25*len(pitchers)]

pitchers = pitchers.drop(columns = drop_cols_pit, axis = 1)
pitchers = pitchers.dropna(axis = 0, how = "any")
pitchers_new = pitchers.drop(columns=["last_name, first_name", "player_id", "year"])

hand_pitch = LabelEncoder()
pitchers_new["pitch_hand"] = hand_pitch.fit_transform(pitchers_new["pitch_hand"])

scaler_pitcher = MinMaxScaler()
scaler_pitcher.fit(pitchers_new)
pitchers_new_s = scaler_pitcher.fit_transform(pitchers_new)


### Testing Clustering Without Dimensional Reduction ###

eval_pitchers_nd = run_clustering_evaluation(pitchers_new_s)

top_6_sil_score_pitchers_nd = eval_pitchers_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_pitchers_nd = eval_pitchers_nd.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_pitchers_nd = eval_pitchers_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

top_sil_score_pitchers_nd = eval_pitchers_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_pitchers_nd = eval_pitchers_nd.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_pitchers_nd = eval_pitchers_nd.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

print()
print('\nBest Silhouette Scores:',top_sil_score_pitchers_nd)
print('\nBest DB Scores:',top_db_score_pitchers_nd)
print('\nBest CH Scores:',top_ch_score_pitchers_nd)

### Performing DR and Clustering ###

## PCA ##

pitchers_new_s = np.array(pitchers_new_s)

pca_3d = PCA(n_components = 3, random_state = 42)
pca_3d.fit(pitchers_new_s)

pitchers_new_s_3d_pca = pca_3d.transform(pitchers_new_s)

eval_pitchers_3d_pca = run_clustering_evaluation(pitchers_new_s_3d_pca)

top_6_sil_score_pitchers_3d_pca = eval_pitchers_3d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_pitchers_3d_pca = eval_pitchers_3d_pca.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_pitchers_3d_pca = eval_pitchers_3d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

graph_6_plot(top_6_sil_score_pitchers_3d_pca, pitchers_new_s_3d_pca, 'silhouette_score', save_plots = True, disp_plots = False)
graph_6_plot(top_6_db_score_pitchers_3d_pca, pitchers_new_s_3d_pca, 'db_score', save_plots = True, disp_plots = False)
graph_6_plot(top_6_ch_score_pitchers_3d_pca, pitchers_new_s_3d_pca, 'ch_score', save_plots = True, disp_plots = False)

top_sil_score_pitchers_3d_pca = eval_pitchers_3d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_pitchers_3d_pca = eval_pitchers_3d_pca.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_pitchers_3d_pca = eval_pitchers_3d_pca.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

graph_6_plot(top_sil_score_pitchers_3d_pca, pitchers_new_s_3d_pca, 'silhouette_score', plot_all = True, dr_method = "PCA", group_class = "Pitchers", save_plots = True, disp_plots = False)
graph_6_plot(top_db_score_pitchers_3d_pca, pitchers_new_s_3d_pca, 'db_score', plot_all = True, dr_method = "PCA", group_class = "Pitchers", save_plots = True, disp_plots = False)
graph_6_plot(top_ch_score_pitchers_3d_pca, pitchers_new_s_3d_pca, 'ch_score', plot_all = True, dr_method = "PCA", group_class = "Pitchers", save_plots = True, disp_plots = False)
print("PCA Done")


## t-SNE ##

tsne_3d = TSNE(n_components = 3, random_state = 42)
tsne_3d.fit(pitchers_new_s)

pitchers_new_s_3d_tsne = tsne_3d.fit_transform(pitchers_new_s)

eval_pitchers_3d_tsne = run_clustering_evaluation(pitchers_new_s_3d_tsne)

top_6_sil_score_pitchers_3d_tsne = eval_pitchers_3d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_pitchers_3d_tsne = eval_pitchers_3d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_pitchers_3d_tsne = eval_pitchers_3d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

graph_6_plot(top_6_sil_score_pitchers_3d_tsne, pitchers_new_s_3d_tsne,'silhouette_score', save_plots = True, disp_plots = False)
graph_6_plot(top_6_db_score_pitchers_3d_tsne, pitchers_new_s_3d_tsne,'db_score', save_plots = True, disp_plots = False)
graph_6_plot(top_6_ch_score_pitchers_3d_tsne, pitchers_new_s_3d_tsne,'ch_score', save_plots = True, disp_plots = False)

top_sil_score_pitchers_3d_tsne = eval_pitchers_3d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_pitchers_3d_tsne = eval_pitchers_3d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_pitchers_3d_tsne = eval_pitchers_3d_tsne.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

graph_6_plot(top_sil_score_pitchers_3d_tsne, pitchers_new_s_3d_tsne, 'silhouette_score', plot_all = True, dr_method = "t-SNE", group_class = "Pitchers", save_plots = True, disp_plots = False)
graph_6_plot(top_db_score_pitchers_3d_tsne, pitchers_new_s_3d_tsne, 'db_score', plot_all = True, dr_method = "t-SNE", group_class = "Pitchers", save_plots = True, disp_plots = False)
graph_6_plot(top_ch_score_pitchers_3d_tsne, pitchers_new_s_3d_tsne, 'ch_score', plot_all = True, dr_method = "t-SNE", group_class = "Pitchers", save_plots = True, disp_plots = False)
print("t-SNE Done")

## MDS ##

mds_3d = MDS(n_components = 3, random_state = 42)
mds_3d.fit(pitchers_new_s)

pitchers_new_s_3d_mds = mds_3d.fit_transform(pitchers_new_s)

eval_pitchers_3d_mds = run_clustering_evaluation(pitchers_new_s_3d_mds)

top_6_sil_score_pitchers_3d_mds = eval_pitchers_3d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'silhouette_score'))
top_6_db_score_pitchers_3d_mds = eval_pitchers_3d_mds.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(6, 'db_score'))
top_6_ch_score_pitchers_3d_mds = eval_pitchers_3d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(6, 'ch_score'))

graph_6_plot(top_6_sil_score_pitchers_3d_mds, pitchers_new_s_3d_mds, 'silhouette_score', save_plots = True, disp_plots = False)
graph_6_plot(top_6_db_score_pitchers_3d_mds, pitchers_new_s_3d_mds, 'db_score', save_plots = True, disp_plots = False)
graph_6_plot(top_6_ch_score_pitchers_3d_mds, pitchers_new_s_3d_mds, 'ch_score', save_plots = True, disp_plots = False)

top_sil_score_pitchers_3d_mds = eval_pitchers_3d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'silhouette_score'))
top_db_score_pitchers_3d_mds = eval_pitchers_3d_mds.groupby('method', group_keys = False).apply(lambda g: g.nsmallest(1, 'db_score'))
top_ch_score_pitchers_3d_mds = eval_pitchers_3d_mds.groupby('method', group_keys = False).apply(lambda g: g.nlargest(1, 'ch_score'))

graph_6_plot(top_sil_score_pitchers_3d_mds, pitchers_new_s_3d_mds, 'silhouette_score', plot_all = True, dr_method = "MDS", group_class = "Pitchers", save_plots = True, disp_plots = False)
graph_6_plot(top_db_score_pitchers_3d_mds, pitchers_new_s_3d_mds, 'db_score', plot_all = True, dr_method = "MDS", group_class = "Pitchers", save_plots = True, disp_plots = False)
graph_6_plot(top_ch_score_pitchers_3d_mds, pitchers_new_s_3d_mds, 'ch_score', plot_all = True, dr_method = "MDS", group_class = "Pitchers", save_plots = True, disp_plots = False)

pitcher_feature_names = pitchers_new.columns.tolist()
pitcher_custom_features = ["k_percent", "bb_percent","whiff_percent","p_era","batting_avg","woba","p_out_ground","babip","xobp"]
print("MDS Done")

variance_plotting(pitchers_new_s_3d_mds, pitchers_new_s, n_clusters = 9, feature_names = pitcher_feature_names,
                  n_parallel_features = len(pitcher_custom_features), custom_features=pitcher_custom_features, is_3d = True)