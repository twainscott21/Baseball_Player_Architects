import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import json
import warnings
from tqdm import tqdm

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import (KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, 
                             OPTICS, estimate_bandwidth, Birch, AffinityPropagation, MeanShift)
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from pandas.plotting import parallel_coordinates


model_map = {
    "KMeans": KMeans,
    "DBSCAN": DBSCAN,
    "OPTICS": OPTICS,
    "MeanShift": MeanShift,
    "SpectralClustering": SpectralClustering,
    "AgglomerativeClustering": AgglomerativeClustering,
    "Birch": Birch,
    "AffinityPropagation": AffinityPropagation,
    "GaussianMixture": GaussianMixture
}

def param_string_to_dict(param_string):
    if not param_string or pd.isna(param_string):
        return {}
    
    if 'bandwidth=' in param_string:
        bandwidth_part = [p for p in param_string.split(',') if 'bandwidth=' in p.strip()][0]
        key, value = bandwidth_part.strip().split('=', 1)
        try:
            return {'bandwidth': float(value.strip())}
        except ValueError:
            return {'bandwidth': value.strip()}
    
    if param_string.startswith('{') and param_string.endswith('}'):
        return json.loads(param_string.replace("'", '"'))
    
    params = {}
    for item in param_string.split(','):
        item = item.strip()
        if '=' in item:
            key, value = item.split('=', 1)
            try:
                params[key.strip()] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                params[key.strip()] = value.strip()
    return params


def evaluate_clustering(X, labels):
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if n_labels < 2 or n_labels >= len(X):
        return {'silhouette_score': -1, 'db_score': np.inf, 'ch_score': -1}
    return {'silhouette_score': silhouette_score(X, labels),
            'db_score': davies_bouldin_score(X, labels),
            'ch_score': calinski_harabasz_score(X, labels)
    }

def run_clustering_evaluation(X):
    results = []

    for k in tqdm(range(2, 11)):
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
        scores = evaluate_clustering(X, labels)
        results.append({'method': 'KMeans', 'param': f'n_clusters = {k}', **scores})
    
    print("Kmeans done  = 1/9")

    for k in range(2, 11):
        model = AgglomerativeClustering(n_clusters = k)
        labels = model.fit_predict(X)
        scores = evaluate_clustering(X, labels)
        results.append({'method': 'AgglomerativeClustering', 'param': f'n_clusters = {k}', **scores})

    print("AgglomerativeClustering done  = 2/9")
    

    for k in range(2, 11):
        model = SpectralClustering(n_clusters = k, random_state = 0, affinity = 'nearest_neighbors')
        labels = model.fit_predict(X)
        scores = evaluate_clustering(X, labels)
        results.append({'method': 'SpectralClustering', 'param': f'n_clusters = {k}', **scores})

    print("SpectralClustering done  = 3/9")


    for k in range(2, 11):
        model = GaussianMixture(n_components = k, random_state = 0)
        labels = model.fit(X).predict(X)
        scores = evaluate_clustering(X, labels)
        results.append({'method': 'GaussianMixture', 'param': f'n_components = {k}', **scores})

    print("GaussianMixture done  = 4/9")

    eps_vals = np.arange(1, 30, 2, dtype = "int")
    min_samples_vals = np.arange(2, 21, dtype = "int")
    for eps in tqdm(eps_vals):
        for min_samples_val in min_samples_vals:
            model = DBSCAN(eps = eps, min_samples = min_samples_val)
            labels = model.fit_predict(X)
            scores = evaluate_clustering(X, labels)
            results.append({
                'method': 'DBSCAN',
                'param': f'eps = {eps:.2f}, min_samples = {min_samples_val}',
                **scores
            })

            #optics = OPTICS(eps = eps, min_samples = min_samples, cluster_method = "dbscan")
            #labels = optics.fit_predict(X)
            #scores = evaluate_clustering(X, labels)
            #results.append({
            #    'method': 'OPTICS',
            #    'param': f'eps = {eps:.2f}, min_samples = {min_samples_val}',
            #    **scores
            #})

    print("DBSCAN done  = 5/9")
    print("OPTICS done  = 6/9")
    

    for quantile in tqdm([0.1, 0.2, 0.3]):
        try:
            bandwidth = estimate_bandwidth(X, quantile = quantile)
            if bandwidth <= 0 or np.isnan(bandwidth):
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = ConvergenceWarning)
                model = MeanShift(bandwidth = bandwidth, bin_seeding = True)
                labels = model.fit_predict(X)

            scores = evaluate_clustering(X, labels)
            results.append({
                'method': 'MeanShift',
                'param': f'quantile = {quantile:.2f}, bandwidth = {bandwidth:.2f}',
                **scores
            })

        except ValueError as e:
            print(f"MeanShift failed with bandwidth = {bandwidth:.4f}: {e}")
            continue

    print("MeanShift done  = 7/9")

    for k in tqdm(range(2, 11)):
        model = Birch(n_clusters = k)
        labels = model.fit_predict(X)
        scores = evaluate_clustering(X, labels)
        results.append({'method': 'Birch', 'param': f'n_clusters = {k}', **scores})

    print("Birch done  = 8/9")


    model = AffinityPropagation(random_state = 0)
    labels = model.fit_predict(X)
    scores = evaluate_clustering(X, labels)
    results.append({'method': 'AffinityPropagation', 'param': 'default', **scores})

    print("AffinityPropagation done  = 9/9")

    return pd.DataFrame(results)

def graph_6_plot(group, data, score_metric, plot_all = False, dr_method = "PCA", group_class = "Pitchers", save_plots = False, disp_plots = True):
    is_3d = data.shape[1] == 3
    
    def plot_clusters(ax, data, labels, params, method = None, score = None):
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                if is_3d:
                    ax.scatter(data[labels == label, 0], data[labels == label, 1],
                              data[labels == label, 2], c = 'black', marker = 'x', s = 10, 
                              alpha = 0.6, label = 'Noise')
                else:
                    ax.scatter(data[labels == label, 0],data[labels == label, 1],
                               c = 'black', marker = 'x', s = 10, alpha = 0.6, label = 'Noise')
            else:
                if is_3d:
                    ax.scatter(data[labels == label, 0], data[labels == label, 1],
                              data[labels == label, 2], c = [color], s = 20, label = f'Cluster {label}')
                else:
                    ax.scatter(data[labels == label, 0], data[labels == label, 1],
                              c = [color], s = 20, label = f'Cluster {label}')
        
        param_str = str(params)
        title = f"{method + ': ' if method else 'default'}{param_str}"
        if score is not None:
            title += f"\n{score_metric}: {score:.3f}"
        ax.set_title(title, fontsize = 10)
        
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        if is_3d:
            ax.set_zlabel(r"$x_3$")
        
        if len(unique_labels) <= 10:
            ax.legend(bbox_to_anchor = (1.15, 1), loc = 'upper left')

    if plot_all:
        n = len(group)
        cols = 4
        rows = 2
        
        fig = plt.figure(figsize = (5*cols, 5*rows))
        title = f"Best Results for {score_metric.replace('_', ' ').title()}" if score_metric else "Best Results"
        title = title + f"\nDimension Reduction: {dr_method}\n Player Type: {group_class}"
        fig.suptitle(title, fontsize = 14, y = 1.02)
        
        axes = []
        for i in range(1, n+1):
            ax = fig.add_subplot(rows, cols, i, projection = '3d' if is_3d else None)
            axes.append(ax)
        
        for _, (_, row), ax in zip(range(n), group.iterrows(), axes):
            params = param_string_to_dict(row['param'])
            score = row[score_metric] if score_metric else None
            method = row['method']
            
            Model = model_map.get(method)
            if Model is None:
                ax.set_title(f"Unknown method: {method}")
                continue
                
            try:
                if method == "MeanShift":
                    bandwidth = params.get("bandwidth")
                    bandwidth = estimate_bandwidth(data, quantile=params["quantile"])  # or your preferred quantile
                    params["bandwidth"] = bandwidth

                model = Model(**params)
                labels = model.fit_predict(data)
                plot_clusters(ax, data, labels, params, method, score)
                if save_plots:
                    filename = f"{group_class}-{dr_method}-{method}-{score_metric}-{'all' if plot_all else 'top'}-{'3d' if is_3d else '2d'}.png"
                    plt.savefig(filename, dpi=300, bbox_inches="tight")

                if disp_plots:
                    plt.show()
                else:
                    plt.close(fig)
            except Exception as e:
                ax.set_title(f"Failed: {params}", fontsize = 10)
                err_text = f"Error:\n{str(e)}"
                if is_3d:
                    ax.text(0.5, 0.5, 0.5, err_text, ha = 'center', va = 'center')
                else:
                    ax.text(0.5, 0.5, err_text, ha = 'center', va = 'center', transform = ax.transAxes)
    
    else:
        for method, method_group in group.groupby('method'):
            n = len(method_group)
            if n == 0:
                continue
                            
            cols = min(3, n)
            rows = (n // cols) + (n % cols > 0)
            
            fig = plt.figure(figsize = (5*cols, 5*rows))
            title = f"Clustering Results: {method}"
            if score_metric:
                title += f" (by {score_metric.replace('_', ' ').title()})"
            fig.suptitle(title, fontsize = 14, y = 1.02)
            
            axes = []
            for i in range(1, n+1):
                ax = fig.add_subplot(rows, cols, i, projection = '3d' if is_3d else None)
                axes.append(ax)
            
            for _, (_, row), ax in zip(range(n), method_group.iterrows(), axes):
                params = param_string_to_dict(row['param'])
                score = row[score_metric] if score_metric else None
                
                try:
                    if method == "MeanShift":
                        bandwidth = params.get("bandwidth")
                        bandwidth = estimate_bandwidth(data, quantile=params["quantile"])  
                        params["bandwidth"] = bandwidth

                    model = model_map[method](**params)
                    labels = model.fit_predict(data)
                    plot_clusters(ax, data, labels, params, score = score)

                    if save_plots:
                        filename = f"{group_class}-{dr_method}-{method}-{score_metric}-{'all' if plot_all else 'top'}-{'3d' if is_3d else '2d'}.png"
                        plt.savefig(filename, dpi=300, bbox_inches="tight")

                    if disp_plots:
                        plt.show()
                    else:
                        plt.close(fig)
                except Exception as e:
                    ax.set_title(f"Failed: {params}", fontsize = 10)
                    err_text = f"Error:\n{str(e)}"
                    if is_3d:
                        ax.text(0.5, 0.5, 0.5, err_text, ha = 'center', va = 'center')
                    else:
                        ax.text(0.5, 0.5, err_text, ha = 'center', va = 'center', transform = ax.transAxes)
    
    if 'axes' in locals():
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout()

    if disp_plots:
        plt.show()
    else:
        plt.close()

def variance_plotting(data, original_features, n_clusters = 3, feature_names = None, 
                      n_parallel_features = 5, custom_features = None, is_3d = False):
    
    if isinstance(original_features, np.ndarray):
        original_features = pd.DataFrame(original_features, columns = feature_names)
   
    kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
    labels = kmeans.fit_predict(data)
    
    important_features = [f for f in custom_features if f in feature_names]
    if len(important_features) < len(custom_features):
        print(f"Warning: {len(custom_features)-len(important_features)} custom features not found")
    plot_features = important_features[:n_parallel_features]

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = '3d' if is_3d else None)
    
    if not is_3d:
        scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'tab20')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
    else:
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = labels, cmap = 'tab20')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$x_3$')
    
    legend_labels = [f'Cluster {i}' for i in range(n_clusters)]
    ax.legend(handles = scatter.legend_elements()[0], 
               labels = legend_labels,
               bbox_to_anchor = (1.05, 1), 
               loc = 'upper left')
    ax.set_title('Dimensionally Reduced Projection with Clusters')
    plt.show()

    plt.figure(figsize = (14, 6))
    plot_df = original_features[plot_features].copy()
    plot_df['Cluster'] = labels
    
    for feature in plot_features:
        plot_df[feature] = (plot_df[feature] - plot_df[feature].min()) / (plot_df[feature].max() - plot_df[feature].min())
    
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    parallel_coordinates(plot_df,'Cluster',color = cluster_colors, alpha = 0.15, linewidth = 0.3)
    
    cluster_means = plot_df.groupby('Cluster').mean().reset_index()
    for cluster_idx in range(n_clusters):
        cluster_data = cluster_means[cluster_means['Cluster'] == cluster_idx]
        plt.plot(
            cluster_data.drop('Cluster', axis = 1).values.flatten(),
            label = f'Cluster {cluster_idx} Mean',
            linewidth = 3,
            alpha = 0.9,
            color = cluster_colors[cluster_idx],
            marker ='o',
            markersize = 8
        )
    
    plt.title(f'Cluster Characteristics ({n_parallel_features} Select Features)')
    plt.xticks(range(len(plot_features)), plot_features, rotation = 45, ha = 'right')
    plt.grid(alpha = 0.2)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (14, 6))
    plot_df = original_features[plot_features].copy()
    plot_df['Cluster'] = labels
    
    for feature in plot_features:
        plot_df[feature] = (plot_df[feature] - plot_df[feature].min()) / (plot_df[feature].max() - plot_df[feature].min())
    
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    parallel_coordinates(plot_df,'Cluster', color = cluster_colors, alpha = 0.8, linewidth = 0.7)
    
    plt.title(f'Cluster Characteristics ({n_parallel_features} Select Features)')
    plt.xticks(range(len(plot_features)), plot_features, rotation = 45, ha = 'right')
    plt.grid(alpha = 0.2)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.tight_layout()
    plt.show()