# Baseball Player Architects
This code enables users to identify various types of baseball players using unsupervised machine learning algorithms. The data used in this code comes from Baseball Savant and features hundreds of predictors (dimensions) to help cluster players together based on their performance in the MLB. There are three main types of dimensional reduction algorithms used in the project: Principal Component Analysis (PCA), t-distributed Stochastic Neighbor Embedding (t-SNE), and Multidimensional Scaling (MDS). The clustering algorithms used in this project are Affinity Propagation, Agglomerative Clustering, Birch, DBSCAN, Gaussian Mixture, k-Means, Mean Shift, and Spectral Clustering. The clusters were tested using three scoring metrics: Silhouette score, Davies-Bouldin (DB) score, and Calinski-Harabasz (CH) score.

## Original Purpose
This project was the second and final project of the class MTH 4224 Introduction to Machine Learning, taken at Florida Institute of Technology in the spring of 2025. 

## Layout of Repository
Data Folder : contains data used in program, downloaded from Baseball Savant (pitching data found [here](https://baseballsavant.mlb.com/leaderboard/custom?year=2025%2C2024%2C2023%2C2022%2C2021%2C2020%2C2019%2C2018%2C2017%2C2016%2C2015&type=pitcher&filter=&min=q&selections=pa%2Ck_percent%2Cbb_percent%2Cwoba%2Cxwoba%2Csweet_spot_percent%2Cbarrel_batted_rate%2Chard_hit_percent%2Cavg_best_speed%2Cavg_hyper_speed%2Cwhiff_percent%2Cswing_percent&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&sort=xwoba&sortDir=asc) and batting data can be found [here](https://baseballsavant.mlb.com/leaderboard/custom?year=2025%2C2024%2C2023%2C2022%2C2021%2C2020%2C2019%2C2018%2C2017%2C2016%2C2015&type=batter&filter=&min=q&selections=pa%2Ck_percent%2Cbb_percent%2Cwoba%2Cxwoba%2Csweet_spot_percent%2Cbarrel_batted_rate%2Chard_hit_percent%2Cavg_best_speed%2Cavg_hyper_speed%2Cwhiff_percent%2Cswing_percent&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&sort=xwoba&sortDir=desc)) 

src : source code for the project

LICENSE : MIT License

README.md : Read all above

Wainscott_Tim_MTH4224_Project_2_Report.pdf : Final report of the project 

## Expected Results
When running the program, the expected graphs you should get should look something like the ones below:

