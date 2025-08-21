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

<img width="1475" height="1025" alt="image" src="https://github.com/user-attachments/assets/1562d19b-db16-4773-9904-d0baf039e8bd" />

<img width="1976" height="1025" alt="image" src="https://github.com/user-attachments/assets/82c58008-722a-4c1a-ad0e-79c8a1f27cdc" />

<img width="1353" height="941" alt="image" src="https://github.com/user-attachments/assets/daafd55b-1125-4375-b396-f46e96d03e10" />

<img width="1486" height="1498" alt="image" src="https://github.com/user-attachments/assets/cb576ceb-6c20-4efa-a1a1-a894a3ffb1ab" />

<img width="1395" height="590" alt="image" src="https://github.com/user-attachments/assets/8e0b7c6c-10fb-40df-99d1-9dd13c9e328f" />

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/38889958-899a-4769-9ed9-6ae8fd96992f" />

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/822f1fc5-6780-4343-8912-310b9768262a" />


