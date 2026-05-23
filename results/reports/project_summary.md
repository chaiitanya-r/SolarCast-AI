Predictive Analytics — Pipeline Run Summary
Environment: NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB) | CUDA
________________________________________
Stage Results
1. Data Cleaning Reduced 78,138 → 45,584 rows (32,554 night rows removed, 0 duplicates, 0 invalid). Output: 45,584 rows × 7 cols.
2. Feature Engineering Output: 45,582 rows × 19 cols.
3. Scaling + Split Chronological split — Train: 36,465 | Test: 9,117. Scaled 8 features.
4. Classification Models
Model	Accuracy	F1	ROC-AUC	Train Time
Logistic Regression	0.9300	0.9300	0.9870	0.3s
Naive Bayes	0.7683	0.7659	0.9283	0.0s
KNN	0.9300	0.9301	0.9795	0.1s
SVM	0.9387	0.9388	0.9900	34.5s
Decision Tree	0.9286	0.9287	0.9795	0.1s
Random Forest	0.9344	0.9345	0.9894	5.9s
Gradient Boosting	0.9392	0.9393	0.9904	14.1s
XGBoost	0.9397	0.9397	0.9911	1.1s
5. KMeans Clustering Sampled 15,000 rows. Selected k=2.
6. KMedoids Clustering Compared KMedoids vs KMeans centers at k=3.
7. Hierarchical Clustering Dendrogram generated. (Warning: ARI could not be computed — index out of bounds.)
8. Gaussian Mixture Model Sampled 12,000 rows. Fitted with k=3 components.
9. ANN Early stopped at epoch 41/500. RMSE: 43.63 | R²: 0.9725 | Time: 63.4s
10. DNN Early stopped at epoch 137/600. RMSE: 42.06 | R²: 0.9745 | Time: 395.2s
11. LSTM LSTM ran full 50 epochs (val_loss: 0.00541). BiLSTM ran full 50 epochs (val_loss: 0.00308).
12. SHAP Explainability Generated summary plots, waterfall plots (3), dependence plots (3), and force plot. Top features saved.
13. Model Comparison
Model	RMSE	MAE	R²	MAPE	Train Time
ANN	43.63	27.70	0.9725	18.13%	63.4s
DNN	42.06	27.87	0.9745	23.18%	395.2s
LSTM	75.00	46.94	0.9188	28.08%	39.9s
BiLSTM	56.38	34.61	0.9541	19.98%	70.4s

