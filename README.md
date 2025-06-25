# 🔥 HVAC Forecasting System with Duration & Clustering Optimization (Blue Star R&D)

This project implements a robust machine learning system to forecast personalized HVAC (Air Conditioning) usage patterns including temperature setpoint, duration, fan speed, and mode for each user using synthetic minute-level data.

Developed during my internship at **Blue Star India's R&D Division**, this project also includes scalable **user clustering using K-Means** to allow grouped behavioral modeling for production-scale deployment.

## 💡 Key Features

- 🧠 Multi-Layer Forecasting: STL + ARIMA for temperature prediction using cyclical decomposition.
- ⏱️ Duration Prediction: Gradient boosting, Random Forest & XGBoost with MAE, RMSE, MAPE evaluation.
- 🔌 AC ON/OFF Detection: Simulates realistic ON/OFF patterns with contextual gap-based OFF insertion.
- 🔁 KMP + Autocorrelation: Detects prominent periodic ON/OFF cycles per user (daily, weekly, monthly).
- 🔍 K-Means Clustering: Scalable user segmentation for multi-model personalization (supports K=150+).
- 📈 Minute-Level Simulation: Fully synthetic data generator with real-world seasonality profiles.

## 🧪 Notebooks Included

| Notebook | Description |
|----------|-------------|
| `work.ipynb` | Full end-to-end pipeline with forecasting, feature engineering, cycle detection |
| `clustering_kmeans.ipynb` | User segmentation using KMeans with silhouette evaluation & PCA |

## ⚙️ Tech Stack

Python · Pandas · Scikit-learn · XGBoost · Statsmodels · Plotly

## 🚀 How to Run

```bash
git clone https://github.com/your-username/hvac-forecasting-system.git
cd hvac-forecasting-system
pip install -r requirements.txt
jupyter notebook work.ipynb
```

## ✍️ Author

**Aryan Dixit**  
Intern @ Blue Star India – R&D Technology Division  
Mathematical Sciences Major @ NTU Singapore

## 📄 License

MIT License
