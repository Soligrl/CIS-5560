# CIS-5560
Introduction to Big Data Science
# 🏠 Redfin Housing Market Forecasting with Apache Spark

This project applies scalable machine learning techniques to forecast U.S. housing market trends using over 8.5GB of Redfin data. By leveraging Apache Spark on a distributed cluster, we built regression models to predict median sale prices and uncover patterns in buyer behavior. The analysis focuses on multiple spatial levels (state, county, zip, and neighborhood) and spans 10+ years of historical data.

---

📌 What the Project Does

- Processes and models 8.5GB+ of Redfin housing data using Apache Spark
- Predicts median sale prices using regression models (RF, GBT, Linear)
- Tunes models using TrainValidationSplit and CrossValidator
- Evaluates performance with RMSE, R², training time, and best params
- Analyzes housing trends at state, county, zip code, and neighborhood levels

---

💡 Why the Project Is Useful

- Highlights how model complexity, tuning, and data granularity affect accuracy
- Helps investors and policymakers understand market behavior by region
- Provides a cloud-scalable ML framework for housing price prediction
- Offers reproducible Spark workflows using `spark-submit` and HDFS storage

---

🚀 How to Get Started

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data
2. Clean & sample large files using Jupyter/Kaggle Notebooks
3. Upload cleaned CSVs to HDFS using Git Bash + SSH
4. Submit PySpark models via `spark-submit` in Databricks Community Edition
5. Compare results using RMSE and R² in output logs or exports

---

🔧 Tools Used

- **Apache Spark MLlib** for regression modeling and pipeline building  
- **Databricks Community Edition** for distributed compute  
- **HDFS** for scalable file storage  
- **Kaggle** for data prep and early sampling  
- **PySpark** for feature engineering, modeling, and tuning

---

📊 Key Results

- **County-level Random Forest** (log target): RMSE = 0.2838, R² = 0.542  
- **Zip-code RF** (raw target): CV gave best results, RMSE = 717.02  
- **State-level Linear Regression**: Minimal gain from tuning, R² = 0.773  
- **Neighborhood GBT**: Extended TrainValidationSplit achieved RMSE = 281,601.80, R² = 0.718

---

🧠 Key Insights

- Log transformations improve model stability with skewed price data  
- CrossValidator boosts model accuracy but requires high compute time  
- Simpler models like Linear Regression work well on clean, structured data  
- Gradient Boosted Trees and Random Forests benefit most from hyperparameter tuning

---

🆘 Where to Get Help

- Open a GitHub Issue with your question  
- Refer to [Spark MLlib docs](https://spark.apache.org/docs/latest/ml-guide.html)  
- Review Databricks Community Edition limitations and tips

---

👩‍💻 Contributors

- **Solange Ruiz** (sruiz85) – Data cleaning, modeling, evaluation, methodology, Spark jobs, documentation  
- **Christian Ahamba** – Cluster setup, data celaning, modeling, evaluation, methodology, spark jobs, result analysis, documentation 
- **Damian Wilson** – 3 related work, documentation 

Project completed as part of **CIS 5560 - Introduction to Big Data Science**  
Department of Information Systems, California State University, Los Angeles  
Supervised by **Prof. Jongwook Woo**

---

📁 Related Files

- `spark_jobs/` – PySpark scripts for Random Forest, GBT, and Linear Regression  
- `cleaned_data/` – CSVs used for modeling (15% samples for CE memory limits)  
- `results/` – Output logs and best model parameters  
- `visuals/` – (Optional) charts, graphs, or screenshots from model outputs  

