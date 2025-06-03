# NYC Taxi Trip Duration Prediction

This project aims to predict the **trip duration** of NYC yellow taxi rides based on trip and location features using machine learning techniques. The final model uses Ridge Regression combined with polynomial feature engineering, and MLflow is integrated to track experiment runs.

---

## 📂 Project Structure

├── Data/ # Contains train.csv, val.csv, test.csv datasets
├── EDA/ # Exploratory Data Analysis notebook and visualizations
├── mlruns/ # MLflow tracking artifacts for experiment management
├── modeling and mlflow/ # Model scripts and pipeline
│ ├── preprocessing.py # Data preprocessing pipeline
│ ├── train.py # Model training script
│ ├── test.py # Model testing script
│ ├── trip_duration_mlflow.py # MLflow experiment tracking script
│ ├── model.py # Model definition
│ └── model.pkl # Trained model file
├── NYC Taxi Trip Duration Prediction Report/ # Final project report (PDF)
├── Score's Picture/ # Images showing model performance (train/val/test scores)
├── predictions_sample.csv # Sample submission/predictions CSV file
├── README.md # This file

---

## 🧰 Tools & Libraries Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn (Ridge Regression, Polynomial Features, Pipeline, etc.)
- Matplotlib & Seaborn (Visualization)
- MLflow (Experiment tracking and model registry)
- Jupyter Notebook (EDA)
- VS Code (Development environment)

---

## 📊 Exploratory Data Analysis (EDA)

- Analyzed trip duration patterns by **weekday**, **month**, and **hour**.
- Visualized heatmaps showing trip frequency by time and day.
- Investigated correlations between features like distance, speed, pickup/dropoff locations.
- Identified outliers but **did not remove** them intentionally to avoid overfitting when using polynomial features.
- Key insight: Longer trips occur on weekends and warmer months; peak activity during evening rush hours.

---

## 🔧 Data Preprocessing & Feature Engineering

- Log transformation applied to trip duration and distance to normalize skewed distributions.
- Polynomial features up to degree 5 used on numerical data to capture non-linear relationships.
- One-hot encoding applied to categorical features.
- Standard scaling applied to numerical features before modeling.

---

## ⚙️ Modeling

- **Model selected:** Ridge Regression with regularization parameter alpha=1.0.
- **Pipeline includes:** preprocessing (scaling, polynomial feature expansion, encoding) followed by Ridge Regression.
- **MLflow** used for tracking and managing experiment runs.

---

## 📈 Model Performance

| Dataset    | RMSE   | R² Score   |
|------------|--------|------------|
| Training   | 0.433  | 70.34%     |
| Validation | 0.443  | 69.32%     |
| Test       | 0.433  | 70.45%     |

- RMSE: Root Mean Squared Error (lower is better)
- R² Score: Percentage of variance explained by the model (higher is better)
- The model generalizes well on validation and test sets, indicating balanced bias-variance.

---

## 🔍 Important Notes

- **Outliers Handling:** Outliers were **not removed** because polynomial features with degree 5 can cause overfitting on clean data. Keeping some noise and outliers helped maintain a balanced model without overfitting.
- **Feature Selection:** Distance (in km) is the strongest predictor of trip duration. Speed and location features also contribute but with less impact.
- **MLflow Integration:** All training runs, parameters, and metrics are logged with MLflow for easy experiment comparison.

---
