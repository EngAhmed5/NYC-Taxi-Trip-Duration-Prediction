from preprocessing import * 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import  Ridge
from sklearn.preprocessing import  PolynomialFeatures ,OneHotEncoder, StandardScaler ,LabelEncoder , FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold 
import seaborn as sns
import matplotlib.pyplot as plt

# Function to evaluate model performance on a dataset
def evaluate_model(model, df, train_features, name):
    y_pred = model.predict(df[train_features])

    rmse = np.sqrt(mean_squared_error(df['trip_duration'], y_pred))
    r2 = r2_score(df['trip_duration'], y_pred)

    print("\nEvaluation Results")
    print("-" * 30)
    print(f"Dataset   : {name}")
    print(f"RMSE      : {rmse:.3f}")
    print(f"R² Score  : {r2 * 100:.2f}%")
    print("-" * 30 + "\n")

# Custom log transformation function (handles non-positive values)
def log_transform(x) :
   return np.log1p(np.maximum(x,0))

# Function to construct a preprocessing and modeling pipeline
def make_pipeline(numerical_feature , categorical_feature , use_poly = True):
   logfeature = FunctionTransformer(log_transform)
    # Define preprocessing for numerical features
   if use_poly == True:
      
      numerical_transform = Pipeline(steps=[
         ('scaler' , StandardScaler()),
         ('poly' , PolynomialFeatures(degree=5)),
         ('log' , logfeature)
      ])
   else :
      numerical_transform = Pipeline(steps=[
         ('scaler' , StandardScaler()),
         ('log' , logfeature)
      ])
      
    # Define preprocessing for categorical features
   categorical_transform = Pipeline(steps=[
      ('ohe' , OneHotEncoder(handle_unknown='ignore'))
   ])
    # Combine numerical and categorical transformations
   transformer = ColumnTransformer(
      transformers=[
         ('num' , numerical_transform , numerical_feature),
         ('cat' , categorical_transform , categorical_feature)
      ]
   )
   
   # Create final pipeline with preprocessing and Ridge regression model
   model = Pipeline(steps=[
      ('processing' , transformer) ,
      ('Ridg' , Ridge(alpha=1.0))
   ])
   
   
   return model 



