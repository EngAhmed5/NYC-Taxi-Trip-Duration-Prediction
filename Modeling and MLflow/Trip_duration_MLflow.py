import mlflow
import joblib
from train import * 
from preprocessing import * 


model_path = r"D:\Ahmed\Trip Duration Project\Modeling and MLflow\model.pkl"
model = joblib.load(model_path)
df_test = pd.read_csv(r"D:\Ahmed\Trip Duration Project\Data\test.csv")

mlflow.set_experiment("Trip Duration Prediction")   
with mlflow.start_run() as run :
    df_test = preprocess_data(df_test) 
    
    numeric_feature = ['pickup_longitude','dropoff_longitude','pickup_latitude',
                     'dropoff_latitude','Distance_Km', 'Manhattan_Km', 
                     'Bearing','Geodesic_Km' ,'trip_area_km2'
                     ]
   
    categroical_feature = ['period_of_day','season','is_weekend',
                        'hour','month','is_peak_hour',
                        'is_holiday','dayofweek','passenger_count',
                        'vendor_id' ,'store_and_fwd_flag' ]
   
    all_feature = numeric_feature + categroical_feature
    
    target_column = 'trip_duration'
    
    y_pred = model.predict(df_test[all_feature])
    
    rmse = np.sqrt(mean_squared_error(df_test['trip_duration'] , y_pred))
    r2 = r2_score(df_test['trip_duration'] , y_pred)
    
    mlflow.log_metric("Test RMSE" , rmse)
    mlflow.log_metric("Test R2 Score" , r2) 
    mlflow.sklearn.log_model(model,"Ridge" )
    
    mlflow.log_param("Model Type", "Ridge Regression")
    mlflow.log_param("Polynomial Degree", 5)
    mlflow.log_param("Use Log Transform", True)
    mlflow.log_param("Ridge Alpha", model.named_steps['Ridg'].alpha)

    results = pd.DataFrame({
        "Actual": df_test[target_column],
        "Predicted": y_pred
    })
    results.head(10).to_csv("predictions_sample.csv", index=False)
    mlflow.log_artifact("predictions_sample.csv")