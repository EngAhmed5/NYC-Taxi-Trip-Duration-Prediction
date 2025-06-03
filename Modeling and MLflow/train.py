import pandas as pd 
from model import * 
from preprocessing import * 
import joblib 
import os 

if __name__ == "__main__":
   
   train_path = r"D:\Ahmed\Trip Duration Project\Data\train.csv"
   val_path = r"D:\Ahmed\Trip Duration Project\Data\val.csv"
   
   df_train = pd.read_csv(train_path)
   df_val = pd.read_csv(val_path)
   
   df_train = preprocess_data(df_train)
   df_val = preprocess_data(df_val)
   '''
   numeric_feature = ['Manhattan_Km','trip_duration','Distance_Km','trip_area_km2','Geodesic_Km']
   outliers = detect_outliers_iqr(df_train, numeric_feature)
   df_train = remove_outliers(df_train, outliers)
   
   outliers = detect_outliers_iqr(df_val, numeric_feature)
   df_train = remove_outliers(df_val, outliers)
   '''
   
   #numeric_feature = ['pickup_longitude','pickup_latitude', 'dropoff_longitude',
   #                  'dropoff_latitude','Distance_Km', 'Manhattan_Km', 'Bearing','Geodesic_Km'
   #                  ,'hour','month','is_peak_hour','is_holiday']
   
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
   
   model = make_pipeline(numeric_feature , categroical_feature,use_poly=True)
   model.fit(df_train[all_feature] , df_train[target_column])
   
   evaluate_model(model , df_train , all_feature , 'Train')
   evaluate_model(model , df_val , all_feature , 'Validation')
   print("\n")
   print ('Model Evalution Complete')
   
   MODEL_PATH = r"D:\Ahmed\Trip Duration Project\Modeling and MLflow\model.pkl"
   os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
   joblib.dump(model, MODEL_PATH)
   
   

