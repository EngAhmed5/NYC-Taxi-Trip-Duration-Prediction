import pandas as pd 
import joblib
from preprocessing import * 
from model import * 


def predict (model , test_data) :
    
    df_test = preprocess_data(test_data)
    
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
    
    evaluate_model(model , df_test , all_feature , 'Test')
    
    
  
if __name__ == "__main__":
    model_path = r"D:\Ahmed\Trip Duration Project\Modeling and MLflow\model.pkl"
    model = joblib.load(model_path)
    df_test = pd.read_csv(r"D:\Ahmed\Trip Duration Project\Data\test.csv")
    
    predict(model , df_test)
    
    