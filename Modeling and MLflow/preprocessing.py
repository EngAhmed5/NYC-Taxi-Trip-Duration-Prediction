import numpy as np
import pandas as pd
import holidays
from geopy.distance import geodesic
from datetime import datetime

# ----------------------------- Time-based Feature Functions -----------------------------
def period_of_day(hour):
   if 5 <= hour < 12:
      return 'morning'
   elif 12 <= hour < 17:
      return 'afternoon'
   elif 17 <= hour < 22:
      return 'evening'
   return 'night'

def is_weekend(weekday):
   return weekday in ['Saturday', 'Sunday']

def determine_season(month):
   if month in [12, 1, 2]:
      return 'Winter'
   elif month in [3, 4, 5]:
      return 'Spring'
   elif month in [6, 7, 8]:
      return 'Summer'
   else:
      return 'Autumn'

# ----------------------------- Geospatial Feature Functions -----------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
   """Calculate haversine distance between two lat/lon points in km."""
   R = 6371.0  # Earth radius in kilometers
   lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
   dlat = lat2 - lat1
   dlon = lon2 - lon1

   a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
   return R * c

def geodesic_distance(row):
   """Compute distance using geopy's geodesic (in km)."""
   return geodesic((row['pickup_latitude'], row['pickup_longitude']),
                  (row['dropoff_latitude'], row['dropoff_longitude'])).km

def compute_bearing(lat1, lon1, lat2, lon2):
   """Compute direction (bearing) from point A to B."""
   lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
   dlon = lon2 - lon1
   x = np.sin(dlon) * np.cos(lat2)
   y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
   bearing = np.degrees(np.arctan2(x, y))
   return (bearing + 360) % 360

def manhattan_approximation(row):
   """Approximate Manhattan distance as lat and lon segments separately."""
   lat_dist = haversine_distance(row['pickup_latitude'], row['pickup_longitude'],
                                 row['dropoff_latitude'], row['pickup_longitude'])
   lon_dist = haversine_distance(row['pickup_latitude'], row['pickup_longitude'],
                                 row['pickup_latitude'], row['dropoff_longitude'])
   return lat_dist + lon_dist

# ----------------------------- Preprocessing Main Function -----------------------------
def preprocess_data(df, country='US'):
   df = df.copy()

   # Drop ID column if exists
   df.drop(columns=['id'], errors='ignore', inplace=True)

   # Convert datetime
   df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
   df['hour'] = df['pickup_datetime'].dt.hour
   df['month'] = df['pickup_datetime'].dt.month
   df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
   df['weekday_name'] = df['pickup_datetime'].dt.strftime('%A')
   df['dayofyear'] = df['pickup_datetime'].dt.dayofyear

   # Time-based categories
   df['period_of_day'] = df['hour'].apply(period_of_day)
   df['is_weekend'] = df['weekday_name'].apply(lambda x: is_weekend(x)).astype(int)
   df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
   df['season'] = df['month'].apply(determine_season)

   # Holiday detection
   holiday_dates = holidays.country_holidays(country)
   df['is_holiday'] = df['pickup_datetime'].dt.date.apply(lambda date: date in holiday_dates).astype(int)

   # Geospatial features
   df['Distance_Km'] = haversine_distance(
      df['pickup_latitude'], df['pickup_longitude'],
      df['dropoff_latitude'], df['dropoff_longitude']
   )

# -------------------- Geodesic Distance --------------------
   df['Geodesic_Km'] = df.apply(geodesic_distance, axis=1)

   df['Manhattan_Km'] = df.apply(manhattan_approximation, axis=1)

   df['Bearing'] = compute_bearing(
      df['pickup_latitude'], df['pickup_longitude'],
      df['dropoff_latitude'], df['dropoff_longitude']
   )

   # Area estimation
   lat_km = 110.574
   lon_km = 111.320 * np.cos(np.radians(df['pickup_latitude']))
   df['trip_area_km2'] = (
      (df['pickup_latitude'] - df['dropoff_latitude']).abs() * lat_km *
      (df['pickup_longitude'] - df['dropoff_longitude']).abs() * lon_km
   )

   # Target Transformation (log)
   df['trip_duration'] = np.log1p(df['trip_duration'])

   # Drop unused weekday_name column
   df.drop(columns=['weekday_name'], inplace=True)

   return df


def detect_outliers_iqr(df, columns, threshold=1.5):
   """
   Detect outliers in specified columns using the IQR method.

   Parameters:
   - df: DataFrame
   - columns: list of column names to check for outliers
   - threshold: multiplier for IQR (default=1.5)

   Returns:
   - Dictionary mapping column names to lists of outlier row indices.
   """
   outlier_indices = {}

   for col in columns:
      Q1 = df[col].quantile(0.25)
      Q3 = df[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - threshold * IQR
      upper_bound = Q3 + threshold * IQR

      outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
      outlier_indices[col] = outliers

   return outlier_indices





def remove_outliers(df, outlier_dict):
   """
   Remove rows from the DataFrame that are considered outliers in any column.

   Parameters:
   - df: DataFrame
   - outlier_dict: dictionary of column-wise outlier indices from detect_outliers_iqr

   Returns:
   - Cleaned DataFrame with outlier rows dropped.
   """
   all_outlier_indices = set()
   for indices in outlier_dict.values():
      all_outlier_indices.update(indices)

   return df.drop(index=all_outlier_indices).reset_index(drop=True)


