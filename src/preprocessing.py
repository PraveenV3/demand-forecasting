import pandas as pd
import os

def load_and_combine(data_dir="data/raw"):
    # Combine all CSVs in directory
    all_dfs = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def preprocess_data(df):
    # Convert to datetime
    df['ticket_date'] = pd.to_datetime(df['ticket_date'], dayfirst=True, errors='coerce')
    
    # Handle missing weights
    df['net_weight_kg'] = df.groupby(['area', 'waste_type'])['net_weight_kg'] \
                          .transform(lambda x: x.fillna(x.rolling(7, min_periods=1).mean()))
    
    # Add temporal features
    df = df.assign(
        week=df['ticket_date'].dt.isocalendar().week,
        month=df['ticket_date'].dt.month,
        year=df['ticket_date'].dt.year
    )
    
    return df

def aggregate_weekly(df):
    # Aggregate net_weight_kg weekly by area and waste_type
    weekly_df = df.groupby(['year', 'week', 'area', 'waste_type'], as_index=False) \
                  .agg({'net_weight_kg': 'sum'})
    return weekly_df

def aggregate_seasonal(df):
    # Define seasons by month
    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(month_to_season)
    seasonal_df = df.groupby(['year', 'season', 'area', 'waste_type'], as_index=False) \
                   .agg({'net_weight_kg': 'sum'})
    return seasonal_df

def save_processed_data(df, output_dir="data/processed", filename="processed_data.csv"):
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Step 1: Combine data
    raw_df = load_and_combine()
    
    # Step 2: Clean and transform
    processed_df = preprocess_data(raw_df)
    
    # Step 3: Aggregate weekly and save
    weekly_df = aggregate_weekly(processed_df)
    save_processed_data(weekly_df, filename="weekly_waste.csv")
    
    # Step 4: Aggregate seasonal and save
    seasonal_df = aggregate_seasonal(processed_df)
    save_processed_data(seasonal_df, filename="seasonal_waste.csv")
    
    # Show preview
    print("\nWeekly Aggregated Data Preview:")
    print(weekly_df.head())
    print("\nSeasonal Aggregated Data Preview:")
    print(seasonal_df.head())
