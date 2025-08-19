import pandas as pd
from sklearn.model_selection import train_test_split
import os

def process_raw_data(input_path: str, output_train_path: str, output_test_path: str):

    df = pd.read_csv(input_path)
    
    
    cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_con_ceros:
        median = df[col].replace(0, pd.NA).median()
        df[col] = df[col].replace(0, median)
    
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Outcome'])
    
    
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    
    print(f"Datos procesados guardados en {output_train_path} y {output_test_path}")

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "diabetes.csv")
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    
    process_raw_data(raw_data_path, train_path, test_path)
