import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from codecarbon import EmissionsTracker

def train_airquality():
    df = pd.read_csv('data/processed/airquality_clean.csv')
    
    df = df.drop(columns=['datetime'], errors='ignore')
    df = df.dropna()
    
    X = df.drop('PM2.5', axis=1)
    y = df['PM2.5']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    tracker = EmissionsTracker(
        project_name="airquality_baseline",
        output_dir="experiments/carbon_logs"
    )
    
    tracker.start()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    emissions = tracker.stop()
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Air Quality Baseline Results:")
    print(f"   MAE: {mae:.4f}")
    print(f"   R2 Score: {r2:.4f}")
    print(f"Emissions: {emissions*1000:.4f} gCO2")
    
    return model, mae, emissions

if __name__ == '__main__':
    train_airquality()