import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker

def train_telco():
    # Load data
    df = pd.read_csv('data/processed/telco_churn_cleaned.csv')
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Track emissions
    tracker = EmissionsTracker(
        project_name="telco_baseline",
        output_dir="experiments/carbon_logs"
    )
    
    tracker.start()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    emissions = tracker.stop()
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f" Telco Baseline Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   AUC-ROC:  {auc:.4f}")
    print(f" Emissions: {emissions*1000:.4f} gCO2")
    
    return model, accuracy, emissions

if __name__ == '__main__':
    train_telco()