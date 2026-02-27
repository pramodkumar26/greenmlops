import pandas as pd
df = pd.read_csv(r'C:\IP\greenmlops\data\raw\carbon\caiso_2024_hourly.csv')
df.columns = [c.strip() for c in df.columns]
ci_col = 'Carbon intensity gCO\u2082eq/kWh (direct)'
df['hour'] = pd.to_datetime(df['Datetime (UTC)']).dt.hour
print(df.groupby('hour')[ci_col].mean().round(1).to_string())
