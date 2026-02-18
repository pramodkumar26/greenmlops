import pandas as pd

# Thresholds (gCO2/kWh)
CLEAN = 200
DIRTY = 300

def get_current_carbon(datetime_str=None):
    df = pd.read_csv('data/raw/carbon/caiso_2024_hourly.csv')
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
    df = df.rename(columns={'Carbon intensity gCO₂eq/kWh (direct)': 'carbon_intensity'})
    
    if datetime_str:
        row = df[df['Datetime (UTC)'] == datetime_str].iloc[0]
    else:
        row = df.iloc[-1]  # latest
    
    carbon = row['carbon_intensity']
    
    if carbon < CLEAN:
        status = 'CLEAN'
    elif carbon < DIRTY:
        status = 'MODERATE'
    else:
        status = 'DIRTY'
    
    return {'carbon_intensity': carbon, 'status': status}

def should_train(urgency, datetime_str=None):
    carbon = get_current_carbon(datetime_str)
    
    if urgency == 'HIGH':
        return True  # Always train immediately
    
    if carbon['status'] == 'CLEAN':
        return True
    elif carbon['status'] == 'MODERATE' and urgency == 'MEDIUM':
        return True
    else:
        return False  # Wait for clean window

if __name__ == '__main__':
    # Test
    result = get_current_carbon('2024-06-15 12:00:00')
    print(f"Carbon: {result['carbon_intensity']} gCO2/kWh — {result['status']}")
    
    for urgency in ['HIGH', 'MEDIUM', 'LOW']:
        decision = should_train(urgency, '2024-06-15 12:00:00')
        print(f"{urgency}: {'Train now' if decision else 'Wait'}")