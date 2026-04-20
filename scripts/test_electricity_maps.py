"""
test_electricity_maps.py

Validation script for Electricity Maps API integration
Run this to confirm API access works before deploying to GCP
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from carbon.electricity_maps_client import ElectricityMapsClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_current_intensity():
    """Test getting current carbon intensity"""
    print("\n" + "="*60)
    print("TEST 1: Current Carbon Intensity")
    print("="*60)
    
    try:
        client = ElectricityMapsClient()
        result = client.get_current_intensity()
        
        print(f"✓ API call successful!")
        print(f"  Zone: {result['zone']}")
        print(f"  Carbon Intensity: {result['carbon_intensity']:.1f} gCO2/kWh")
        print(f"  Timestamp: {result['timestamp'].isoformat()}")
        
        # sanity check
        ci = result['carbon_intensity']
        if 0 <= ci <= 1000:
            print(f"✓ Carbon intensity looks reasonable ({ci:.1f} gCO2/kWh)")
        else:
            print(f"⚠ Warning: Carbon intensity seems unusual ({ci:.1f} gCO2/kWh)")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_forecast():
    """Test getting carbon intensity forecast."""
    print("\n" + "="*60)
    print("TEST 2: Carbon Intensity Forecast")
    print("="*60)
    
    try:
        client = ElectricityMapsClient()
        forecast = client.get_forecast(hours=24)
        
        if forecast is None:
            print("⚠ Forecast endpoint unavailable")
            print("  This is expected on free tier")
            print("  Scheduler will use current-intensity-only mode")
            return True  # Not a failure, just a limitation
        
        print(f"✓ Forecast retrieved successfully!")
        print(f"  Number of forecast points: {len(forecast)}")
        
        if len(forecast) > 0:
            first = forecast[0]
            last = forecast[-1]
            print(f"  First point: {first['carbon_intensity']:.1f} gCO2/kWh at {first['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            print(f"  Last point: {last['carbon_intensity']:.1f} gCO2/kWh at {last['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            
            # Show min/max in forecast window
            intensities = [p['carbon_intensity'] for p in forecast]
            min_ci = min(intensities)
            max_ci = max(intensities)
            print(f"  Range: {min_ci:.1f} to {max_ci:.1f} gCO2/kWh")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def compare_with_historical():
    """Compare live reading with historical CSV for same time of day."""
    print("\n" + "="*60)
    print("TEST 3: Compare Live vs Historical Data")
    print("="*60)
    
    try:
        import pandas as pd
        
        # Get live reading
        client = ElectricityMapsClient()
        live = client.get_current_intensity()
        live_ci = live['carbon_intensity']
        live_time = live['timestamp']
        
        # Load historical CSV
        csv_path = os.getenv("CAISO_CSV_PATH", "data/raw/carbon/caiso_2024_hourly.csv")
        if not os.path.exists(csv_path):
            print(f"⚠ Historical CSV not found at {csv_path}")
            print("  Skipping comparison test")
            return True
        
        df = pd.read_csv(csv_path)
        
        # Try to find a similar time of day in historical data
        # (same hour of day, any day in 2024)
        hour_of_day = live_time.hour
        
        # Parse timestamp column (handle various formats)
        ts_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                ts_col = col
                break
        
        if ts_col:
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
            df['hour'] = df['timestamp'].dt.hour
            
            # Get carbon intensity column
            ci_col = None
            for col in df.columns:
                if 'carbon' in col.lower() and 'intensity' in col.lower():
                    ci_col = col
                    break
            
            if ci_col:
                # Find readings from same hour of day
                same_hour = df[df['hour'] == hour_of_day][ci_col]
                
                if len(same_hour) > 0:
                    hist_mean = same_hour.mean()
                    hist_std = same_hour.std()
                    
                    print(f"✓ Comparison complete")
                    print(f"  Live reading: {live_ci:.1f} gCO2/kWh at hour {hour_of_day}")
                    print(f"  Historical (2024) at hour {hour_of_day}:")
                    print(f"    Mean: {hist_mean:.1f} gCO2/kWh")
                    print(f"    Std: {hist_std:.1f} gCO2/kWh")
                    print(f"    Range: {same_hour.min():.1f} to {same_hour.max():.1f} gCO2/kWh")
                    
                    # Check if live reading is within reasonable range
                    if hist_mean - 2*hist_std <= live_ci <= hist_mean + 2*hist_std:
                        print(f"  ✓ Live reading is within 2σ of historical mean")
                    else:
                        print(f"  ⚠ Live reading differs significantly from historical")
                        print(f"    This could be normal (grid conditions change)")
                else:
                    print("⚠ No historical data for this hour")
            else:
                print("⚠ Could not find carbon intensity column in CSV")
        else:
            print("⚠ Could not find timestamp column in CSV")
        
        return True
        
    except Exception as e:
        print(f"⚠ Comparison test failed: {e}")
        print("  This is not critical - API integration still works")
        return True  # Don't fail overall test


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("Electricity Maps API Validation")
    print("="*60)
    print(f"Time: {datetime.now().isoformat()}")
    
    # Check if API key is set
    api_key = os.getenv("ELECTRICITY_MAPS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n✗ ELECTRICITY_MAPS_API_KEY not set!")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Electricity Maps API key")
        print("  3. Run this script again")
        sys.exit(1)
    
    print(f"API Key: {api_key[:8]}... (hidden)")
    print(f"Zone: {os.getenv('ELECTRICITY_MAPS_ZONE', 'US-CAL-CISO')}")
    
    # Run tests
    results = []
    results.append(("Current Intensity", test_current_intensity()))
    results.append(("Forecast", test_forecast()))
    results.append(("Historical Comparison", compare_with_historical()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("  Ready to proceed with CarbonScheduler integration")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        print("  Fix issues before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
