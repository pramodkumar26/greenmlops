"""
check_api_quick.py

Quick interactive check of Electricity Maps API.
Use this for manual testing and debugging.

Usage:
    python scripts/check_api_quick.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from carbon.electricity_maps_client import ElectricityMapsClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    print("Electricity Maps API - Quick Check")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("ELECTRICITY_MAPS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n❌ API key not configured!")
        print("\nSteps to fix:")
        print("1. Copy .env.example to .env")
        print("2. Get your API key from https://www.electricitymaps.com/")
        print("3. Add it to .env file")
        return
    
    zone = os.getenv("ELECTRICITY_MAPS_ZONE", "US-CAL-CISO")
    print(f"\nAPI Key: {api_key[:8]}***")
    print(f"Zone: {zone}")
    
    try:
        # Initialize client
        print("\n📡 Connecting to Electricity Maps API...")
        client = ElectricityMapsClient()
        
        # Get current intensity
        print("\n🔍 Fetching current carbon intensity...")
        result = client.get_current_intensity()
        
        ci = result['carbon_intensity']
        timestamp = result['timestamp']
        
        print(f"\n✅ Success!")
        print(f"\n📊 Current Carbon Intensity:")
        print(f"   {ci:.1f} gCO2/kWh")
        print(f"   at {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Interpret the value
        print(f"\n💡 Interpretation:")
        if ci < 100:
            print(f"   🟢 Very clean! Excellent time to train models.")
        elif ci < 180:
            print(f"   🟢 Clean. Good time to train (below 180 threshold).")
        elif ci < 300:
            print(f"   🟡 Moderate. Consider waiting for cleaner window.")
        elif ci < 500:
            print(f"   🟠 High. Delay training if possible.")
        else:
            print(f"   🔴 Very high. Definitely wait for cleaner window.")
        
        # Check forecast availability
        print(f"\n🔮 Checking forecast availability...")
        if client.is_forecast_available():
            print(f"   ✅ Forecast endpoint available!")
            print(f"   Your API tier includes 24-hour forecasts.")
            
            forecast = client.get_forecast(hours=6)
            if forecast and len(forecast) > 0:
                print(f"\n📈 Next 6 hours forecast:")
                for i, point in enumerate(forecast[:6]):
                    time_str = point['timestamp'].strftime('%H:%M')
                    ci_val = point['carbon_intensity']
                    print(f"   {time_str}: {ci_val:.1f} gCO2/kWh")
        else:
            print(f"   ⚠️  Forecast endpoint not available")
            print(f"   This is normal for free tier.")
            print(f"   Scheduler will use current-intensity-only mode.")
        
        print(f"\n✅ API integration is working correctly!")
        print(f"\nNext steps:")
        print(f"1. Run full validation: python scripts/test_electricity_maps.py")
        print(f"2. Update CarbonScheduler to use live mode")
        print(f"3. Test end-to-end scheduling with live data")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"- Check your API key is correct")
        print(f"- Verify you have internet connection")
        print(f"- Check Electricity Maps API status")
        print(f"- Try a different zone if CAISO is unavailable")


if __name__ == "__main__":
    main()
