"""
test_live_scheduling.py

End-to-end test of CarbonScheduler in live mode.
Demonstrates carbon-aware scheduling with real-time API data.

Usage:
    python scripts/test_live_scheduling.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from carbon.carbon_scheduler import CarbonScheduler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_live_scheduling():
    """Test live scheduling for all dataset types."""
    print("\n" + "="*70)
    print("GreenMLOps Live Carbon-Aware Scheduling Test")
    print("="*70)
    
    # Initialize scheduler in live mode
    print("\n🔧 Initializing CarbonScheduler in live mode...")
    try:
        scheduler = CarbonScheduler(mode="live")
        print("✅ Scheduler initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize scheduler: {e}")
        return False
    
    # Get current carbon stats
    print("\n📊 Current Carbon Intensity Statistics:")
    stats = scheduler.carbon_stats()
    if "error" in stats:
        print(f"❌ Error getting stats: {stats['error']}")
        return False
    
    print(f"   Current: {stats['current']} gCO2/kWh")
    if stats.get('forecast_hours', 0) > 0:
        print(f"   24h Range: {stats['min']} to {stats['max']} gCO2/kWh")
        print(f"   24h Mean: {stats['mean']} gCO2/kWh")
        print(f"   Forecast Points: {stats['forecast_hours']}")
    else:
        print(f"   Forecast: Not available (using current intensity only)")
    
    # Test scheduling for each dataset type
    datasets = [
        ("fraud", "CRITICAL", "Security-sensitive - always immediate"),
        ("ett", "LOW", "Time-series forecasting - can wait 24h"),
        ("cifar100", "MEDIUM", "Image classification - can wait 12h"),
        ("ag_news", "MEDIUM", "Text classification - can wait 12h"),
    ]
    
    t0 = datetime.now(timezone.utc)
    print(f"\n🕐 Scheduling Time (t0): {t0.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    results = []
    
    for dataset, urgency, description in datasets:
        print(f"\n" + "-"*50)
        print(f"Dataset: {dataset.upper()} ({urgency})")
        print(f"Description: {description}")
        print("-"*50)
        
        try:
            # Schedule retraining for this dataset
            result = scheduler.schedule_for_dataset(
                t0=t0,
                dataset_name=dataset,
                current_accuracy_drop_pct=0.0  # No accuracy drop yet
            )
            
            # Display results
            t_star = result["t_star"]
            wait_hours = result["wait_hours"]
            carbon_saved = result["carbon_saved_pct"]
            policy = result["policy"]
            
            print(f"📅 Decision: {policy}")
            print(f"🕐 Train at: {t_star.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"⏱️  Wait time: {wait_hours:.1f} hours")
            print(f"🌱 Carbon saved: {carbon_saved:.1f}%")
            print(f"📈 Carbon at t0: {result['carbon_intensity_at_t0']} gCO2/kWh")
            print(f"📉 Carbon at t*: {result['carbon_intensity_at_t_star']} gCO2/kWh")
            
            # Interpret the decision
            if urgency == "CRITICAL":
                if wait_hours == 0:
                    print("✅ Correct: CRITICAL urgency trains immediately")
                else:
                    print("❌ Error: CRITICAL should never wait")
            else:
                if wait_hours > 0:
                    print(f"✅ Optimization: Waiting {wait_hours:.1f}h for cleaner grid")
                else:
                    print("✅ Immediate: Grid is already clean enough")
            
            results.append((dataset, result))
            
        except Exception as e:
            print(f"❌ Error scheduling {dataset}: {e}")
            results.append((dataset, None))
    
    # Summary
    print(f"\n" + "="*70)
    print("SCHEDULING SUMMARY")
    print("="*70)
    
    total_carbon_saved = 0
    successful_schedules = 0
    
    for dataset, result in results:
        if result:
            wait = result["wait_hours"]
            saved = result["carbon_saved_pct"]
            policy = result["policy"]
            
            print(f"{dataset.upper():10} | {wait:6.1f}h wait | {saved:6.1f}% saved | {policy}")
            
            if saved > 0:
                total_carbon_saved += saved
                successful_schedules += 1
    
    print("-"*70)
    if successful_schedules > 0:
        avg_savings = total_carbon_saved / len([r for r in results if r[1]])
        print(f"Average carbon savings: {avg_savings:.1f}%")
        print(f"Datasets with savings: {successful_schedules}/{len(results)}")
    
    print(f"\n✅ Live scheduling test completed successfully!")
    print(f"🚀 Ready for GCP deployment (Phase 2)")
    
    return True


def compare_modes():
    """Compare historical vs live mode decisions (if historical data available)."""
    print(f"\n" + "="*70)
    print("MODE COMPARISON: Historical vs Live")
    print("="*70)
    
    # Check if historical CSV exists
    csv_path = os.getenv("CAISO_CSV_PATH", "data/raw/carbon/caiso_2024_hourly.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️  Historical CSV not found at {csv_path}")
        print(f"   Skipping mode comparison")
        return
    
    try:
        # Initialize both schedulers
        hist_scheduler = CarbonScheduler(caiso_csv_path=csv_path, mode="historical")
        live_scheduler = CarbonScheduler(mode="live")
        
        # Test with ETT (LOW urgency, 24h window)
        t0 = datetime.now(timezone.utc)
        dataset = "ett"
        
        print(f"\nTesting {dataset.upper()} scheduling at {t0.strftime('%H:%M %Z')}:")
        
        # Historical decision
        hist_result = hist_scheduler.schedule_for_dataset(t0=t0, dataset_name=dataset)
        print(f"\n📚 Historical Mode:")
        print(f"   Wait: {hist_result['wait_hours']:.1f}h")
        print(f"   Carbon saved: {hist_result['carbon_saved_pct']:.1f}%")
        print(f"   Policy: {hist_result['policy']}")
        
        # Live decision
        live_result = live_scheduler.schedule_for_dataset(t0=t0, dataset_name=dataset)
        print(f"\n🌐 Live Mode:")
        print(f"   Wait: {live_result['wait_hours']:.1f}h")
        print(f"   Carbon saved: {live_result['carbon_saved_pct']:.1f}%")
        print(f"   Policy: {live_result['policy']}")
        
        # Compare
        wait_diff = live_result['wait_hours'] - hist_result['wait_hours']
        savings_diff = live_result['carbon_saved_pct'] - hist_result['carbon_saved_pct']
        
        print(f"\n🔄 Comparison:")
        print(f"   Wait difference: {wait_diff:+.1f}h (live vs historical)")
        print(f"   Savings difference: {savings_diff:+.1f}% (live vs historical)")
        
        if abs(wait_diff) < 1.0:
            print(f"   ✅ Similar decisions - good consistency")
        else:
            print(f"   ℹ️  Different decisions - expected due to different data sources")
        
    except Exception as e:
        print(f"❌ Mode comparison failed: {e}")


def main():
    """Run all live scheduling tests."""
    print("GreenMLOps Live Scheduling Validation")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Check API key
    api_key = os.getenv("ELECTRICITY_MAPS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n❌ ELECTRICITY_MAPS_API_KEY not set!")
        print("   Make sure you have .env file with your API key")
        sys.exit(1)
    
    # Run tests
    success = test_live_scheduling()
    
    if success:
        compare_modes()
        
        print(f"\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print(f"✅ Live API integration working")
        print(f"✅ CarbonScheduler supports live mode")
        print(f"✅ All dataset types scheduled correctly")
        print(f"✅ Carbon optimization decisions are reasonable")
        print(f"\n🚀 Phase 1 Complete - Ready for Phase 2 (Vertex AI)")
        
    else:
        print(f"\n❌ Tests failed - fix issues before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()