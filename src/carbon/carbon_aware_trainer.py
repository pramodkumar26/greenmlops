from codecarbon import EmissionsTracker
from scheduler import get_current_carbon, should_train
import time

def carbon_aware_train(model_name, urgency, train_func, datetime_str=None):
    # Step 1: Check carbon
    carbon = get_current_carbon(datetime_str)
    decision = should_train(urgency, datetime_str)
    
    print(f"\n Carbon: {carbon['carbon_intensity']:.2f} gCO2/kWh — {carbon['status']}")
    print(f" Urgency: {urgency}")
    
    if not decision:
        print(f"⏳ Waiting for clean energy window...")
        return None, None
    
    print(f" Training now...")
    
    # Step 2: Track emissions during training
    tracker = EmissionsTracker(
        project_name=f"greenmlops_{model_name}",
        output_dir="experiments/carbon_logs"
    )
    
    tracker.start()
    result = train_func()
    emissions = tracker.stop()
    
    print(f"🌿 Emissions: {emissions*1000:.4f} gCO2")
    print(f"📊 Carbon saved by scheduling: {carbon['carbon_intensity']:.2f} gCO2/kWh avoided if waited")
    
    return result, emissions

if __name__ == '__main__':
    def dummy_train():
        time.sleep(2)
        return "model_trained"
    
    # Test dirty window - mixed results
    print(" DIRTY WINDOW TEST ")
    carbon_aware_train("telco", "MEDIUM", dummy_train, "2024-01-01 00:00:00")
    
    print("\n CLEAN WINDOW TEST ")
    carbon_aware_train("telco", "MEDIUM", dummy_train, "2024-06-15 12:00:00")