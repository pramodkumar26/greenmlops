from codecarbon import EmissionsTracker

def train_with_tracking(model_name, train_func):
    tracker = EmissionsTracker(
        project_name=f"greenmlops_{model_name}",
        output_dir="experiments/carbon_logs",
        
        
    )
    
    tracker.start()
    result = train_func()
    emissions = tracker.stop()
    
    print(f"{model_name} training complete")
    print(f" Emissions: {emissions*1000:.4f} gCO2")
    
    return result, emissions

# Test with dummy training
if __name__ == '__main__':
    import time
    
    def dummy_train():
        time.sleep(2)  # Simulate training
        return "model_trained"
    
    train_with_tracking("telco_test", dummy_train)