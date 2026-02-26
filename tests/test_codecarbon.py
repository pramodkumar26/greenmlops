import torch
import torchvision.models as models
from codecarbon import EmissionsTracker
import time

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracker = EmissionsTracker(
    project_name="codecarbon_gpu_test",
    measure_power_secs=5,
    log_level="WARNING"
)

model = models.resnet18(pretrained=False).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

tracker.start()

for i in range(50):
    inputs = torch.randn(32, 3, 32, 32).to(device)
    labels = torch.randint(0, 100, (32,)).to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Step {i}/50 done")

emissions = tracker.stop()

print(f"\n=== CODECARBON RESULTS ===")
print(f"Emissions : {emissions:.8f} kg CO2")
print(f"Emissions : {emissions * 1000:.6f} g CO2")

if emissions > 0:
    print("CodeCarbon GPU tracking is working correctly")
else:
    print("WARNING: emissions = 0, GPU tracking may not be configured correctly")