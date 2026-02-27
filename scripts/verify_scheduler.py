import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from datetime import datetime, timezone
from carbon.carbon_scheduler import CarbonScheduler
CSV_PATH = r"C:\IP\greenmlops\data\raw\carbon\caiso_2024_hourly.csv"

scheduler = CarbonScheduler(CSV_PATH)

print("=== Carbon Stats (full year 2024) ===")
stats = scheduler.carbon_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print()
print("=== Scheduling Scenarios ===")

scenarios = [
    ("ETT drift at 04:00 (dirtiest hour)",       datetime(2024, 6, 15, 4, 0, tzinfo=timezone.utc),  "ett"),
    ("ETT drift at 06:00 (dirty hour)",           datetime(2024, 6, 15, 6, 0, tzinfo=timezone.utc),  "ett"),
    ("CIFAR drift at 05:00 (12h budget)",         datetime(2024, 6, 15, 5, 0, tzinfo=timezone.utc),  "cifar100"),
    ("Fraud drift at 05:00 (critical)",           datetime(2024, 6, 15, 5, 0, tzinfo=timezone.utc),  "fraud"),
    ("AG News drift at 03:00 (tight budget)",     datetime(2024, 6, 15, 3, 0, tzinfo=timezone.utc),  "ag_news"),
]

for label, t0, dataset in scenarios:
    result = scheduler.schedule_for_dataset(t0=t0, dataset_name=dataset)
    print(f"{label}")
    print(f"  t0={result['t0'].strftime('%H:%M')}  ci={result['carbon_intensity_at_t0']} gCO2/kWh")
    print(f"  t*={result['t_star'].strftime('%H:%M')}  ci={result['carbon_intensity_at_t_star']} gCO2/kWh")
    print(f"  saved={result['carbon_saved_pct']}%  wait={result['wait_hours']}h  policy={result['policy']}")
    print()