# Quick Start: Electricity Maps API Integration

**Goal:** Get the Electricity Maps API working in 15 minutes.

## Prerequisites

- Python 3.9+ installed
- GreenMLOps repository cloned
- Internet connection

## Step-by-Step (15 minutes)

### 1. Get API Key (5 minutes)

1. Open https://www.electricitymaps.com/ in your browser
2. Click "Sign Up" or "Get API Key"
3. Create account (email + password)
4. Navigate to API section or account settings
5. Copy your API key (looks like: `abc123def456...`)

**Note:** Free tier is sufficient for testing!

### 2. Configure Environment (2 minutes)

```bash
cd greenmlops

# Copy the example file
cp .env.example .env

# Open .env in your editor
# Replace "your_api_key_here" with your actual key
```

Your `.env` should look like:
```bash
ELECTRICITY_MAPS_API_KEY=abc123def456ghi789...
ELECTRICITY_MAPS_ZONE=US-CAL-CISO
```

Save and close.

### 3. Install Dependencies (3 minutes)

```bash
# If you haven't installed requirements yet
pip install -r requirements.txt

# Key packages needed:
# - requests (HTTP client)
# - python-dotenv (load .env files)
# - pandas (data processing)
```

### 4. Test the API (5 minutes)

```bash
# Quick check (shows current carbon intensity)
python scripts/check_api_quick.py

# Full validation (runs 3 tests)
python scripts/test_electricity_maps.py
```

**Expected output:**
```
✅ Success!

📊 Current Carbon Intensity:
   245.3 gCO2/kWh
   at 2026-04-16 18:00:00 UTC

💡 Interpretation:
   🟡 Moderate. Consider waiting for cleaner window.

✅ API integration is working correctly!
```

## Troubleshooting

### ❌ "API key not found"
- Did you create `.env` file? (not just `.env.example`)
- Did you add your actual API key?
- No quotes needed around the key

### ❌ "401 Unauthorized"
- Your API key is wrong or expired
- Copy it again from Electricity Maps dashboard
- Make sure you copied the entire key

### ❌ "Module not found"
- Run `pip install -r requirements.txt`
- Make sure you're in the `greenmlops` directory

### ⚠️ "Forecast unavailable"
- This is normal on free tier
- Not a problem - scheduler works without forecast
- Upgrade to paid tier if you need forecasts

## What You Just Built

✅ API client that fetches live carbon intensity  
✅ Environment configuration for credentials  
✅ Validation scripts to test everything works  
✅ Foundation for live carbon-aware scheduling  

## Next Steps

Now that the API works, you can:

1. **Update CarbonScheduler** to use live mode (Day 3)
2. **Test end-to-end** with real drift detection (Day 4)
3. **Deploy to GCP** with Vertex AI (Phase 2)

See `PHASE1_CHECKLIST.md` for detailed day-by-day tasks.

## Files You Created

```
greenmlops/
└── .env                    ← Your API credentials (not committed to git)
```

## Files We Provided

```
greenmlops/
├── src/carbon/
│   └── electricity_maps_client.py
├── scripts/
│   ├── test_electricity_maps.py
│   └── check_api_quick.py
├── .env.example
└── ELECTRICITY_MAPS_SETUP.md
```

## Success! 🎉

If `check_api_quick.py` shows a carbon intensity value, you're done with the quick start!

The API integration is working and you're ready to proceed with Phase 1.

---

**Time spent:** ~15 minutes  
**Cost:** $0  
**Status:** ✅ API Integration Working  

**Next:** Read `PHASE1_CHECKLIST.md` for Days 3-5 tasks.
