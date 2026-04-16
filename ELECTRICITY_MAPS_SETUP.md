# Electricity Maps API Setup Guide

This guide walks you through setting up the Electricity Maps API integration for GreenMLOps live deployment.

## Step 1: Get Your API Key

1. Go to [Electricity Maps](https://www.electricitymaps.com/)
2. Sign up for an account (free tier available)
3. Navigate to your account settings or API section
4. Copy your API key

**Free Tier vs Paid:**
- **Free tier**: Provides current carbon intensity only
- **Paid tier**: Includes 24-hour forecast endpoint

The scheduler works with both tiers. Forecast enables better optimization but is not required.

## Step 2: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```bash
   ELECTRICITY_MAPS_API_KEY=your_actual_api_key_here
   ELECTRICITY_MAPS_ZONE=US-CAL-CISO
   ```

3. Make sure `.env` is in your `.gitignore` (it should be already)

## Step 3: Install Dependencies

If you haven't already installed the project dependencies:

```bash
pip install -r requirements.txt
```

The key dependencies for API integration are:
- `requests` - HTTP client for API calls
- `python-dotenv` - Load environment variables from .env file

## Step 4: Test the API Integration

Run the validation script to confirm everything works:

```bash
cd greenmlops
python scripts/test_electricity_maps.py
```

**Expected output:**
```
============================================================
Electricity Maps API Validation
============================================================
Time: 2026-04-16T...
API Key: abc12345... (hidden)
Zone: US-CAL-CISO

============================================================
TEST 1: Current Carbon Intensity
============================================================
✓ API call successful!
  Zone: US-CAL-CISO
  Carbon Intensity: 245.3 gCO2/kWh
  Timestamp: 2026-04-16T18:00:00+00:00
✓ Carbon intensity value is reasonable (245.3 gCO2/kWh)

============================================================
TEST 2: Carbon Intensity Forecast
============================================================
⚠ Forecast endpoint unavailable
  This is expected on free tier
  Scheduler will use current-intensity-only mode

============================================================
TEST 3: Compare Live vs Historical Data
============================================================
✓ Comparison complete
  Live reading: 245.3 gCO2/kWh at hour 18
  Historical (2024) at hour 18:
    Mean: 238.5 gCO2/kWh
    Std: 45.2 gCO2/kWh
    Range: 150.0 to 350.0 gCO2/kWh
  ✓ Live reading is within 2σ of historical mean

============================================================
Test Summary
============================================================
✓ PASS: Current Intensity
✓ PASS: Forecast
✓ PASS: Historical Comparison

✓ All tests passed!
  Ready to proceed with CarbonScheduler integration
```

## Step 5: Understanding the Results

### Test 1: Current Intensity
- Verifies you can fetch live carbon intensity data
- Should return a value between 0-1000 gCO2/kWh (typical range: 100-500)

### Test 2: Forecast
- Checks if forecast endpoint is available
- **Free tier**: Will show "unavailable" - this is normal
- **Paid tier**: Should return 24 forecast points

### Test 3: Historical Comparison
- Compares live reading with historical CAISO 2024 data
- Helps validate that live readings are reasonable
- Large differences are OK (grid conditions change year to year)

## Troubleshooting

### Error: "API key not found"
- Make sure you created `.env` file (not just `.env.example`)
- Check that `ELECTRICITY_MAPS_API_KEY` is set in `.env`
- No quotes needed around the key value

### Error: "401 Unauthorized"
- Your API key is invalid or expired
- Get a new key from Electricity Maps dashboard
- Make sure you copied the entire key (no spaces)

### Error: "Zone not found"
- The zone identifier might be wrong
- For California ISO, use: `US-CAL-CISO`
- See [Electricity Maps zones](https://api.electricitymap.org/v3/zones) for other regions

### Warning: "Forecast endpoint unavailable"
- This is expected on free tier
- Scheduler will work in current-intensity-only mode
- To get forecast access, upgrade to paid tier

### Error: "Historical CSV not found"
- Test 3 will skip if CSV is missing
- This doesn't affect API integration
- CSV is only used for comparison validation

## Next Steps

Once all tests pass:

1. ✅ API integration is working
2. ⏭️ Proceed to update `CarbonScheduler` to support live mode
3. ⏭️ Test scheduler with live API data
4. ⏭️ Deploy to GCP (Phase 2)

## API Rate Limits

Be aware of rate limits:
- **Free tier**: Typically 100-1000 requests/day
- **Paid tier**: Higher limits depending on plan

For development:
- Cache API responses when testing
- Don't call the API in tight loops
- The scheduler calls once per drift check (daily in production)

## Cost Estimate

- **Free tier**: $0/month - sufficient for testing and low-volume production
- **Paid tier**: ~$50-200/month - needed for forecast access and higher rate limits

For the GreenMLOps deployment (4 datasets, daily checks), free tier should be sufficient.
