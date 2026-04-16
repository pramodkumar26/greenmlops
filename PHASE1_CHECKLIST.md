# Phase 1: Electricity Maps API Integration - Checklist

**Duration:** Days 1-5 (April 17-24, 2026)  
**Cost:** $0  
**Status:** 🟡 In Progress

---

## Day 1: API Key Setup ✅

- [ ] Sign up for Electricity Maps account
- [ ] Get API key from dashboard
- [ ] Check API tier (free vs paid)
- [ ] Verify forecast endpoint availability
- [ ] Document tier limitations

**Deliverable:** API key ready to use

---

## Day 2: Local Testing ⏳

- [x] Create `src/carbon/electricity_maps_client.py`
- [x] Create `.env.example` template
- [x] Create `scripts/test_electricity_maps.py`
- [ ] Copy `.env.example` to `.env`
- [ ] Add your API key to `.env`
- [ ] Run test script: `python scripts/test_electricity_maps.py`
- [ ] Verify all 3 tests pass

**Deliverable:** Working API client with passing tests

**Commands:**
```bash
cd greenmlops
cp .env.example .env
# Edit .env and add your API key
python scripts/test_electricity_maps.py
```

---

## Day 3: CarbonScheduler Integration ⏳

- [ ] Update `src/carbon/carbon_scheduler.py` to support live mode
- [ ] Add `mode` parameter: `historical` or `live`
- [ ] In live mode, use `ElectricityMapsClient` instead of CSV
- [ ] Keep all urgency and MaxDelay logic unchanged
- [ ] Add unit tests for live mode
- [ ] Test both modes work correctly

**Files to modify:**
- `src/carbon/carbon_scheduler.py`
- `tests/test_carbon_scheduler.py` (add live mode tests)

**Key changes:**
```python
class CarbonScheduler:
    def __init__(self, caiso_csv_path: str = None, mode: str = "historical"):
        self.mode = mode
        if mode == "historical":
            self.caiso = load_caiso_data(caiso_csv_path)
        elif mode == "live":
            self.api_client = ElectricityMapsClient()
        else:
            raise ValueError(f"Unknown mode: {mode}")
```

**Deliverable:** CarbonScheduler works in both historical and live modes

---

## Day 4: End-to-End Validation ⏳

- [ ] Create test script: `scripts/test_live_scheduling.py`
- [ ] Simulate a drift event with live API data
- [ ] Run scheduler in live mode
- [ ] Verify it picks a clean window correctly
- [ ] Compare decision with historical mode
- [ ] Log results to MLflow (optional)

**Test scenario:**
1. Get current carbon intensity from API
2. Simulate drift detected for ETT dataset (LOW urgency, D_max=24h)
3. Scheduler should find lowest carbon window in next 24 hours
4. If current intensity is already clean (<180 gCO2/kWh), train immediately
5. Otherwise, schedule for cleaner window

**Deliverable:** Successful end-to-end test with live API

---

## Day 5: Documentation and Commit ⏳

- [ ] Update main README with live mode instructions
- [ ] Document API tier limitations
- [ ] Add troubleshooting guide
- [ ] Commit all changes to git
- [ ] Tag release: `v0.2.0-live-api`
- [ ] Write brief validation notes

**Files to update:**
- `README.md` - add live mode section
- `ELECTRICITY_MAPS_SETUP.md` - already created
- `DEPLOYMENT-PLAN.md` - mark Phase 1 complete

**Git commands:**
```bash
git add .
git commit -m "feat: Add Electricity Maps API integration for live carbon scheduling"
git tag v0.2.0-live-api
git push origin main --tags
```

**Deliverable:** Clean commit with working live mode

---

## Success Criteria

Phase 1 is complete when:

✅ Electricity Maps API client works and passes all tests  
✅ CarbonScheduler supports both `historical` and `live` modes  
✅ Live mode makes real API calls and returns valid scheduling decisions  
✅ End-to-end test shows scheduler picks clean windows correctly  
✅ All code committed and documented  
✅ Ready to proceed to Phase 2 (Vertex AI)

---

## Risk Mitigation

**Risk:** API key doesn't work  
**Mitigation:** Test immediately on Day 1, get new key if needed

**Risk:** Forecast endpoint unavailable (free tier)  
**Mitigation:** Scheduler works without forecast - uses current intensity only

**Risk:** API rate limits hit during testing  
**Mitigation:** Cache responses, add delays between test runs

**Risk:** CAISO zone not available  
**Mitigation:** Check zone list first, use alternative zone if needed

---

## Next Phase Preview

Once Phase 1 is complete, Phase 2 begins:
- Upload baseline model checkpoints to GCS
- Create Vertex AI training job scripts
- Test one job manually (ETT - cheapest)
- Estimated cost: $15-25
- Duration: 5-7 days

**Do not start Phase 2 until Phase 1 success criteria are met.**
