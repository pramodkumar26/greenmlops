"""
Client for Electricity Maps API - gets live carbon intensity data
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
import requests

logger = logging.getLogger(__name__)


class ElectricityMapsClient:
    """Client for Electricity Maps API"""
    
    BASE_URL = "https://api.electricitymap.org/v3"
    
    def __init__(self, api_key: Optional[str] = None, zone: str = "US-CAL-CISO"):
        self.api_key = api_key or os.getenv("ELECTRICITY_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found - set ELECTRICITY_MAPS_API_KEY env var")
        
        self.zone = zone
        self.headers = {"auth-token": self.api_key}
        self._forecast_available = None
        
        logger.info(f"Initialized ElectricityMapsClient for zone {self.zone}")
    
    def get_current_intensity(self) -> Dict[str, any]:
        """Get current carbon intensity for the zone"""
        endpoint = f"{self.BASE_URL}/carbon-intensity/latest"
        params = {"zone": self.zone}
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            carbon_intensity = data.get("carbonIntensity")
            timestamp_str = data.get("datetime")
            
            if carbon_intensity is None:
                raise ValueError(f"carbonIntensity not found in response: {data}")
            
            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str
                else datetime.now(timezone.utc)
            )
            
            logger.info(
                f"Current carbon intensity for {self.zone}: {carbon_intensity:.1f} gCO2/kWh at {timestamp.isoformat()}"
            )
            
            return {
                "carbon_intensity": float(carbon_intensity),
                "timestamp": timestamp,
                "zone": self.zone,
            }
            
        except requests.HTTPError as e:
            logger.error(f"API request failed: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to get current carbon intensity: {e}")
            raise
    
    def get_forecast(self, hours: int = 24) -> Optional[List[Dict[str, any]]]:
        """
        Get carbon intensity forecast for next N hours.
        May not be available on free tier - returns None if unavailable.
        """
        # check cache
        if self._forecast_available is False:
            logger.debug("Forecast unavailable, skipping API call")
            return None
        
        endpoint = f"{self.BASE_URL}/carbon-intensity/forecast"
        params = {"zone": self.zone}
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            # handle forecast unavailable gracefully
            if response.status_code in [403, 404]:
                logger.warning(
                    f"Forecast endpoint unavailable (status {response.status_code}). "
                    "May require paid plan. Falling back to current-intensity-only mode."
                )
                self._forecast_available = False
                return None
            
            response.raise_for_status()
            data = response.json()
            
            forecast_list = data.get("forecast", [])
            if not forecast_list:
                logger.warning("Forecast response contains no data")
                return None
            
            # convert to our format
            forecast = []
            for point in forecast_list[:hours]:
                carbon_intensity = point.get("carbonIntensity")
                timestamp_str = point.get("datetime")
                
                if carbon_intensity is not None and timestamp_str:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    forecast.append({
                        "carbon_intensity": float(carbon_intensity),
                        "timestamp": timestamp,
                    })
            
            self._forecast_available = True
            logger.info(f"Retrieved {len(forecast)} forecast points for {self.zone}")
            
            return forecast
            
        except requests.HTTPError as e:
            logger.error(f"Forecast request failed: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to get forecast: {e}")
            raise
    
    def is_forecast_available(self) -> bool:
        """Check if forecast endpoint works for this API key"""
        if self._forecast_available is not None:
            return self._forecast_available
        
        # try to get forecast to check
        try:
            forecast = self.get_forecast(hours=1)
            return forecast is not None
        except Exception:
            return False
