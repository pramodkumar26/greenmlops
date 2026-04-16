"""
electricity_maps_client.py

Client for Electricity Maps API to fetch live carbon intensity data.
Replaces the historical CAISO CSV with real-time grid carbon intensity.

API Documentation: https://docs.electricitymaps.com/
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
import requests

logger = logging.getLogger(__name__)


class ElectricityMapsClient:
    """
    Client for Electricity Maps API.
    
    Supports two modes:
    - Current intensity: get_current_intensity() - available on free tier
    - Forecast: get_forecast() - may require paid tier
    
    Falls back gracefully if forecast is unavailable.
    """
    
    BASE_URL = "https://api.electricitymap.org/v3"
    
    def __init__(self, api_key: Optional[str] = None, zone: str = "US-CAL-CISO"):
        """
        Initialize the Electricity Maps client.
        
        Parameters
        ----------
        api_key : str, optional
            API key for Electricity Maps. If not provided, reads from
            ELECTRICITY_MAPS_API_KEY environment variable.
        zone : str, default "US-CAL-CISO"
            Grid zone identifier. CAISO is the California ISO zone.
        """
        self.api_key = api_key or os.getenv("ELECTRICITY_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Electricity Maps API key not found. "
                "Set ELECTRICITY_MAPS_API_KEY environment variable or pass api_key parameter."
            )
        
        self.zone = zone
        self.headers = {"auth-token": self.api_key}
        self._forecast_available = None  # Cache forecast availability check
        
        logger.info(
            "ElectricityMapsClient initialized for zone %s", self.zone
        )
    
    def get_current_intensity(self) -> Dict[str, any]:
        """
        Get current carbon intensity for the configured zone.
        
        Returns
        -------
        dict with keys:
            carbon_intensity : float
                Current carbon intensity in gCO2eq/kWh
            timestamp : datetime
                Timestamp of the reading (UTC)
            zone : str
                Grid zone identifier
        
        Raises
        ------
        requests.HTTPError
            If API request fails
        ValueError
            If response format is unexpected
        """
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
            
            # Parse response
            carbon_intensity = data.get("carbonIntensity")
            timestamp_str = data.get("datetime")
            
            if carbon_intensity is None:
                raise ValueError(
                    f"carbonIntensity not found in API response: {data}"
                )
            
            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str
                else datetime.now(timezone.utc)
            )
            
            logger.info(
                "Current carbon intensity for %s: %.1f gCO2/kWh at %s",
                self.zone, carbon_intensity, timestamp.isoformat()
            )
            
            return {
                "carbon_intensity": float(carbon_intensity),
                "timestamp": timestamp,
                "zone": self.zone,
            }
            
        except requests.HTTPError as e:
            logger.error(
                "Electricity Maps API request failed: %s - %s",
                e.response.status_code, e.response.text
            )
            raise
        except Exception as e:
            logger.error("Failed to get current carbon intensity: %s", e)
            raise
    
    def get_forecast(self, hours: int = 24) -> Optional[List[Dict[str, any]]]:
        """
        Get carbon intensity forecast for the next N hours.
        
        Note: Forecast endpoint may require a paid Electricity Maps plan.
        Returns None if forecast is unavailable (free tier limitation).
        
        Parameters
        ----------
        hours : int, default 24
            Number of hours to forecast. API typically provides up to 24h.
        
        Returns
        -------
        list of dict or None
            List of forecast points, each with:
                carbon_intensity : float
                timestamp : datetime
            Returns None if forecast endpoint is unavailable.
        
        Raises
        ------
        requests.HTTPError
            If API request fails (other than 403/404 for unavailable forecast)
        """
        # Check cache if we already know forecast is unavailable
        if self._forecast_available is False:
            logger.debug("Forecast known to be unavailable, skipping API call")
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
            
            # Handle forecast unavailable gracefully
            if response.status_code in [403, 404]:
                logger.warning(
                    "Forecast endpoint unavailable (status %d). "
                    "This may require a paid Electricity Maps plan. "
                    "Scheduler will fall back to current-intensity-only mode.",
                    response.status_code
                )
                self._forecast_available = False
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Parse forecast data
            forecast_list = data.get("forecast", [])
            if not forecast_list:
                logger.warning("Forecast response contains no data points")
                return None
            
            # Convert to our format
            forecast = []
            for point in forecast_list[:hours]:  # Limit to requested hours
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
            logger.info(
                "Retrieved %d forecast points for %s (next %d hours)",
                len(forecast), self.zone, hours
            )
            
            return forecast
            
        except requests.HTTPError as e:
            logger.error(
                "Electricity Maps forecast request failed: %s - %s",
                e.response.status_code, e.response.text
            )
            raise
        except Exception as e:
            logger.error("Failed to get forecast: %s", e)
            raise
    
    def is_forecast_available(self) -> bool:
        """
        Check if forecast endpoint is available for this API key.
        
        Returns
        -------
        bool
            True if forecast is available, False otherwise.
        """
        if self._forecast_available is not None:
            return self._forecast_available
        
        # Try to get forecast to check availability
        try:
            forecast = self.get_forecast(hours=1)
            return forecast is not None
        except Exception:
            return False
