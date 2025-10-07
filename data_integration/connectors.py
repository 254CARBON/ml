"""Data connectors for 254Carbon financial data sources."""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import httpx
import structlog
from pathlib import Path

logger = structlog.get_logger("data_connectors")


class DataConnector:
    """Base class for data connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base connector with a config mapping.

        Parameters
        - config: Connection and retrieval options (e.g., base_url, api_key)
        """
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        """Establish connection to data source."""
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def disconnect(self):
        """Close connection to data source."""
        if self.client:
            await self.client.aclose()
    
    async def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from source."""
        raise NotImplementedError


class YieldCurveConnector(DataConnector):
    """Connector for yield curve data from 254Carbon data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Set up yield-curve connector with endpoint and auth options."""
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.254carbon.internal")
        self.api_key = config.get("api_key")
        self.tenors = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    
    async def fetch_yield_curves(
        self,
        currencies: List[str] = ["USD", "EUR", "GBP"],
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        curve_types: List[str] = ["government", "swap"]
    ) -> pd.DataFrame:
        """Fetch yield curve data."""
        
        logger.info("Fetching yield curve data", 
                   currencies=currencies, 
                   start_date=start_date, 
                   end_date=end_date)
        
        # Simulate API call to 254Carbon data service
        # In production, this would make actual HTTP requests
        all_data = []
        
        for currency in currencies:
            for curve_type in curve_types:
                # Simulate realistic yield curve data
                dates = pd.date_range(start_date, end_date, freq='D')
                
                for date in dates:
                    row = {
                        "date": date,
                        "currency": currency,
                        "curve_type": curve_type,
                        "source": "254carbon_api"
                    }
                    
                    # Generate realistic rates based on currency and date
                    base_rate = self._get_base_rate(currency, date)
                    
                    for tenor in self.tenors:
                        tenor_years = self._tenor_to_years(tenor)
                        rate = self._generate_realistic_rate(base_rate, tenor_years, currency, date)
                        row[f"rate_{tenor}"] = rate
                    
                    # Add market context
                    row.update(self._get_market_context(currency, date))
                    
                    all_data.append(row)
        
        df = pd.DataFrame(all_data)
        logger.info("Yield curve data fetched", shape=df.shape)
        return df
    
    def _get_base_rate(self, currency: str, date: datetime) -> float:
        """Get base rate for currency and date."""
        base_rates = {
            "USD": 0.025,
            "EUR": 0.015,
            "GBP": 0.030,
            "JPY": 0.005
        }
        
        base = base_rates.get(currency, 0.025)
        
        # Add time-based variation
        time_factor = np.sin(2 * np.pi * date.dayofyear / 365) * 0.01
        crisis_factor = -0.015 if date.year == 2020 else 0  # COVID impact
        
        return max(0.001, base + time_factor + crisis_factor)
    
    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor string to years."""
        if tenor.endswith("M"):
            return int(tenor[:-1]) / 12
        elif tenor.endswith("Y"):
            return int(tenor[:-1])
        else:
            return 1.0
    
    def _generate_realistic_rate(self, base_rate: float, tenor_years: float, currency: str, date: datetime) -> float:
        """Generate realistic rate for tenor."""
        # Yield curve shape
        if tenor_years <= 0.5:
            rate = base_rate + 0.001 * np.random.normal()
        elif tenor_years <= 2:
            rate = base_rate + 0.005 * tenor_years + 0.001 * np.random.normal()
        else:
            rate = base_rate + 0.01 + 0.002 * np.log(tenor_years) + 0.001 * np.random.normal()
        
        return max(0.001, rate)
    
    def _get_market_context(self, currency: str, date: datetime) -> Dict[str, float]:
        """Get market context for the date."""
        return {
            "vix": 20 + 10 * np.sin(date.dayofyear / 100) + 5 * np.random.normal(),
            "dxy": 95 + 5 * np.sin(date.dayofyear / 200) + 2 * np.random.normal(),
            "oil_price": 60 + 20 * np.sin(date.dayofyear / 150) + 5 * np.random.normal(),
            "gold_price": 1800 + 200 * np.sin(date.dayofyear / 180) + 50 * np.random.normal()
        }


class CommodityCurveConnector(DataConnector):
    """Connector for commodity forward curve data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Configure commodity connector with endpoint and default commodity set."""
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.254carbon.internal")
        self.commodities = ["NG", "CL", "HO", "RB", "GC", "SI"]  # Gas, Oil, Heating Oil, RBOB, Gold, Silver
    
    async def fetch_commodity_curves(
        self,
        commodities: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        contract_months: int = 24
    ) -> pd.DataFrame:
        """Fetch commodity forward curve data."""
        
        if commodities is None:
            commodities = self.commodities
        
        logger.info("Fetching commodity curve data", 
                   commodities=commodities, 
                   start_date=start_date, 
                   end_date=end_date)
        
        all_data = []
        
        for commodity in commodities:
            # Simulate API call
            dates = pd.date_range(start_date, end_date, freq='D')
            
            for date in dates:
                row = {
                    "date": date,
                    "commodity": commodity,
                    "source": "254carbon_api"
                }
                
                # Generate spot price
                spot_price = self._get_spot_price(commodity, date)
                row["spot_price"] = spot_price
                
                # Generate forward curve
                for month in range(1, contract_months + 1):
                    forward_price = self._generate_forward_price(spot_price, month, commodity, date)
                    row[f"forward_{month}m"] = forward_price
                
                # Add market fundamentals
                row.update(self._get_commodity_fundamentals(commodity, date))
                
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        logger.info("Commodity curve data fetched", shape=df.shape)
        return df
    
    def _get_spot_price(self, commodity: str, date: datetime) -> float:
        """Get spot price for commodity."""
        base_prices = {
            "NG": 3.0,    # $/MMBtu
            "CL": 60.0,   # $/barrel
            "HO": 2.0,    # $/gallon
            "RB": 2.1,    # $/gallon
            "GC": 1800.0, # $/oz
            "SI": 25.0    # $/oz
        }
        
        base = base_prices.get(commodity, 50.0)
        
        # Add seasonality and volatility
        seasonal = 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
        volatility = 0.3 * np.random.normal()
        
        return max(0.1, base * (1 + seasonal + volatility))
    
    def _generate_forward_price(self, spot_price: float, month: int, commodity: str, date: datetime) -> float:
        """Generate forward price for specific month."""
        # Storage costs and convenience yield
        storage_cost = 0.02 * month / 12
        convenience_yield = 0.01 * np.exp(-month / 12)
        
        # Seasonal patterns
        if commodity in ["NG", "HO"]:  # Heating commodities
            seasonal = 0.1 * np.sin(2 * np.pi * (month + date.month) / 12)
        else:
            seasonal = 0.05 * np.sin(2 * np.pi * month / 24)
        
        forward_price = spot_price * (1 + storage_cost - convenience_yield + seasonal)
        return max(0.1, forward_price)
    
    def _get_commodity_fundamentals(self, commodity: str, date: datetime) -> Dict[str, float]:
        """Get commodity fundamentals."""
        base_fundamentals = {
            "inventory": 50 + 20 * np.sin(2 * np.pi * date.dayofyear / 365) + 5 * np.random.normal(),
            "production": 100 + 10 * np.sin(date.dayofyear / 100) + 2 * np.random.normal(),
            "consumption": 95 + 15 * np.sin(2 * np.pi * date.dayofyear / 365) + 3 * np.random.normal()
        }
        
        if commodity == "NG":
            base_fundamentals["temperature"] = 50 + 30 * np.sin(2 * np.pi * date.dayofyear / 365)
            base_fundamentals["heating_degree_days"] = max(0, 65 - base_fundamentals["temperature"])
        
        return base_fundamentals


class FXCurveConnector(DataConnector):
    """Connector for FX forward curve data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FX connector with endpoint and default currency pairs."""
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.254carbon.internal")
        self.currency_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]
    
    async def fetch_fx_curves(
        self,
        currency_pairs: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        forward_tenors: List[str] = None
    ) -> pd.DataFrame:
        """Fetch FX forward curve data."""
        
        if currency_pairs is None:
            currency_pairs = self.currency_pairs
        
        if forward_tenors is None:
            forward_tenors = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
        
        logger.info("Fetching FX curve data", 
                   pairs=currency_pairs, 
                   start_date=start_date, 
                   end_date=end_date)
        
        all_data = []
        
        for pair in currency_pairs:
            dates = pd.date_range(start_date, end_date, freq='D')
            
            for date in dates:
                row = {
                    "date": date,
                    "currency_pair": pair,
                    "source": "254carbon_api"
                }
                
                # Generate spot rate
                spot_rate = self._get_spot_rate(pair, date)
                row["spot_rate"] = spot_rate
                
                # Get interest rates
                domestic_rate, foreign_rate = self._get_interest_rates(pair, date)
                row["domestic_rate"] = domestic_rate
                row["foreign_rate"] = foreign_rate
                
                # Generate forward curve
                for tenor in forward_tenors:
                    tenor_days = self._tenor_to_days(tenor)
                    forward_rate = self._generate_forward_rate(
                        spot_rate, domestic_rate, foreign_rate, tenor_days
                    )
                    row[f"forward_{tenor}"] = forward_rate
                
                # Add volatility data
                row.update(self._get_fx_volatility_data(pair, date))
                
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        logger.info("FX curve data fetched", shape=df.shape)
        return df
    
    def _get_spot_rate(self, pair: str, date: datetime) -> float:
        """Get spot FX rate."""
        base_rates = {
            "EURUSD": 1.20,
            "GBPUSD": 1.35,
            "USDJPY": 110.0,
            "USDCHF": 0.95,
            "AUDUSD": 0.75,
            "USDCAD": 1.25
        }
        
        base = base_rates.get(pair, 1.0)
        
        # Add realistic FX volatility
        daily_return = np.random.normal(0, 0.01)  # 1% daily vol
        trend = 0.0001 * (date - datetime(2020, 1, 1)).days
        
        return max(0.01, base * (1 + daily_return + trend))
    
    def _get_interest_rates(self, pair: str, date: datetime) -> tuple:
        """Get domestic and foreign interest rates."""
        # Simplified interest rate mapping
        if pair.startswith("USD"):
            domestic_rate = 0.025 + 0.01 * np.sin(date.dayofyear / 100)
            if pair == "USDJPY":
                foreign_rate = 0.005 + 0.002 * np.sin(date.dayofyear / 150)
            elif pair == "USDCHF":
                foreign_rate = 0.010 + 0.005 * np.sin(date.dayofyear / 120)
            else:
                foreign_rate = 0.015 + 0.008 * np.sin(date.dayofyear / 130)
        else:
            # EUR, GBP base currencies
            domestic_rate = 0.015 + 0.008 * np.sin(date.dayofyear / 120)
            foreign_rate = 0.025 + 0.01 * np.sin(date.dayofyear / 100)  # USD
        
        return domestic_rate, foreign_rate
    
    def _tenor_to_days(self, tenor: str) -> int:
        """Convert tenor string to days."""
        if tenor.endswith("W"):
            return int(tenor[:-1]) * 7
        elif tenor.endswith("M"):
            return int(tenor[:-1]) * 30
        elif tenor.endswith("Y"):
            return int(tenor[:-1]) * 365
        else:
            return 30
    
    def _generate_forward_rate(
        self,
        spot_rate: float,
        domestic_rate: float,
        foreign_rate: float,
        tenor_days: int
    ) -> float:
        """Generate forward rate using interest rate parity."""
        tenor_years = tenor_days / 365.25
        rate_differential = (domestic_rate - foreign_rate) * tenor_years
        
        # Interest rate parity with small deviations
        forward_rate = spot_rate * np.exp(rate_differential + 0.001 * np.random.normal())
        
        return max(0.01, forward_rate)
    
    def _get_fx_volatility_data(self, pair: str, date: datetime) -> Dict[str, float]:
        """Get FX volatility data."""
        base_vol = 0.12 if "JPY" in pair else 0.15
        
        return {
            "implied_vol_1m": base_vol + 0.05 * np.random.normal(),
            "implied_vol_3m": base_vol + 0.03 * np.random.normal(),
            "implied_vol_1y": base_vol + 0.02 * np.random.normal(),
            "risk_reversal_1m": 0.01 * np.random.normal(),
            "butterfly_1m": max(0, 0.005 + 0.002 * np.random.normal())
        }


class InstrumentMetadataConnector(DataConnector):
    """Connector for financial instrument metadata."""
    
    async def fetch_instrument_metadata(
        self,
        instrument_types: List[str] = None,
        sectors: List[str] = None
    ) -> pd.DataFrame:
        """Fetch instrument metadata for embedding generation."""
        
        if instrument_types is None:
            instrument_types = ["futures", "options", "swaps", "bonds", "equities"]
        
        if sectors is None:
            sectors = ["energy", "metals", "agriculture", "rates", "fx", "equity"]
        
        logger.info("Fetching instrument metadata", types=instrument_types, sectors=sectors)
        
        instruments = []
        
        # Generate realistic instrument metadata
        for sector in sectors:
            sector_instruments = self._generate_sector_instruments(sector)
            instruments.extend(sector_instruments)
        
        df = pd.DataFrame(instruments)
        logger.info("Instrument metadata fetched", count=len(df))
        return df
    
    def _generate_sector_instruments(self, sector: str) -> List[Dict[str, Any]]:
        """Generate instruments for a specific sector."""
        instruments = []
        
        if sector == "energy":
            base_instruments = [
                ("NG", "Natural Gas", "Henry Hub", "NYMEX"),
                ("CL", "Crude Oil", "WTI", "NYMEX"),
                ("HO", "Heating Oil", "ULSD", "NYMEX"),
                ("RB", "Gasoline", "RBOB", "NYMEX"),
                ("BZ", "Brent Crude", "ICE Brent", "ICE")
            ]
            
            for symbol, name, description, exchange in base_instruments:
                # Generate multiple contract months
                for month in range(1, 25):  # 24 months out
                    instruments.append({
                        "instrument_id": f"{symbol}_M{month:02d}",
                        "symbol": symbol,
                        "name": f"{name} {month}M Futures",
                        "description": f"{description} {month}-month futures contract",
                        "sector": sector,
                        "asset_class": "commodity",
                        "exchange": exchange,
                        "contract_month": month,
                        "currency": "USD",
                        "unit": "MMBtu" if symbol == "NG" else "barrel",
                        "tick_size": 0.001 if symbol == "NG" else 0.01,
                        "active": True
                    })
        
        elif sector == "rates":
            rate_instruments = [
                ("ZN", "10-Year Treasury Note", "US 10Y Treasury"),
                ("ZB", "30-Year Treasury Bond", "US 30Y Treasury"),
                ("ZF", "5-Year Treasury Note", "US 5Y Treasury"),
                ("ZT", "2-Year Treasury Note", "US 2Y Treasury")
            ]
            
            for symbol, name, description in rate_instruments:
                for month in range(1, 13):  # 12 months
                    instruments.append({
                        "instrument_id": f"{symbol}_M{month:02d}",
                        "symbol": symbol,
                        "name": f"{name} {month}M Futures",
                        "description": f"{description} {month}-month futures contract",
                        "sector": sector,
                        "asset_class": "rates",
                        "exchange": "CBOT",
                        "contract_month": month,
                        "currency": "USD",
                        "active": True
                    })
        
        elif sector == "fx":
            fx_pairs = [
                ("EUR", "USD", "Euro/US Dollar"),
                ("GBP", "USD", "British Pound/US Dollar"),
                ("USD", "JPY", "US Dollar/Japanese Yen"),
                ("AUD", "USD", "Australian Dollar/US Dollar")
            ]
            
            for base, quote, description in fx_pairs:
                pair = f"{base}{quote}"
                instruments.append({
                    "instrument_id": f"{pair}_SPOT",
                    "symbol": pair,
                    "name": f"{base}/{quote} Spot",
                    "description": f"{description} spot rate",
                    "sector": sector,
                    "asset_class": "fx",
                    "exchange": "OTC",
                    "base_currency": base,
                    "quote_currency": quote,
                    "active": True
                })
        
        return instruments


class DataQualityValidator:
    """Validates data quality for ML training."""
    
    def __init__(self):
        """Initialize validation rule thresholds for checks."""
        self.validation_rules = {
            "completeness": 0.95,  # 95% non-null values
            "consistency": 0.99,   # 99% consistent formats
            "accuracy": 0.98,      # 98% within expected ranges
            "timeliness": 24       # Data not older than 24 hours
        }
    
    def validate_dataset(self, df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
        """Validate dataset quality."""
        
        logger.info("Validating dataset quality", dataset_type=dataset_type, shape=df.shape)
        
        validation_results = {
            "dataset_type": dataset_type,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "validation_timestamp": datetime.now().isoformat(),
            "passed": True,
            "issues": []
        }
        
        # Completeness check
        null_percentages = df.isnull().sum() / len(df)
        high_null_cols = null_percentages[null_percentages > (1 - self.validation_rules["completeness"])]
        
        if len(high_null_cols) > 0:
            validation_results["passed"] = False
            validation_results["issues"].append({
                "type": "completeness",
                "severity": "high",
                "description": f"Columns with high null percentage: {list(high_null_cols.index)}",
                "details": high_null_cols.to_dict()
            })
        
        # Range validation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith("rate_") and (df[col] < 0).any():
                validation_results["passed"] = False
                validation_results["issues"].append({
                    "type": "accuracy",
                    "severity": "high",
                    "description": f"Negative rates found in {col}",
                    "count": (df[col] < 0).sum()
                })
            
            if col.endswith("_price") and (df[col] <= 0).any():
                validation_results["passed"] = False
                validation_results["issues"].append({
                    "type": "accuracy",
                    "severity": "high",
                    "description": f"Non-positive prices found in {col}",
                    "count": (df[col] <= 0).sum()
                })
        
        # Timeliness check
        if "date" in df.columns:
            latest_date = df["date"].max()
            if isinstance(latest_date, pd.Timestamp):
                days_old = (datetime.now() - latest_date.to_pydatetime()).days
                if days_old > self.validation_rules["timeliness"]:
                    validation_results["issues"].append({
                        "type": "timeliness",
                        "severity": "medium",
                        "description": f"Data is {days_old} days old",
                        "latest_date": latest_date.isoformat()
                    })
        
        # Consistency checks
        if dataset_type == "yield_curves":
            # Check yield curve shape consistency
            rate_cols = [col for col in df.columns if col.startswith("rate_")]
            if len(rate_cols) >= 2:
                # Check for inverted curves (simplified)
                short_rate = df[rate_cols[0]]
                long_rate = df[rate_cols[-1]]
                inverted_count = (short_rate > long_rate + 0.01).sum()  # Allow 1bp tolerance
                
                if inverted_count > len(df) * 0.1:  # More than 10% inverted
                    validation_results["issues"].append({
                        "type": "consistency",
                        "severity": "medium",
                        "description": f"High number of inverted yield curves: {inverted_count}",
                        "percentage": inverted_count / len(df) * 100
                    })
        
        validation_results["quality_score"] = self._calculate_quality_score(validation_results)
        
        logger.info("Dataset validation completed", 
                   passed=validation_results["passed"],
                   quality_score=validation_results["quality_score"],
                   issues=len(validation_results["issues"]))
        
        return validation_results
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        base_score = 100.0
        
        for issue in validation_results["issues"]:
            if issue["severity"] == "high":
                base_score -= 20
            elif issue["severity"] == "medium":
                base_score -= 10
            elif issue["severity"] == "low":
                base_score -= 5
        
        return max(0.0, base_score)


class DataPipelineOrchestrator:
    """Orchestrates data fetching from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Wire connectors and validator from a composite config.

        Parameters
        - config: Dict of per-source configs keyed by connector name
        """
        self.config = config
        self.connectors = {
            "yield_curves": YieldCurveConnector(config.get("yield_curves", {})),
            "commodity_curves": CommodityCurveConnector(config.get("commodity_curves", {})),
            "fx_curves": FXCurveConnector(config.get("fx_curves", {})),
            "instruments": InstrumentMetadataConnector(config.get("instruments", {}))
        }
        self.validator = DataQualityValidator()
    
    async def fetch_all_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        output_dir: str = "data/real"
    ) -> Dict[str, Any]:
        """Fetch data from all sources."""
        
        logger.info("Starting data pipeline orchestration")
        
        results = {}
        
        # Connect to all sources
        for name, connector in self.connectors.items():
            await connector.connect()
        
        try:
            # Fetch yield curves
            logger.info("Fetching yield curve data")
            yield_data = await self.connectors["yield_curves"].fetch_yield_curves(
                currencies=["USD", "EUR", "GBP"],
                start_date=start_date,
                end_date=end_date
            )
            
            # Validate and save
            validation = self.validator.validate_dataset(yield_data, "yield_curves")
            results["yield_curves"] = {
                "data": yield_data,
                "validation": validation,
                "shape": yield_data.shape
            }
            
            # Save to disk
            import os
            os.makedirs(output_dir, exist_ok=True)
            yield_data.to_parquet(f"{output_dir}/yield_curves.parquet", index=False)
            
            # Fetch commodity curves
            logger.info("Fetching commodity curve data")
            commodity_data = await self.connectors["commodity_curves"].fetch_commodity_curves(
                commodities=["NG", "CL", "HO"],
                start_date=start_date,
                end_date=end_date
            )
            
            validation = self.validator.validate_dataset(commodity_data, "commodity_curves")
            results["commodity_curves"] = {
                "data": commodity_data,
                "validation": validation,
                "shape": commodity_data.shape
            }
            
            commodity_data.to_parquet(f"{output_dir}/commodity_curves.parquet", index=False)
            
            # Fetch FX curves
            logger.info("Fetching FX curve data")
            fx_data = await self.connectors["fx_curves"].fetch_fx_curves(
                currency_pairs=["EURUSD", "GBPUSD", "USDJPY"],
                start_date=start_date,
                end_date=end_date
            )
            
            validation = self.validator.validate_dataset(fx_data, "fx_curves")
            results["fx_curves"] = {
                "data": fx_data,
                "validation": validation,
                "shape": fx_data.shape
            }
            
            fx_data.to_parquet(f"{output_dir}/fx_curves.parquet", index=False)
            
            # Fetch instrument metadata
            logger.info("Fetching instrument metadata")
            instrument_data = await self.connectors["instruments"].fetch_instrument_metadata()
            
            validation = self.validator.validate_dataset(instrument_data, "instruments")
            results["instruments"] = {
                "data": instrument_data,
                "validation": validation,
                "shape": instrument_data.shape
            }
            
            instrument_data.to_parquet(f"{output_dir}/instruments.parquet", index=False)
            
            # Generate summary report
            summary = self._generate_summary_report(results)
            results["summary"] = summary
            
            # Save summary
            with open(f"{output_dir}/data_summary.json", "w") as f:
                import json
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Data pipeline orchestration completed", summary=summary)
            
        finally:
            # Disconnect from all sources
            for connector in self.connectors.values():
                await connector.disconnect()
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of data fetching."""
        
        total_rows = sum(result["shape"][0] for result in results.values() if "shape" in result)
        total_columns = sum(result["shape"][1] for result in results.values() if "shape" in result)
        
        passed_validations = sum(1 for result in results.values() 
                               if "validation" in result and result["validation"]["passed"])
        
        quality_scores = [result["validation"]["quality_score"] 
                         for result in results.values() 
                         if "validation" in result]
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0
        
        return {
            "total_datasets": len([r for r in results.values() if "data" in r]),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "passed_validations": passed_validations,
            "average_quality_score": avg_quality_score,
            "datasets": {
                name: {
                    "rows": result["shape"][0],
                    "columns": result["shape"][1],
                    "quality_score": result["validation"]["quality_score"],
                    "passed": result["validation"]["passed"]
                }
                for name, result in results.items()
                if "shape" in result
            }
        }


async def main():
    """Main data integration function."""
    config = {
        "yield_curves": {
            "base_url": "https://api.254carbon.internal/curves",
            "api_key": "your_api_key_here"
        },
        "commodity_curves": {
            "base_url": "https://api.254carbon.internal/commodities",
            "api_key": "your_api_key_here"
        },
        "fx_curves": {
            "base_url": "https://api.254carbon.internal/fx",
            "api_key": "your_api_key_here"
        },
        "instruments": {
            "base_url": "https://api.254carbon.internal/instruments",
            "api_key": "your_api_key_here"
        }
    }
    
    orchestrator = DataPipelineOrchestrator(config)
    results = await orchestrator.fetch_all_data()
    
    print("Data integration completed:")
    for dataset, result in results.items():
        if "shape" in result:
            print(f"  {dataset}: {result['shape'][0]} rows, quality score: {result['validation']['quality_score']:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
