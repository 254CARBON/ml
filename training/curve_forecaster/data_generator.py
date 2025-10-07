"""Synthetic financial curve data generators for model training.

This module produces three families of datasets commonly used in curve
forecasting experiments:
- Yield curves: ``rate_{tenor}y`` columns across standard tenors.
- Commodity forward curves: monthly ``forward_{m}m`` prices and features.
- FX forward curves: daily tenor forward points with spot and IR differentials.

Each generator returns a ``pandas.DataFrame`` with a ``date`` column and
feature columns suitable for time series or panel modeling. The
``create_training_dataset`` helper produces multiple datasets and writes
partitioned Parquet files to disk.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import structlog

logger = structlog.get_logger("data_generator")


class CurveDataGenerator:
    """Generate synthetic yield curve and forward curve data."""
    
    def __init__(self, seed: int = 42):
        """Initialize RNG with a fixed seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_yield_curve_data(
        self,
        n_samples: int = 1000,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Generate synthetic yield curve data."""
        
        # Define curve tenors (in years)
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start, end, periods=n_samples)
        
        data = []
        
        # Initialize base curve parameters
        base_short_rate = 0.02  # 2% base short rate
        base_long_rate = 0.04   # 4% base long rate
        
        for i, date in enumerate(dates):
            # Add market regime changes and volatility
            market_cycle = np.sin(2 * np.pi * i / 252)  # Annual cycle
            economic_shock = np.random.normal(0, 0.001)  # Random shocks
            
            # Generate curve with realistic shape
            rates = []
            for tenor in tenors:
                # Nelson-Siegel-like curve shape
                beta0 = base_long_rate + 0.01 * market_cycle + economic_shock
                beta1 = (base_short_rate - base_long_rate) + 0.005 * market_cycle
                beta2 = 0.01 * np.sin(i / 100) + 0.002 * np.random.normal()
                tau = 2.0
                
                rate = (beta0 + 
                       beta1 * (1 - np.exp(-tenor/tau)) / (tenor/tau) +
                       beta2 * ((1 - np.exp(-tenor/tau)) / (tenor/tau) - np.exp(-tenor/tau)))
                
                # Add tenor-specific noise
                rate += np.random.normal(0, 0.0005)
                
                # Ensure positive rates
                rate = max(rate, 0.001)
                rates.append(rate)
            
            # Create row
            row = {"date": date}
            for tenor, rate in zip(tenors, rates):
                row[f"rate_{tenor}y"] = rate
            
            # Add market features
            row["vix"] = 20 + 10 * np.sin(i / 50) + 5 * np.random.normal()
            row["fed_funds"] = max(0, base_short_rate + 0.01 * market_cycle + 0.005 * np.random.normal())
            row["unemployment"] = max(3, 6 + 2 * np.sin(i / 100) + np.random.normal())
            row["inflation"] = max(0, 0.02 + 0.01 * market_cycle + 0.002 * np.random.normal())
            
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info("Generated yield curve data", samples=len(df), tenors=len(tenors))
        return df
    
    def generate_commodity_curve_data(
        self,
        commodity: str = "NG",  # Natural Gas
        n_samples: int = 1000,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Generate synthetic commodity forward curve data."""
        
        # Define forward months (1M to 36M)
        forward_months = list(range(1, 37))
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start, end, periods=n_samples)
        
        data = []
        
        # Base commodity parameters
        if commodity == "NG":
            base_price = 3.0  # $/MMBtu
            seasonality_amplitude = 1.0
            volatility = 0.3
        elif commodity == "CL":  # Crude Oil
            base_price = 60.0  # $/barrel
            seasonality_amplitude = 5.0
            volatility = 0.25
        else:
            base_price = 50.0
            seasonality_amplitude = 2.0
            volatility = 0.2
        
        for i, date in enumerate(dates):
            # Seasonal patterns
            seasonal_factor = seasonality_amplitude * np.sin(2 * np.pi * (i % 365) / 365)
            
            # Market shocks and trends
            trend = 0.001 * i  # Slight upward trend
            shock = volatility * np.random.normal(0, 0.01)
            
            # Generate forward curve
            spot_price = base_price + seasonal_factor + trend + shock
            spot_price = max(spot_price, 0.1)  # Ensure positive prices
            
            prices = []
            for month in forward_months:
                # Contango/backwardation structure
                if commodity == "NG":
                    # Natural gas typically in contango in summer, backwardation in winter
                    month_seasonal = 0.1 * np.sin(2 * np.pi * (month + i % 12) / 12)
                    storage_cost = 0.02 * month / 12  # Storage costs
                else:
                    # Oil typically slight contango
                    month_seasonal = 0.05 * np.sin(2 * np.pi * month / 24)
                    storage_cost = 0.01 * month / 12
                
                forward_price = spot_price * (1 + storage_cost + month_seasonal + 0.001 * np.random.normal())
                forward_price = max(forward_price, 0.1)
                prices.append(forward_price)
            
            # Create row
            row = {"date": date, "commodity": commodity, "spot_price": spot_price}
            for month, price in zip(forward_months, prices):
                row[f"forward_{month}m"] = price
            
            # Add market features
            row["inventory"] = 50 + 20 * np.sin(2 * np.pi * (i % 365) / 365) + 5 * np.random.normal()
            row["temperature"] = 50 + 30 * np.sin(2 * np.pi * (i % 365) / 365) + 5 * np.random.normal()
            row["production"] = 100 + 10 * np.sin(i / 100) + 2 * np.random.normal()
            
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info("Generated commodity curve data", commodity=commodity, samples=len(df))
        return df
    
    def generate_fx_curve_data(
        self,
        currency_pair: str = "EURUSD",
        n_samples: int = 1000,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Generate synthetic FX forward curve data."""
        
        # Define forward tenors (in days)
        forward_tenors = [7, 14, 30, 60, 90, 180, 270, 365]
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start, end, periods=n_samples)
        
        data = []
        
        # Base FX parameters
        if currency_pair == "EURUSD":
            base_rate = 1.20
            volatility = 0.15
        elif currency_pair == "GBPUSD":
            base_rate = 1.35
            volatility = 0.18
        elif currency_pair == "USDJPY":
            base_rate = 110.0
            volatility = 0.12
        else:
            base_rate = 1.0
            volatility = 0.15
        
        for i, date in enumerate(dates):
            # Random walk with mean reversion
            mean_reversion = 0.001 * (base_rate - (base_rate if i == 0 else data[-1]["spot_rate"]))
            shock = volatility * np.random.normal(0, 0.01)
            
            if i == 0:
                spot_rate = base_rate
            else:
                spot_rate = data[-1]["spot_rate"] + mean_reversion + shock
            
            spot_rate = max(spot_rate, 0.01)  # Ensure positive rates
            
            # Generate forward curve based on interest rate differentials
            base_domestic_rate = 0.02 + 0.01 * np.sin(i / 100)
            base_foreign_rate = 0.015 + 0.008 * np.sin(i / 120)
            
            forwards = []
            for tenor_days in forward_tenors:
                # Interest rate parity
                tenor_years = tenor_days / 365.25
                rate_diff = (base_domestic_rate - base_foreign_rate) * tenor_years
                forward_rate = spot_rate * np.exp(rate_diff + 0.001 * np.random.normal())
                forwards.append(forward_rate)
            
            # Create row
            row = {
                "date": date,
                "currency_pair": currency_pair,
                "spot_rate": spot_rate,
                "domestic_rate": base_domestic_rate,
                "foreign_rate": base_foreign_rate
            }
            
            for tenor, forward in zip(forward_tenors, forwards):
                row[f"forward_{tenor}d"] = forward
            
            # Add market features
            row["volatility_1m"] = max(0.05, volatility + 0.05 * np.random.normal())
            row["risk_reversal"] = 0.01 * np.random.normal()
            row["butterfly"] = max(0, 0.005 + 0.002 * np.random.normal())
            
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info("Generated FX curve data", pair=currency_pair, samples=len(df))
        return df
    
    def create_training_dataset(
        self,
        output_path: str = "data/training_data.parquet",
        n_samples: int = 2000
    ) -> Dict[str, pd.DataFrame]:
        """Create comprehensive training dataset."""
        
        logger.info("Generating comprehensive training dataset", samples=n_samples)
        
        datasets = {}
        
        # Generate yield curves
        datasets["yield_curves"] = self.generate_yield_curve_data(n_samples)
        
        # Generate commodity curves
        datasets["ng_curves"] = self.generate_commodity_curve_data("NG", n_samples)
        datasets["cl_curves"] = self.generate_commodity_curve_data("CL", n_samples)
        
        # Generate FX curves
        datasets["eurusd_curves"] = self.generate_fx_curve_data("EURUSD", n_samples)
        datasets["gbpusd_curves"] = self.generate_fx_curve_data("GBPUSD", n_samples)
        
        # Save datasets
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        for name, df in datasets.items():
            path = output_path.replace(".parquet", f"_{name}.parquet")
            df.to_parquet(path, index=False)
            logger.info("Saved dataset", name=name, path=path, shape=df.shape)
        
        return datasets


if __name__ == "__main__":
    generator = CurveDataGenerator()
    datasets = generator.create_training_dataset()
    print("Generated datasets:", list(datasets.keys()))
