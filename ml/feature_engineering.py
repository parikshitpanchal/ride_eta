"""
Advanced Feature Engineering for Ride-Hailing Delay Prediction
Implements all user-requested rolling features with strict leakage prevention
"""

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from ml.configs import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering with time-awareness to prevent leakage"""
    
    def __init__(self):
        logger.info("Initialized Feature Engineer")
    
    def ewma(self,values, alpha=0.3):
        value = values[0]
        for v in values[1:]:
            value = alpha * v + (1 - alpha) * value
        return value

    def engineer_features(self, df):
        """Main feature engineering pipeline"""
        logger.info("Starting feature engineering...")
        
        # Ensure timestamp is datetime (handles ISO strings and Excel serial floats)
        def _parse_timestamp(val):
            if pd.isna(val):
                return pd.NaT
            try:
                num_val = float(val)
                if 10000 < num_val < 60000:
                    return pd.to_datetime(num_val, unit='D', origin='1899-12-30')
            except (ValueError, TypeError):
                pass
            return pd.to_datetime(val, errors='coerce')

        df['booking_timestamp'] = df['booking_timestamp'].apply(_parse_timestamp)
        df = df.sort_values('booking_timestamp').reset_index(drop=True)
        
        # 1. Precompute features for the current order (which may be leakage but needed as raw/calculation helper)
        # individual_trip_ratio: actual total time / total estimated time (estimated pickup + estimated trip)
        df['individual_trip_ratio'] = df['actual_total_time_minutes'] / (df['google_pickup_eta_minutes'] + df['google_trip_eta_minutes'] + 0.01)
        
        # delay_severity: as specified in requirements
        df['delay_severity'] = '0'
        df.loc[(df['total_delay_minutes'] > 0) & (df['total_delay_minutes'] <= 10), 'delay_severity'] = '1'
        df.loc[(df['total_delay_minutes'] > 10) & (df['total_delay_minutes'] <= 20), 'delay_severity'] = '2'
        df.loc[df['total_delay_minutes'] > 20, 'delay_severity'] = '3'
        
        # 2. Create historical aggregation features (TIME-AWARE)
        df = self._create_historical_features(df)
        
        # 3. Fill missing values for cold-start cases
        df = self._handle_missing_values(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def _create_historical_features(self, df):
        """
        Create historical aggregation features using ONLY past data (strictly before booking_timestamp)
        """
        logger.info("Creating time-aware historical aggregation features...")
        
        # Initialize rolling features
        hist_features = [
            'average_delay', 'driver_avg_pickup_delay', 'driver_avg_trip_delay',
            'driver_last_7_day_delay_rate', 'driver_last_30_day_delay_rate',
            'driver_route_avg_delay', 'driver_zone_avg_pickup_delay', 'driver_zone_avg_trip_delay',
            'historical_route_ratio_avg', 'driver_reliability', 'driver_risk_score',
            'driver_efficiency_score', 'driver_recent_delay_rate', 'route_frequency',
            'historical_route_delay_rate', 'route_reliability_score'
        ]
        
        for feature in hist_features:
            df[feature] = np.nan
        
        # Initialize tracking dictionaries for history lookup
        driver_history = {}
        driver_route_history = {}
        driver_zone_history = {}
        route_history = {}
        
        # Lists to hold the rolling results (faster than using df.at/iloc in loop)
        average_delay_vals = []
        driver_avg_pickup_delay_vals = []
        driver_avg_trip_delay_vals = []
        driver_last_7_day_delay_rate_vals = []
        driver_last_30_day_delay_rate_vals = []
        driver_route_avg_delay_vals = []
        driver_zone_avg_pickup_delay_vals = []
        driver_zone_avg_trip_delay_vals = []
        historical_route_ratio_avg_vals = []
        driver_reliability_vals = []
        driver_risk_score_vals = []
        driver_efficiency_score_vals = []
        driver_recent_delay_rate_vals = []
        route_frequency_vals = []
        historical_route_delay_rate_vals = []
        route_reliability_score_vals = []
        
        # Process each ride sequentially
        for idx in tqdm(range(len(df)), desc="Computing historical features"):
            row = df.iloc[idx]
            driver_id = row['driver_id']
            pickup_zone = row['pickup_zone']
            drop_zone = row['drop_zone']
            current_time = row['booking_timestamp']
            cancellation_rate = row['driver_cancellation_rate']
            
            route_key = (pickup_zone, drop_zone)
            driver_route_key = (driver_id, pickup_zone, drop_zone)
            driver_zone_key = (driver_id, pickup_zone)
            
            # === 1. DRIVER HISTORICAL AGGREGATIONS ===
            if driver_id in driver_history:
                dh = driver_history[driver_id]
                
                # average_delay (pickup + trip delay)
                average_delay_vals.append(self.ewma(dh["total_delays"]))
                
                # driver_avg_pickup_delay
                driver_avg_pickup_delay_vals.append(self.ewma(dh['pickup_delays']))
                
                # driver_avg_trip_delay
                driver_avg_trip_delay_vals.append(self.ewma(dh['trip_delays']))
                
                # time windows (7 and 30 days)
                seven_days_ago = current_time - pd.Timedelta(days=7)
                thirty_days_ago = current_time - pd.Timedelta(days=30)
                
                # driver_last_7_day_delay_rate
                mask_7d = [t >= seven_days_ago for t in dh['timestamps']]
                if sum(mask_7d) > 0:
                    delays_7d = [d for d, m in zip(dh['binary_delays'], mask_7d) if m]
                    driver_last_7_day_delay_rate_vals.append(np.mean(delays_7d))
                else:
                    driver_last_7_day_delay_rate_vals.append(0.3)
                
                # driver_last_30_day_delay_rate
                mask_30d = [t >= thirty_days_ago for t in dh['timestamps']]
                if sum(mask_30d) > 0:
                    delays_30d = [d for d, m in zip(dh['binary_delays'], mask_30d) if m]
                    driver_last_30_day_delay_rate_vals.append(np.mean(delays_30d))
                else:
                    driver_last_30_day_delay_rate_vals.append(0.3)
                
                # driver_reliability: ontime / total
                # Ontime is binary_delays == 0 (non-delayed)
                ontime_count = sum(1 - np.array(dh['binary_delays']))
                reliability = ontime_count / len(dh['binary_delays'])
                driver_reliability_vals.append(reliability)
                
                # driver_risk_score: 1 - reliability
                driver_risk_score_vals.append(1.0 - reliability)
                
                # driver_efficiency_score: reliability * 0.70 + (1 - cancellation_rate) * 0.30
                driver_efficiency_score_vals.append(reliability * 0.70 + (1.0 - cancellation_rate) * 0.30)
                
                # driver_recent_delay_rate: avg of last 10 orders
                recent_delays = dh['binary_delays'][-10:]
                driver_recent_delay_rate_vals.append(np.mean(recent_delays))
            else:
                # Cold start driver defaults
                average_delay_vals.append(5.0)
                driver_avg_pickup_delay_vals.append(5.0)
                driver_avg_trip_delay_vals.append(5.0)
                driver_last_7_day_delay_rate_vals.append(0.3)
                driver_last_30_day_delay_rate_vals.append(0.3)
                driver_reliability_vals.append(0.7)
                driver_risk_score_vals.append(0.3)
                driver_efficiency_score_vals.append(0.7 * 0.70 + (1.0 - cancellation_rate) * 0.30)
                driver_recent_delay_rate_vals.append(0.3)
            
            # === 2. DRIVER ROUTE HISTORICAL AGGREGATIONS ===
            if driver_route_key in driver_route_history:
                drh = driver_route_history[driver_route_key]
                # driver_route_avg_delay
                driver_route_avg_delay_vals.append(self.ewma(drh['total_delays']))
                # route_frequency
                route_frequency_vals.append(len(drh['total_delays']))
            else:
                driver_route_avg_delay_vals.append(5.0)
                route_frequency_vals.append(0)
            
            # === 3. DRIVER ZONE HISTORICAL AGGREGATIONS ===
            if driver_zone_key in driver_zone_history:
                dzh = driver_zone_history[driver_zone_key]
                # driver_zone_avg_pickup_delay
                driver_zone_avg_pickup_delay_vals.append(self.ewma(dzh['pickup_delays']))
                # driver_zone_avg_trip_delay
                driver_zone_avg_trip_delay_vals.append(self.ewma(dzh['trip_delays']))
            else:
                driver_zone_avg_pickup_delay_vals.append(5.0)
                driver_zone_avg_trip_delay_vals.append(5.0)
            
            # === 4. ROUTE HISTORICAL AGGREGATIONS (all drivers) ===
            if route_key in route_history:
                rh = route_history[route_key]
                # historical_route_ratio_avg
                historical_route_ratio_avg_vals.append(self.ewma(rh['individual_trip_ratios']))
                # historical_route_delay_rate
                delay_rate = np.mean(rh['binary_delays'])
                historical_route_delay_rate_vals.append(delay_rate)
                # route_reliability_score
                route_reliability_score_vals.append(1.0 - delay_rate)
            else:
                historical_route_ratio_avg_vals.append(1.0)
                historical_route_delay_rate_vals.append(0.3)
                route_reliability_score_vals.append(0.7)
            
            # === 5. UPDATE HISTORIES WITH CURRENT ROW (to use for subsequent rows) ===
            is_delayed = int(row['order_delayed'])
            pickup_delay = row['pickup_delay_minutes']
            trip_delay = row['trip_delay_minutes']
            total_delay = row['total_delay_minutes']
            trip_ratio = row['individual_trip_ratio']
            
            # Update driver history
            if driver_id not in driver_history:
                driver_history[driver_id] = {
                    'binary_delays': [], 'total_delays': [], 'pickup_delays': [], 'trip_delays': [], 'timestamps': []
                }
            driver_history[driver_id]['binary_delays'].append(is_delayed)
            driver_history[driver_id]['total_delays'].append(total_delay)
            driver_history[driver_id]['pickup_delays'].append(pickup_delay)
            driver_history[driver_id]['trip_delays'].append(trip_delay)
            driver_history[driver_id]['timestamps'].append(current_time)
            
            # Update driver-route history
            if driver_route_key not in driver_route_history:
                driver_route_history[driver_route_key] = {'total_delays': []}
            driver_route_history[driver_route_key]['total_delays'].append(total_delay)
            
            # Update driver-zone history
            if driver_zone_key not in driver_zone_history:
                driver_zone_history[driver_zone_key] = {'pickup_delays': [], 'trip_delays': []}
            driver_zone_history[driver_zone_key]['pickup_delays'].append(pickup_delay)
            driver_zone_history[driver_zone_key]['trip_delays'].append(trip_delay)
            
            # Update route history
            if route_key not in route_history:
                route_history[route_key] = {'binary_delays': [], 'individual_trip_ratios': []}
            route_history[route_key]['binary_delays'].append(is_delayed)
            route_history[route_key]['individual_trip_ratios'].append(trip_ratio)
            
        # Assign to dataframe
        df['average_delay'] = average_delay_vals
        df['driver_avg_pickup_delay'] = driver_avg_pickup_delay_vals
        df['driver_avg_trip_delay'] = driver_avg_trip_delay_vals
        df['driver_last_7_day_delay_rate'] = driver_last_7_day_delay_rate_vals
        df['driver_last_30_day_delay_rate'] = driver_last_30_day_delay_rate_vals
        df['driver_route_avg_delay'] = driver_route_avg_delay_vals
        df['driver_zone_avg_pickup_delay'] = driver_zone_avg_pickup_delay_vals
        df['driver_zone_avg_trip_delay'] = driver_zone_avg_trip_delay_vals
        df['historical_route_ratio_avg'] = historical_route_ratio_avg_vals
        df['driver_reliability'] = driver_reliability_vals
        df['driver_risk_score'] = driver_risk_score_vals
        df['driver_efficiency_score'] = driver_efficiency_score_vals
        df['driver_recent_delay_rate'] = driver_recent_delay_rate_vals
        df['route_frequency'] = route_frequency_vals
        df['historical_route_delay_rate'] = historical_route_delay_rate_vals
        df['route_reliability_score'] = route_reliability_score_vals
        
        logger.info("Historical features computed successfully")
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in historical features (cold start problem)"""
        logger.info("Handling missing values...")
        
        # Fill any remaining NaN with median for numeric columns, or 0 if median is NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0
                df[col] = df[col].fillna(fill_val)
        
        return df


def main():
    """Main execution function"""
    # Check if already engineered
    output_path = config.DATA_DIR / "ride_orders_engineered_v24.csv"
    if output_path.exists():
        logger.info(f"Engineered dataset already exists at {output_path}")
        logger.info("Delete it if you want to regenerate features.")
        return
    
    # Load Ahmedabad synthetic data
    data_path = config.DATA_DIR / config.SYNTHETIC_DATA_FILE
    logger.info(f"Loading Ahmedabad data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['booking_timestamp'])
    
    # Quick demo mode check
    if hasattr(config, 'QUICK_DEMO_MODE') and config.QUICK_DEMO_MODE:
        sample_size = getattr(config, 'QUICK_DEMO_SAMPLE_SIZE', 30000)
        logger.info(f"QUICK DEMO MODE: Sampling {sample_size} rides")
        df = df.sample(n=min(sample_size, len(df)), random_state=42).sort_values('booking_timestamp').reset_index(drop=True)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    # Save engineered dataset
    df_engineered.to_csv(output_path, index=False)
    logger.info(f"Saved engineered dataset to {output_path}")
    
    logger.info(f"Final dataset shape: {df_engineered.shape}")
    logger.info("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
