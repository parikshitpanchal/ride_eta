"""
Advanced Synthetic Data Generation for Ride-Hailing Intelligence Platform
AHMEDABAD, GUJARAT, INDIA Edition - VERSION 2 (Enhanced with PDF Requirements)

Generates realistic ride-order data with:
- Multi-task targets (ETA regression, delay classification, severity)
- H3 spatial features (pickup_h3_cell, drop_h3_cell)
- Driver sequence data (last_10_rides tracking)
- Ahmedabad-specific patterns (zones, traffic, festivals)

CRITICAL RULES:
1. Delay definition: order_delayed = 1 if total_delay_minutes > 5 (NOT > 0)
2. All targets generated even if not all models trained in v1
3. Time-based split: Train (Jan-Oct), Val (Nov), Test (Dec)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import h3
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# H3 resolution for Ahmedabad (resolution 9 = ~0.1km hexagons)
H3_RESOLUTION = 9


class AhmedabadRideDataGeneratorV2:
    """Generates realistic synthetic ride-hailing data for Ahmedabad with advanced features"""
    
    def __init__(self, n_rows=30000, random_seed=42):
        self.n_rows = n_rows
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize driver pool with persistent characteristics + sequence tracking
        self.drivers = self._initialize_drivers()
        
        # Initialize zone characteristics for Ahmedabad
        self.zone_characteristics = self._initialize_ahmedabad_zones()
        
        # Initialize route frequency tracker
        self.route_frequency = {}
        
        logger.info(f"Initialized Ahmedabad V2 data generator for {n_rows} rows")
    
    def _initialize_drivers(self):
        """Create driver pool with consistent characteristics + sequence tracking"""
        drivers = {}
        for i in range(config.N_DRIVERS):
            driver_id = f"DRV_{i:05d}"
            
            # Base driver characteristics that persist over time
            base_rating = np.clip(np.random.normal(config.DRIVER_RATING_MEAN, 
                                                   config.DRIVER_RATING_STD), 1.0, 5.0)
            
            drivers[driver_id] = {
                'base_rating': base_rating,
                'experience_days': np.random.randint(30, 1800),
                'base_speed_factor': np.random.normal(1.0, 0.15),
                'reliability_factor': np.random.beta(8, 2),
                'acceptance_rate': np.random.beta(7, 3),
                'cancellation_rate': np.random.beta(2, 8),
                'ride_history': [],  # Track last 10 rides
                'last_10_delays': [],  # Track delay pattern (0/1)
                'last_10_speeds': [],  # Track avg speeds
                'last_10_durations': [],  # Track ride durations
                'preferred_zones': np.random.choice(config.ZONES, size=3, replace=False).tolist(),
                'efficiency_score': np.random.beta(7, 3),  # How efficient the driver is
            }
        
        return drivers
    
    def _initialize_ahmedabad_zones(self):
        """Initialize Ahmedabad-specific zone characteristics"""
        zone_chars = {}
        
        # Define Ahmedabad zone categories
        high_demand_zones = ["SG_Highway", "Prahlad_Nagar", "Satellite", "CG_Road", 
                             "Airport_Area", "Iskcon", "Bodakdev"]
        high_congestion_zones = ["SG_Highway", "Iskcon", "CG_Road", "Airport_Area", 
                                 "Navrangpura", "Satellite"]
        corporate_zones = ["SG_Highway", "Prahlad_Nagar", "Bodakdev", "Gift_City_Corridor"]
        outskirt_zones = ["Naroda", "Nikol", "Chandkheda", "South_Bopal"]
        
        for zone in config.ZONES:
            # Base delay rate varies by zone type
            if zone in high_congestion_zones:
                base_delay_rate = np.random.beta(4, 6)  # Higher delays
            elif zone in outskirt_zones:
                base_delay_rate = np.random.beta(2, 8)  # Lower delays
            else:
                base_delay_rate = np.random.beta(3, 7)
            
            # Congestion factor
            if zone in high_congestion_zones:
                congestion_factor = np.random.uniform(1.5, 2.5)
            elif zone in corporate_zones:
                congestion_factor = np.random.uniform(1.2, 1.8)
            else:
                congestion_factor = np.random.uniform(0.5, 1.5)
            
            # Demand factor
            if zone in high_demand_zones:
                demand_factor = np.random.uniform(1.5, 2.5)
            elif zone in corporate_zones:
                demand_factor = np.random.uniform(1.3, 2.0)
            else:
                demand_factor = np.random.uniform(0.5, 1.2)
            
            zone_chars[zone] = {
                'base_delay_rate': base_delay_rate,
                'congestion_factor': congestion_factor,
                'demand_factor': demand_factor
            }
        
        return zone_chars
    
    def generate(self):
        """Generate complete synthetic dataset for Ahmedabad with advanced features"""
        logger.info("Starting V2 synthetic data generation for Ahmedabad...")
        
        # Generate base ride data
        df = self._generate_ahmedabad_base_rides()
        
        # Add H3 spatial features
        df = self._add_h3_spatial_features(df)
        
        # Generate realistic outcomes with interactions
        df = self._generate_ahmedabad_realistic_outcomes(df)
        
        # Calculate all targets (multi-task)
        df = self._calculate_all_targets(df)
        
        # Sort by timestamp for time-based splits
        df = df.sort_values('booking_timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} Ahmedabad rides with advanced features")
        logger.info(f"Delay rate (>5 min threshold): {df['order_delayed'].mean():.2%}")
        
        return df
    
    def _generate_ahmedabad_base_rides(self):
        """Generate base ride information with Ahmedabad-specific patterns"""
        logger.info("Generating base ride data for Ahmedabad...")
        
        # Generate timestamps
        start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
        end = datetime.strptime(config.END_DATE, "%Y-%m-%d")
        timestamps = [start + timedelta(
            seconds=np.random.randint(0, int((end - start).total_seconds()))
        ) for _ in range(self.n_rows)]
        
        timestamps.sort()
        
        data = {
            'order_id': [f"ORD_{i:08d}" for i in range(1000000,1000000+self.n_rows)],
            'booking_timestamp': timestamps
        }
        
        df = pd.DataFrame(data)
        
        # Extract time features
        df['hour'] = df['booking_timestamp'].dt.hour
        df['day_of_week'] = df['booking_timestamp'].dt.dayofweek
        df['week_of_year'] = df['booking_timestamp'].dt.isocalendar().week
        df['month'] = df['booking_timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Ahmedabad-specific festivals
        df['festival_date'] = df['booking_timestamp'].dt.strftime('%Y-%m-%d')
        df['festival_flag'] = df['festival_date'].isin(config.AHMEDABAD_FESTIVALS).astype(int)
        df = df.drop('festival_date', axis=1)
        
        # Holiday flag
        df['holiday_flag'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        
        # Ahmedabad peak hours
        df['is_peak_hour'] = (
            (df['hour'].between(config.MORNING_PEAK_START, config.MORNING_PEAK_END)) | 
            (df['hour'].between(config.EVENING_PEAK_START, config.EVENING_PEAK_END))
        ).astype(int)

        
        # Zones and locations
        df['pickup_zone'] = np.random.choice(config.ZONES, size=len(df))
        df['drop_zone'] = np.random.choice(config.ZONES, size=len(df))
        
        # Coordinates
        df['pickup_latitude'] = df['pickup_zone'].map(
            lambda z: config.ZONE_COORDINATES.get(z, (23.03, 72.52))[0]
        ) + np.random.uniform(-0.01, 0.01, size=len(df))
        
        df['pickup_longitude'] = df['pickup_zone'].map(
            lambda z: config.ZONE_COORDINATES.get(z, (23.03, 72.52))[1]
        ) + np.random.uniform(-0.01, 0.01, size=len(df))
        
        df['drop_latitude'] = df['drop_zone'].map(
            lambda z: config.ZONE_COORDINATES.get(z, (23.04, 72.54))[0]
        ) + np.random.uniform(-0.01, 0.01, size=len(df))
        
        df['drop_longitude'] = df['drop_zone'].map(
            lambda z: config.ZONE_COORDINATES.get(z, (23.04, 72.54))[1]
        ) + np.random.uniform(-0.01, 0.01, size=len(df))
        
        # Distances
        df['driver_to_pickup_distance_km'] = np.random.gamma(2, 1.5, size=len(df))
        df['trip_distance_km'] = np.random.gamma(3, 2.5, size=len(df))
        
        # Driver assignment
        driver_ids = list(self.drivers.keys())
        df['driver_id'] = np.random.choice(driver_ids, size=len(df))
        
        # Driver features (no driver_rating — only driver_base_rating)
        df['driver_base_rating'] = df['driver_id'].map(lambda x: self.drivers[x]['base_rating'])
        df['driver_experience_days'] = df['driver_id'].map(lambda x: self.drivers[x]['experience_days'])
        df['driver_acceptance_rate'] = df['driver_id'].map(lambda x: self.drivers[x]['acceptance_rate'])
        df['driver_cancellation_rate'] = df['driver_id'].map(lambda x: self.drivers[x]['cancellation_rate'])
        df['driver_total_rides'] = df['driver_experience_days'] * np.random.uniform(3, 8, size=len(df))
        
        # Traffic
        # df['traffic_level'] = 'Low'
        # morning_mask = df['hour'].between(config.MORNING_PEAK_START, config.MORNING_PEAK_END)
        # df.loc[morning_mask, 'traffic_level'] = np.random.choice(
        #     ['Medium', 'High', 'Very_High'], size=morning_mask.sum(), p=[0.3, 0.5, 0.2]
        # )
        # evening_mask = df['hour'].between(config.EVENING_PEAK_START, config.EVENING_PEAK_END)
        # df.loc[evening_mask, 'traffic_level'] = np.random.choice(
        #     ['High', 'Very_High'], size=evening_mask.sum(), p=[0.4, 0.6]
        # )
        
        # df['congestion_score'] = df['traffic_level'].map({
        #     'Low': 20, 'Medium': 50, 'High': 75, 'Very_High': 95
        # })
        
        
        # ETA
        df['google_pickup_eta_minutes'] = df['driver_to_pickup_distance_km'] * np.random.uniform(2.5, 4.5, size=len(df))
        df['google_trip_eta_minutes'] = df['trip_distance_km'] * np.random.uniform(2.0, 4.0, size=len(df))
        
        # Time of day
        df['time_of_day'] = 'Night'
        df.loc[df['hour'].between(5, 11), 'time_of_day'] = 'Morning'
        df.loc[df['hour'].between(12, 16), 'time_of_day'] = 'Afternoon'
        df.loc[df['hour'].between(17, 20), 'time_of_day'] = 'Evening'
        
        logger.info(f"Generated base data for {len(df)} rides")
        return df
    
    def _add_h3_spatial_features(self, df):
        """Add H3 hexagon IDs for pickup and drop locations"""
        logger.info("Adding H3 spatial features...")
        
        try:
            # Try new h3 API (v4+)
            df['pickup_h3_cell'] = df.apply(
                lambda row: h3.latlng_to_cell(row['pickup_latitude'], row['pickup_longitude'], H3_RESOLUTION),
                axis=1
            )
            df['drop_h3_cell'] = df.apply(
                lambda row: h3.latlng_to_cell(row['drop_latitude'], row['drop_longitude'], H3_RESOLUTION),
                axis=1
            )
        except AttributeError:
            # Fallback to old h3 API (v3)
            df['pickup_h3_cell'] = df.apply(
                lambda row: h3.geo_to_h3(row['pickup_latitude'], row['pickup_longitude'], H3_RESOLUTION),
                axis=1
            )
            df['drop_h3_cell'] = df.apply(
                lambda row: h3.geo_to_h3(row['drop_latitude'], row['drop_longitude'], H3_RESOLUTION),
                axis=1
            )
        
        logger.info("Added H3 spatial features")
        return df

    
    def _generate_ahmedabad_realistic_outcomes(self, df):
        """Generate realistic pickup and trip times with proper delay logic"""
        logger.info("Generating realistic outcomes with balanced delays...")
        
        actual_pickup_times = []
        actual_trip_times = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rides"):
            expected_pickup = row['google_pickup_eta_minutes']
            expected_trip = row['google_trip_eta_minutes']
            
            pickup_delay_factor = 0.0
            trip_delay_factor = 0.0
            
            # Driver characteristics
            driver = self.drivers[row['driver_id']]
            driver_speed_factor = driver['base_speed_factor']
            driver_reliability = driver['reliability_factor']
            
            # IMPORTANT: Good drivers (reliability > 0.7) should usually be on time
            if driver_reliability > 0.7:
                # Good driver - mostly on time or early
                pickup_delay_factor = np.random.uniform(-0.2, 0.1)  # Often early!
                trip_delay_factor = np.random.uniform(-0.15, 0.1)
            elif driver_reliability > 0.5:
                # Average driver - slight delays
                pickup_delay_factor = np.random.uniform(-0.1, 0.2)
                trip_delay_factor = np.random.uniform(-0.05, 0.15)
            else:
                # Poor driver - more delays
                pickup_delay_factor = np.random.uniform(0.1, 0.4)
                trip_delay_factor = np.random.uniform(0.1, 0.3)
            
            # # Traffic impact (but not extreme)
            # if row['traffic_level'] == 'Very_High':
            #     trip_delay_factor += np.random.uniform(0.2, 0.4)
            # elif row['traffic_level'] == 'High':
            #     trip_delay_factor += np.random.uniform(0.1, 0.25)
            # elif row['traffic_level'] == 'Low':
            #     trip_delay_factor -= np.random.uniform(0.05, 0.15)  # Faster in low traffic
            
            # Peak hour (slight impact)
            if row['is_peak_hour'] == 1:
                pickup_delay_factor += np.random.uniform(0.05, 0.15)
                trip_delay_factor += np.random.uniform(0.05, 0.15)
            
            # Festival impact (moderate)
            if row['festival_flag'] == 1:
                pickup_delay_factor += np.random.uniform(0.1, 0.25)
                trip_delay_factor += np.random.uniform(0.15, 0.3)
            
            # Distance impact (longer trips have more variance)
            if row['trip_distance_km'] > 15:
                trip_delay_factor += np.random.uniform(0, 0.15)
            
            # Calculate actual times
            actual_pickup = expected_pickup * (1 + pickup_delay_factor)
            actual_trip = expected_trip * (1 + trip_delay_factor)
            
            # Add some randomness to make it realistic
            actual_pickup = max(1.0, actual_pickup + np.random.normal(0, 1.5))
            actual_trip = max(2.0, actual_trip + np.random.normal(0, 2.0))
            
            actual_pickup_times.append(actual_pickup)
            actual_trip_times.append(actual_trip)
        
        df['actual_pickup_time_minutes'] = actual_pickup_times
        df['actual_trip_time_minutes'] = actual_trip_times
        df['actual_total_time_minutes'] = df['actual_pickup_time_minutes'] + df['actual_trip_time_minutes']
        
        # Calculate delays
        df['pickup_delay_minutes'] = df['actual_pickup_time_minutes'] - df['google_pickup_eta_minutes']
        df['trip_delay_minutes'] = df['actual_trip_time_minutes'] - df['google_trip_eta_minutes']
        df['total_delay_minutes'] = df['pickup_delay_minutes'] + df['trip_delay_minutes']
        
        logger.info(f"Generated outcomes - Avg total delay: {df['total_delay_minutes'].mean():.2f} min")
        return df
    
    def _calculate_all_targets(self, df):
        """Calculate all target variables for multi-task learning"""
        logger.info("Calculating all targets...")
        
        # Target 1: Binary classification (order_delayed) - CRITICAL: threshold is 5 minutes
        df['order_delayed'] = (df['total_delay_minutes'] > 5).astype(int)
        
        # Target 2: ETA regression (actual total time)
        # df['actual_eta_minutes'] = df['actual_total_time_minutes']
        
        # Target 3: Delay severity (categorical)
        df['delay_severity'] = "0"
        df.loc[(df['total_delay_minutes'] > 0) & (df['total_delay_minutes'] <= 10), 'delay_severity'] = "1"
        df.loc[(df['total_delay_minutes'] > 10) & (df['total_delay_minutes'] <= 20), 'delay_severity'] = "2"
        df.loc[df['total_delay_minutes'] > 20, 'delay_severity'] = "3"
        
        # Log statistics
        delay_rate = df['order_delayed'].mean()
        logger.info(f"Delay rate (>5 min): {delay_rate:.2%}")
        logger.info(f"Delay severity distribution:\n{df['delay_severity'].value_counts(normalize=True)}")
        
        return df
    
    


def main():
    """Generate V2 synthetic data for Ahmedabad"""
    print("="*80)
    print("🚗 AHMEDABAD RIDE DATA GENERATOR V2")
    print("="*80)
    print(f"\nGenerating {config.N_ROWS} synthetic ride orders...")
    print(f"Location: Ahmedabad, Gujarat, India")
    print(f"Date range: {config.START_DATE} to {config.END_DATE}")
    print(f"Advanced features: H3 spatial, driver sequences")
    print("="*80 + "\n")
    
    # Generate data
    generator = AhmedabadRideDataGeneratorV2(n_rows=config.N_ROWS, random_seed=config.RANDOM_SEED)
    df = generator.generate()
    
    # Save to CSV
    output_path = config.DATA_DIR / "ride_orders_ahmedabad_v24.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total orders: {len(df):,}")
    print(f"   Date range: {df['booking_timestamp'].min()} to {df['booking_timestamp'].max()}")
    print(f"   Delay rate (>5 min): {df['order_delayed'].mean():.2%}")
    print(f"   Average delay: {df['total_delay_minutes'].mean():.2f} minutes")
    print(f"   \n   Delay severity:")
    print(f"      None: {(df['delay_severity'] == 0).sum():,} ({(df['delay_severity'] == 0).mean():.1%})")
    print(f"      Minor: {(df['delay_severity'] == 1).sum():,} ({(df['delay_severity'] == 1).mean():.1%})")
    print(f"      Moderate: {(df['delay_severity'] == 2).sum():,} ({(df['delay_severity'] == 2).mean():.1%})")
    print(f"      Severe: {(df['delay_severity'] == 3).sum():,} ({(df['delay_severity'] == 3).mean():.1%})")
    print(f"\n📁 Saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
