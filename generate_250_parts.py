#!/usr/bin/env python3
"""
Generate sample inventory data for 250 parts with realistic patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_part_data():
    """Generate characteristics for 250 different parts"""
    parts = []
    
    # Part categories with different characteristics
    categories = {
        'Fast_Moving': {'count': 50, 'demand_range': (20, 50), 'stock_multiplier': 1.5},
        'Medium_Moving': {'count': 100, 'demand_range': (8, 25), 'stock_multiplier': 2.0},
        'Slow_Moving': {'count': 75, 'demand_range': (1, 10), 'stock_multiplier': 3.0},
        'Seasonal': {'count': 25, 'demand_range': (5, 30), 'stock_multiplier': 2.5}
    }
    
    # Part name prefixes for variety
    prefixes = ['Widget', 'Component', 'Assembly', 'Module', 'Unit', 'Element', 'Part', 'Device']
    suffixes = ['Pro', 'Max', 'Elite', 'Standard', 'Basic', 'Plus', 'Ultra', 'Prime']
    
    part_id = 1
    
    for category, config in categories.items():
        for i in range(config['count']):
            # Generate part identifiers
            product_id = f"P{part_id:03d}"
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes) if random.random() > 0.3 else ""
            product_name = f"{prefix} {chr(65 + (part_id % 26))}{part_id}" + (f" {suffix}" if suffix else "")
            
            # Generate demand characteristics
            base_demand = random.uniform(*config['demand_range'])
            
            # Generate stock parameters
            reorder_point = int(base_demand * random.uniform(5, 15))  # 5-15 days of stock
            max_stock = int(reorder_point * config['stock_multiplier'] * random.uniform(3, 6))
            initial_stock = random.randint(reorder_point, max_stock)
            
            # Lead time varies by supplier
            lead_time = random.choice([3, 5, 7, 10, 14, 21])
            
            parts.append({
                'product_id': product_id,
                'product_name': product_name,
                'category': category,
                'base_demand': base_demand,
                'reorder_point': reorder_point,
                'max_stock': max_stock,
                'initial_stock': initial_stock,
                'supplier_lead_time': lead_time
            })
            
            part_id += 1
    
    return parts

def generate_daily_demand(base_demand, category, day_of_year, day_of_week):
    """Generate realistic daily demand with patterns"""
    demand = base_demand
    
    # Add day-of-week patterns
    weekday_factors = {
        0: 1.1,  # Monday - higher
        1: 1.0,  # Tuesday - normal
        2: 1.0,  # Wednesday - normal
        3: 1.0,  # Thursday - normal
        4: 0.9,  # Friday - lower
        5: 0.7,  # Saturday - much lower
        6: 0.6   # Sunday - lowest
    }
    demand *= weekday_factors[day_of_week]
    
    # Add seasonal patterns for seasonal items
    if category == 'Seasonal':
        # Peak in summer (days 150-250), low in winter
        seasonal_factor = 0.5 + 0.8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        demand *= seasonal_factor
    
    # Add random variation (±30%)
    variation = np.random.normal(1.0, 0.3)
    demand *= max(0.1, variation)  # Ensure demand doesn't go negative
    
    # Round to reasonable precision
    return max(0, round(demand, 1))

def generate_inventory_data():
    """Generate complete inventory dataset"""
    parts = generate_part_data()
    
    # Date range: 3 months of daily data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    all_data = []
    
    for part in parts:
        current_stock = part['initial_stock']
        
        for date in date_range:
            day_of_year = date.timetuple().tm_yday
            day_of_week = date.weekday()
            
            # Generate daily demand
            daily_demand = generate_daily_demand(
                part['base_demand'], 
                part['category'], 
                day_of_year, 
                day_of_week
            )
            
            # Update stock (simple model: stock decreases by demand)
            current_stock -= daily_demand
            
            # Simulate restocking when below reorder point
            if current_stock <= part['reorder_point']:
                # Restock to 80-95% of max capacity
                restock_amount = random.uniform(0.8, 0.95) * part['max_stock'] - current_stock
                current_stock += max(0, restock_amount)
            
            # Ensure stock doesn't go negative
            current_stock = max(0, current_stock)
            
            all_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_id': part['product_id'],
                'product_name': part['product_name'],
                'current_stock': int(current_stock),
                'daily_demand': daily_demand,
                'reorder_point': part['reorder_point'],
                'max_stock': part['max_stock'],
                'supplier_lead_time': part['supplier_lead_time']
            })
    
    return pd.DataFrame(all_data)

def main():
    """Generate and save the inventory data"""
    print("Generating inventory data for 250 parts...")
    
    df = generate_inventory_data()
    
    print(f"Generated {len(df):,} records for {df['product_id'].nunique()} unique parts")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save to CSV
    output_file = 'data/sample_inventory.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    # Show summary statistics
    print("\nSummary by category:")
    part_categories = df.groupby('product_id').first()['product_name'].str.extract(r'(Fast_Moving|Medium_Moving|Slow_Moving|Seasonal)', expand=False)
    
    print("\nSample of generated data:")
    print(df.head(10))
    
    print(f"\nFile size: {len(df):,} rows")
    print("Data generation complete!")

if __name__ == "__main__":
    main()
