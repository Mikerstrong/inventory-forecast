#!/usr/bin/env python3
"""
Inventory Management and Forecasting System

This script loads inventory data, performs demand forecasting using Prophet,
and generates an interactive Plotly HTML dashboard with stock level indicators.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

# Suppress Prophet warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

class InventoryForecastSystem:
    def __init__(self, data_file='data/sample_inventory.csv'):
        """Initialize the inventory forecast system."""
        self.data_file = data_file
        self.inventory_data = None
        self.forecast_data = {}
        
    def load_inventory_data(self):
        """Load and prepare inventory data from CSV file."""
        try:
            self.inventory_data = pd.read_csv(self.data_file)
            self.inventory_data['date'] = pd.to_datetime(self.inventory_data['date'])
            print(f"✓ Loaded inventory data: {len(self.inventory_data)} records")
            print(f"✓ Products: {self.inventory_data['product_name'].nunique()}")
            print(f"✓ Date range: {self.inventory_data['date'].min()} to {self.inventory_data['date'].max()}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def forecast_demand(self, product_id, days_ahead=30):
        """Forecast demand for a specific product using Prophet."""
        try:
            # Filter data for the specific product
            product_data = self.inventory_data[
                self.inventory_data['product_id'] == product_id
            ][['date', 'demand']].copy()
            
            if len(product_data) < 3:
                print(f"✗ Not enough data for {product_id}")
                return None
                
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            product_data = product_data.rename(columns={'date': 'ds', 'demand': 'y'})
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.80
            )
            
            model.fit(product_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Store forecast data
            self.forecast_data[product_id] = {
                'historical': product_data,
                'forecast': forecast,
                'model': model
            }
            
            print(f"✓ Forecasted demand for {product_id}")
            return forecast
            
        except Exception as e:
            print(f"✗ Error forecasting {product_id}: {e}")
            return None
    
    def calculate_stock_status(self, product_id):
        """Calculate stock status with color indicators."""
        try:
            product_info = self.inventory_data[
                self.inventory_data['product_id'] == product_id
            ].iloc[-1]  # Get latest record
            
            current_stock = product_info['current_stock']
            reorder_point = product_info['reorder_point']
            lead_time_days = product_info['lead_time_days']
            
            # Get forecasted demand for lead time period
            if product_id in self.forecast_data:
                forecast = self.forecast_data[product_id]['forecast']
                future_demand = forecast[forecast['ds'] > product_info['date']]['yhat']
                lead_time_demand = future_demand.head(lead_time_days).sum() if len(future_demand) > 0 else 0
            else:
                # Fallback to average demand if no forecast
                avg_demand = self.inventory_data[
                    self.inventory_data['product_id'] == product_id
                ]['demand'].mean()
                lead_time_demand = avg_demand * lead_time_days
            
            # Determine status
            if current_stock <= reorder_point:
                status = 'Critical'
                color = 'red'
            elif current_stock <= (reorder_point + lead_time_demand):
                status = 'Warning'
                color = 'orange'
            else:
                status = 'Good'
                color = 'green'
            
            return {
                'status': status,
                'color': color,
                'current_stock': current_stock,
                'reorder_point': reorder_point,
                'lead_time_demand': lead_time_demand,
                'product_name': product_info['product_name']
            }
            
        except Exception as e:
            print(f"✗ Error calculating status for {product_id}: {e}")
            return None
    
    def generate_dashboard(self):
        """Generate interactive Plotly HTML dashboard."""
        try:
            if self.inventory_data is None:
                print("✗ No inventory data loaded")
                return False
            
            # Get unique products
            products = self.inventory_data['product_id'].unique()
            
            # Forecast for each product
            for product in products:
                self.forecast_demand(product)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Demand Forecast', 'Current Stock Levels', 
                               'Stock Status Overview', 'Forecast Accuracy'),
                specs=[[{'secondary_y': True}, {'type': 'bar'}],
                       [{'type': 'indicator'}, {'type': 'scatter'}]]
            )
            
            # Color mapping for status
            colors = {'Good': 'green', 'Warning': 'orange', 'Critical': 'red'}
            
            # Plot 1: Demand forecast for all products
            row, col = 1, 1
            for i, product in enumerate(products):
                if product in self.forecast_data:
                    forecast = self.forecast_data[product]['forecast']
                    historical = self.forecast_data[product]['historical']
                    
                    # Historical data
                    fig.add_trace(
                        go.Scatter(
                            x=historical['ds'],
                            y=historical['y'],
                            mode='lines+markers',
                            name=f'{product} Historical',
                            line=dict(width=2),
                            showlegend=True
                        ),
                        row=row, col=col
                    )
                    
                    # Forecast data
                    future_forecast = forecast[forecast['ds'] > historical['ds'].max()]
                    fig.add_trace(
                        go.Scatter(
                            x=future_forecast['ds'],
                            y=future_forecast['yhat'],
                            mode='lines',
                            name=f'{product} Forecast',
                            line=dict(dash='dash', width=2),
                            showlegend=True
                        ),
                        row=row, col=col
                    )
            
            # Plot 2: Current stock levels
            row, col = 1, 2
            stock_data = []
            status_data = []
            
            for product in products:
                status = self.calculate_stock_status(product)
                if status:
                    stock_data.append({
                        'product': status['product_name'],
                        'current_stock': status['current_stock'],
                        'reorder_point': status['reorder_point'],
                        'color': colors[status['status']],
                        'status': status['status']
                    })
                    status_data.append(status)
            
            if stock_data:
                products_names = [item['product'] for item in stock_data]
                current_stocks = [item['current_stock'] for item in stock_data]
                reorder_points = [item['reorder_point'] for item in stock_data]
                bar_colors = [item['color'] for item in stock_data]
                
                fig.add_trace(
                    go.Bar(
                        x=products_names,
                        y=current_stocks,
                        name='Current Stock',
                        marker_color=bar_colors,
                        showlegend=True
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=products_names,
                        y=reorder_points,
                        mode='markers',
                        name='Reorder Point',
                        marker=dict(symbol='diamond', size=10, color='black'),
                        showlegend=True
                    ),
                    row=row, col=col
                )
            
            # Plot 3: Status indicators
            row, col = 2, 1
            if status_data:
                critical_count = sum(1 for s in status_data if s['status'] == 'Critical')
                warning_count = sum(1 for s in status_data if s['status'] == 'Warning')
                good_count = sum(1 for s in status_data if s['status'] == 'Good')
                
                fig.add_trace(
                    go.Indicator(
                        mode="number+gauge",
                        value=critical_count,
                        title={"text": "Critical Items"},
                        gauge={'axis': {'range': [None, len(products)]},
                               'bar': {'color': "red"},
                               'steps': [{'range': [0, len(products)], 'color': "lightgray"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': len(products) * 0.3}},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ),
                    row=row, col=col
                )
            
            # Plot 4: Simple forecast accuracy (placeholder)
            row, col = 2, 2
            accuracy_data = [95, 87, 92]  # Placeholder accuracy percentages
            product_names = [f"Product {i+1}" for i in range(len(accuracy_data))]
            
            fig.add_trace(
                go.Scatter(
                    x=product_names,
                    y=accuracy_data,
                    mode='markers+lines',
                    name='Forecast Accuracy %',
                    marker=dict(size=12, color='blue'),
                    line=dict(color='blue', width=2),
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Inventory Management Dashboard',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24}
                },
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Save as HTML
            output_path = 'templates/dashboard.html'
            fig.write_html(output_path, include_plotlyjs='cdn')
            
            print(f"✓ Dashboard generated successfully: {output_path}")
            
            # Generate summary report
            self.print_summary_report(status_data)
            
            return True
            
        except Exception as e:
            print(f"✗ Error generating dashboard: {e}")
            return False
    
    def print_summary_report(self, status_data):
        """Print a summary report of inventory status."""
        print("\n" + "="*50)
        print("INVENTORY STATUS SUMMARY")
        print("="*50)
        
        if not status_data:
            print("No status data available")
            return
        
        critical_items = [s for s in status_data if s['status'] == 'Critical']
        warning_items = [s for s in status_data if s['status'] == 'Warning']
        good_items = [s for s in status_data if s['status'] == 'Good']
        
        print(f"🔴 Critical Items: {len(critical_items)}")
        for item in critical_items:
            print(f"   - {item['product_name']}: {item['current_stock']} units")
        
        print(f"\n🟡 Warning Items: {len(warning_items)}")
        for item in warning_items:
            print(f"   - {item['product_name']}: {item['current_stock']} units")
        
        print(f"\n🟢 Good Items: {len(good_items)}")
        for item in good_items:
            print(f"   - {item['product_name']}: {item['current_stock']} units")
        
        print("\n" + "="*50)

def main():
    """Main function to run the inventory forecast system."""
    print("🚀 Starting Inventory Forecast System")
    print("-" * 40)
    
    # Initialize system
    system = InventoryForecastSystem()
    
    # Load data
    if not system.load_inventory_data():
        return
    
    # Generate dashboard
    if system.generate_dashboard():
        print("\n✅ Inventory forecast system completed successfully!")
        print("📊 Open templates/dashboard.html in your browser to view the dashboard")
    else:
        print("\n❌ Failed to generate dashboard")

if __name__ == "__main__":
    main()