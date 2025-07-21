#!/usr/bin/env python3
"""
Inventory Management and Forecasting System

This script loads inventory data, performs demand forecasting using Prophet,
and generates an interactive HTML dashboard with Plotly visualizations
including green/yellow/red indicators for inventory status.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
import os
from datetime import datetime, timedelta

# Suppress Prophet warnings
warnings.filterwarnings('ignore')

class InventoryForecaster:
    def __init__(self, data_path='data/sample_inventory.csv'):
        """Initialize the inventory forecaster with data path."""
        self.data_path = data_path
        self.df = None
        self.forecasts = {}
        
    def load_data(self):
        """Load inventory data from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"Loaded {len(self.df)} records from {self.data_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_inventory_status(self, current_stock, reorder_point, max_stock):
        """Determine inventory status and color coding."""
        if current_stock <= reorder_point:
            return 'Critical', 'red'
        elif current_stock <= reorder_point * 1.5:
            return 'Low', 'orange'
        elif current_stock >= max_stock * 0.9:
            return 'High', 'blue'
        else:
            return 'Good', 'green'
    
    def forecast_demand(self, product_id, periods=30):
        """Forecast demand for a specific product using Prophet."""
        try:
            # Filter data for the specific product
            product_data = self.df[self.df['product_id'] == product_id].copy()
            
            if len(product_data) < 2:
                print(f"Not enough data for {product_id}")
                return None
            
            # Prepare data for Prophet
            prophet_data = product_data[['date', 'daily_demand']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Create and fit the model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            model.fit(prophet_data)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Store forecast results
            self.forecasts[product_id] = {
                'historical': prophet_data,
                'forecast': forecast,
                'model': model
            }
            
            return forecast
            
        except Exception as e:
            print(f"Error forecasting demand for {product_id}: {e}")
            return None
    
    def create_dashboard(self):
        """Create the main dashboard with multiple visualizations."""
        if self.df is None:
            print("No data loaded. Please run load_data() first.")
            return None
        
        # Get latest data for each product
        latest_data = self.df.groupby('product_id').last().reset_index()
        
        # Add status information
        latest_data['status'], latest_data['color'] = zip(*latest_data.apply(
            lambda row: self.get_inventory_status(
                row['current_stock'], 
                row['reorder_point'], 
                row['max_stock']
            ), axis=1
        ))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Current Inventory Status',
                'Stock Levels by Product',
                'Daily Demand Trends',
                'Reorder Analysis',
                'Inventory Forecast (Product A)',
                'Stock Status Distribution'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"colspan": 2}, None],
                [{"type": "scatter"}, {"type": "pie"}]
            ]
        )
        
        # 1. Inventory Status Indicator
        critical_count = len(latest_data[latest_data['status'] == 'Critical'])
        low_count = len(latest_data[latest_data['status'] == 'Low'])
        total_products = len(latest_data)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=((total_products - critical_count - low_count) / total_products) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Inventory Health %"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Stock Levels Bar Chart
        fig.add_trace(
            go.Bar(
                x=latest_data['product_name'],
                y=latest_data['current_stock'],
                marker_color=latest_data['color'],
                name='Current Stock',
                text=latest_data['status'],
                textposition='auto',
            ),
            row=1, col=2
        )
        
        # Add reorder points as a line
        fig.add_trace(
            go.Scatter(
                x=latest_data['product_name'],
                y=latest_data['reorder_point'],
                mode='markers+lines',
                name='Reorder Point',
                line=dict(color='red', dash='dash'),
                marker=dict(color='red', size=8)
            ),
            row=1, col=2
        )
        
        # 3. Daily Demand Trends
        for product in self.df['product_id'].unique()[:3]:  # Show top 3 products
            product_data = self.df[self.df['product_id'] == product]
            fig.add_trace(
                go.Scatter(
                    x=product_data['date'],
                    y=product_data['daily_demand'],
                    mode='lines+markers',
                    name=f'Demand - {product}',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # 4. Forecast for Product A (if available)
        if 'P001' in self.forecasts:
            forecast_data = self.forecasts['P001']['forecast']
            historical_data = self.forecasts['P001']['historical']
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_data['ds'],
                    y=historical_data['y'],
                    mode='markers',
                    name='Historical Demand',
                    marker=dict(color='blue', size=6)
                ),
                row=3, col=1
            )
            
            # Forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2)
                ),
                row=3, col=1
            )
            
            # Confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
                    y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 5. Status Distribution Pie Chart
        status_counts = latest_data['status'].value_counts()
        colors = ['green', 'orange', 'red', 'blue']
        
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                marker=dict(colors=colors[:len(status_counts)]),
                name="Status Distribution"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Inventory Management Dashboard",
            title_x=0.5,
            font=dict(size=12)
        )
        
        return fig
    
    def generate_html_report(self, output_path='templates/dashboard.html'):
        """Generate HTML dashboard report."""
        # Run forecasts for all products
        for product_id in self.df['product_id'].unique():
            self.forecast_demand(product_id)
        
        # Create dashboard
        fig = self.create_dashboard()
        
        if fig is None:
            return False
        
        # Get latest data and add status for summary cards
        latest_data = self.df.groupby('product_id').last().reset_index()
        latest_data['status'], latest_data['color'] = zip(*latest_data.apply(
            lambda row: self.get_inventory_status(
                row['current_stock'], 
                row['reorder_point'], 
                row['max_stock']
            ), axis=1
        ))
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Forecast Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-cards {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px;
            min-width: 200px;
            text-align: center;
        }}
        .card h3 {{ margin-top: 0; }}
        .status-critical {{ border-left: 5px solid #e74c3c; }}
        .status-low {{ border-left: 5px solid #f39c12; }}
        .status-good {{ border-left: 5px solid #27ae60; }}
        .status-high {{ border-left: 5px solid #3498db; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Inventory Forecast Dashboard</h1>
        <p>Real-time inventory management and demand forecasting</p>
        <small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
    
    <div class="summary-cards">
"""
        
        # Add summary cards
        status_counts = latest_data.groupby(['status']).size().to_dict()
        
        for status, count in status_counts.items():
            status_class = f"status-{status.lower()}"
            html_content += f"""
        <div class="card {status_class}">
            <h3>{count}</h3>
            <p>{status} Products</p>
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="chart-container">
        <div id="dashboard-chart"></div>
    </div>
    
    <div class="footer">
        <p>🔄 Dashboard updates automatically when inventory_forecast.py is run</p>
        <p>📈 Powered by Prophet forecasting and Plotly visualizations</p>
    </div>
    
    <script>
"""
        
        # Add Plotly chart
        html_content += f"var plotlyData = {fig.to_json()};\n"
        html_content += """
        Plotly.newPlot('dashboard-chart', plotlyData.data, plotlyData.layout, {responsive: true});
    </script>
</body>
</html>
"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Dashboard generated successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return False


def main():
    """Main function to run the inventory forecasting system."""
    print("🚀 Starting Inventory Forecasting System...")
    
    # Initialize forecaster
    forecaster = InventoryForecaster()
    
    # Load data
    if not forecaster.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    print("📊 Data loaded successfully!")
    print(f"   - {len(forecaster.df)} total records")
    print(f"   - {forecaster.df['product_id'].nunique()} unique products")
    print(f"   - Date range: {forecaster.df['date'].min()} to {forecaster.df['date'].max()}")
    
    # Generate forecasts and dashboard
    print("\n🔮 Generating demand forecasts...")
    
    # Generate HTML report
    print("📄 Creating HTML dashboard...")
    if forecaster.generate_html_report():
        print("✅ Dashboard created successfully!")
        print("🌐 Open templates/dashboard.html in your browser to view the dashboard")
    else:
        print("❌ Failed to create dashboard")


if __name__ == "__main__":
    main()