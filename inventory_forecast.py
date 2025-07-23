#!/usr/bin/env python3
"""
Inventory Management and Forecasting System - Streamlit Web App

This Streamlit application provides an interactive inventory management dashboard
with demand forecasting using Prophet and real-time visualizations with Plotly.
Run with: streamlit run inventory_forecast.py
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import streamlit as st
import warnings
import os
from datetime import datetime, timedelta
import numpy as np

# Suppress Prophet warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Inventory Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
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
            st.warning(f"Error forecasting demand for {product_id}: {e}")
            return None
    
    def get_latest_data(self):
        """Get the latest data for each product with status."""
        if self.df is None:
            return None
            
        latest_data = self.df.groupby('product_id').last().reset_index()
        
        # Add status information
        latest_data['status'], latest_data['color'] = zip(*latest_data.apply(
            lambda row: self.get_inventory_status(
                row['current_stock'], 
                row['reorder_point'], 
                row['max_stock']
            ), axis=1
        ))
        
        return latest_data

def create_inventory_health_gauge(latest_data):
    """Create inventory health gauge."""
    critical_count = len(latest_data[latest_data['status'] == 'Critical'])
    low_count = len(latest_data[latest_data['status'] == 'Low'])
    total_products = len(latest_data)
    
    health_percentage = ((total_products - critical_count - low_count) / total_products) * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_percentage,
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
    ))
    
    fig.update_layout(height=300)
    return fig

def create_stock_levels_chart(latest_data):
    """Create stock levels bar chart."""
    fig = go.Figure()
    
    # Current stock bars
    fig.add_trace(go.Bar(
        x=latest_data['product_name'],
        y=latest_data['current_stock'],
        marker_color=latest_data['color'],
        name='Current Stock',
        text=latest_data['status'],
        textposition='auto',
    ))
    
    # Add reorder points as a line
    fig.add_trace(go.Scatter(
        x=latest_data['product_name'],
        y=latest_data['reorder_point'],
        mode='markers+lines',
        name='Reorder Point',
        line=dict(color='red', dash='dash'),
        marker=dict(color='red', size=8)
    ))
    
    fig.update_layout(
        title="Stock Levels by Product",
        xaxis_title="Product",
        yaxis_title="Stock Level",
        height=400
    )
    
    return fig

def create_demand_trends_chart(df, selected_products):
    """Create daily demand trends chart."""
    fig = go.Figure()
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, product in enumerate(selected_products):
        product_data = df[df['product_id'] == product]
        fig.add_trace(go.Scatter(
            x=product_data['date'],
            y=product_data['daily_demand'],
            mode='lines+markers',
            name=f'Demand - {product}',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Daily Demand Trends",
        xaxis_title="Date",
        yaxis_title="Daily Demand",
        height=400
    )
    
    return fig

def create_forecast_chart(forecaster, selected_product):
    """Create forecast chart for selected product."""
    if selected_product not in forecaster.forecasts:
        forecaster.forecast_demand(selected_product)
    
    if selected_product not in forecaster.forecasts:
        return None
    
    forecast_data = forecaster.forecasts[selected_product]['forecast']
    historical_data = forecaster.forecasts[selected_product]['historical']
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['y'],
        mode='markers',
        name='Historical Demand',
        marker=dict(color='blue', size=6)
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
        y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Demand Forecast - {selected_product}",
        xaxis_title="Date",
        yaxis_title="Demand",
        height=400
    )
    
    return fig

def create_status_distribution_chart(latest_data):
    """Create status distribution pie chart."""
    status_counts = latest_data['status'].value_counts()
    colors = ['green', 'orange', 'red', 'blue']
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        marker=dict(colors=colors[:len(status_counts)]),
    )])
    
    fig.update_layout(
        title="Stock Status Distribution",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application."""
    # Header
    st.title("📊 Inventory Management & Forecasting Dashboard")
    st.markdown("Real-time inventory management and demand forecasting powered by Prophet")
    
    # Initialize forecaster
    forecaster = InventoryForecaster()
    
    # Load data
    if not forecaster.load_data():
        st.error("Failed to load data. Please check if 'data/sample_inventory.csv' exists.")
        st.stop()
    
    # Get latest data
    latest_data = forecaster.get_latest_data()
    
    if latest_data is None:
        st.error("No data available.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📋 Dashboard Controls")
    
    # Data summary in sidebar
    st.sidebar.subheader("📈 Data Summary")
    st.sidebar.metric("Total Records", len(forecaster.df))
    st.sidebar.metric("Unique Products", forecaster.df['product_id'].nunique())
    st.sidebar.metric("Date Range", f"{forecaster.df['date'].min().strftime('%Y-%m-%d')} to {forecaster.df['date'].max().strftime('%Y-%m-%d')}")
    
    # Product selection
    st.sidebar.subheader("🎯 Product Selection")
    all_products = forecaster.df['product_id'].unique()
    selected_products_trends = st.sidebar.multiselect(
        "Select products for trend analysis",
        all_products,
        default=all_products[:3]
    )
    
    selected_product_forecast = st.sidebar.selectbox(
        "Select product for forecasting",
        all_products,
        index=0
    )
    
    forecast_periods = st.sidebar.slider(
        "Forecast periods (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    # Status summary cards
    st.subheader("📊 Inventory Status Summary")
    
    status_counts = latest_data.groupby('status').size()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical_count = status_counts.get('Critical', 0)
        st.metric(
            label="🔴 Critical", 
            value=critical_count,
            delta=f"{critical_count} products need immediate attention" if critical_count > 0 else "All good!"
        )
    
    with col2:
        low_count = status_counts.get('Low', 0)
        st.metric(
            label="🟠 Low Stock", 
            value=low_count,
            delta=f"{low_count} products running low" if low_count > 0 else "All good!"
        )
    
    with col3:
        good_count = status_counts.get('Good', 0)
        st.metric(
            label="🟢 Good", 
            value=good_count,
            delta=f"{good_count} products in healthy range"
        )
    
    with col4:
        high_count = status_counts.get('High', 0)
        st.metric(
            label="🔵 High Stock", 
            value=high_count,
            delta=f"{high_count} products overstocked" if high_count > 0 else "No overstock"
        )
    
    # Main dashboard
    st.subheader("📈 Dashboard Overview")
    
    # Row 1: Health gauge and status distribution
    col1, col2 = st.columns(2)
    
    with col1:
        health_fig = create_inventory_health_gauge(latest_data)
        st.plotly_chart(health_fig, use_container_width=True)
    
    with col2:
        status_fig = create_status_distribution_chart(latest_data)
        st.plotly_chart(status_fig, use_container_width=True)
    
    # Row 2: Stock levels
    st.subheader("📦 Current Stock Levels")
    stock_fig = create_stock_levels_chart(latest_data)
    st.plotly_chart(stock_fig, use_container_width=True)
    
    # Row 3: Demand trends
    if selected_products_trends:
        st.subheader("📊 Daily Demand Trends")
        trends_fig = create_demand_trends_chart(forecaster.df, selected_products_trends)
        st.plotly_chart(trends_fig, use_container_width=True)
    
    # Row 4: Forecasting
    st.subheader("🔮 Demand Forecasting")
    
    with st.spinner(f'Generating forecast for {selected_product_forecast}...'):
        forecaster.forecast_demand(selected_product_forecast, forecast_periods)
        forecast_fig = create_forecast_chart(forecaster, selected_product_forecast)
        
        if forecast_fig:
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Show forecast summary
            if selected_product_forecast in forecaster.forecasts:
                forecast_data = forecaster.forecasts[selected_product_forecast]['forecast']
                future_demand = forecast_data[forecast_data['ds'] > forecaster.df['date'].max()]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_forecast = future_demand['yhat'].mean()
                    st.metric("Average Forecasted Demand", f"{avg_forecast:.1f}")
                with col2:
                    max_forecast = future_demand['yhat'].max()
                    st.metric("Peak Forecasted Demand", f"{max_forecast:.1f}")
                with col3:
                    total_forecast = future_demand['yhat'].sum()
                    st.metric(f"Total Demand ({forecast_periods}d)", f"{total_forecast:.0f}")
        else:
            st.warning(f"Could not generate forecast for {selected_product_forecast}. Not enough data.")
    
    # Product details table
    st.subheader("📋 Product Details")
    
    # Format the data for display
    display_data = latest_data.copy()
    display_data = display_data[['product_id', 'product_name', 'current_stock', 'reorder_point', 'max_stock', 'status']]
    display_data['current_stock'] = display_data['current_stock'].astype(int)
    display_data['reorder_point'] = display_data['reorder_point'].astype(int)
    display_data['max_stock'] = display_data['max_stock'].astype(int)
    
    # Apply color coding
    def color_status(val):
        if val == 'Critical':
            return 'background-color: #ffcccc'
        elif val == 'Low':
            return 'background-color: #ffe6cc'
        elif val == 'High':
            return 'background-color: #cce6ff'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = display_data.style.applymap(color_status, subset=['status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            🔄 Dashboard updates in real-time • 📈 Powered by Prophet forecasting and Plotly visualizations
            <br><small>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()