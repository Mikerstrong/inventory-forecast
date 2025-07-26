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
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')
            prophet_data = prophet_data.dropna().sort_values('ds').reset_index(drop=True)
            
            # Create and fit the model with conservative settings
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,  # Disable to avoid timestamp issues
                yearly_seasonality=False
            )
            model.fit(prophet_data)
            
            # Create future dataframe manually to avoid timestamp arithmetic
            future_dates = []
            
            # Add historical dates
            for _, row in prophet_data.iterrows():
                future_dates.append(row['ds'])
            
            # Add future dates manually
            last_date = prophet_data['ds'].max()
            for i in range(1, periods + 1):
                future_date = last_date + pd.Timedelta(days=i)
                future_dates.append(future_date)
            
            future = pd.DataFrame({'ds': future_dates})
            future = future.drop_duplicates().sort_values('ds').reset_index(drop=True)
            
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
        height=400,
        xaxis=dict(
            tickangle=45,
            rangeslider=dict(visible=True),  # Adds a scroll bar for horizontal scrolling
            type='category',
        ),
        dragmode='pan',  # Enables panning
    )

    return fig

def create_demand_trends_chart(df, selected_products):
    """Create daily demand trends chart."""
    fig = go.Figure()
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    # Collect spike info for all products
    spike_rows = []
    for i, product in enumerate(selected_products):
        product_data = df[df['product_id'] == product]
        # Spike detection: demand > mean + 2*std
        mean = product_data['daily_demand'].mean()
        std = product_data['daily_demand'].std()
        spike_mask = product_data['daily_demand'] > (mean + 2 * std)
        spikes = product_data[spike_mask]
        # Add main demand line
        fig.add_trace(go.Scatter(
            x=product_data['date'],
            y=product_data['daily_demand'],
            mode='lines+markers',
            name=f'Demand - {product}',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
        # Overlay spike markers
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes['date'],
                y=spikes['daily_demand'],
                mode='markers',
                name=f'Spike - {product}',
                marker=dict(color='red', size=12, symbol='star'),
                showlegend=True
            ))
            for _, row in spikes.iterrows():
                spike_rows.append({
                    'product_id': row['product_id'],
                    'product_name': row.get('product_name', str(product)),
                    'date': row['date'],
                    'demand': row['daily_demand']
                })
    fig.update_layout(
        title="Daily Demand Trends (Spikes Highlighted)",
        xaxis_title="Date",
        yaxis_title="Daily Demand",
        height=400
    )
    return fig, spike_rows

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
    
    # Confidence intervals - using separate traces to avoid concatenation issues
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        showlegend=False,
        name='Confidence Interval'
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
    
    # Low stock items dropdown
    low_stock_items = latest_data[latest_data['status'].isin(['Critical', 'Low'])]
    if len(low_stock_items) > 0:
        st.sidebar.subheader("⚠️ Low Stock Alert")
        low_stock_options = [f"{row['product_id']} - {row['product_name']} ({row['status']})" 
                           for _, row in low_stock_items.iterrows()]
        selected_low_stock = st.sidebar.selectbox(
            "Items needing attention:",
            ["Select an item..."] + low_stock_options,
            help="Products with Critical or Low stock status"
        )
        
        # Extract product ID from selection
        if selected_low_stock != "Select an item...":
            selected_low_stock_id = selected_low_stock.split(" - ")[0]
            st.sidebar.info(f"Selected: {selected_low_stock_id}")
            
            # Button to use this product for forecasting
            if st.sidebar.button("📊 Forecast This Item", key="forecast_low_stock"):
                st.session_state.forecast_product = selected_low_stock_id
    else:
        st.sidebar.success("✅ No low stock items!")
    
    selected_products_trends = st.sidebar.multiselect(
        "Select products for trend analysis",
        all_products,
        default=all_products[:3]
    )
    
    # Use session state for forecast selection if available, otherwise default
    default_forecast_index = 0
    if hasattr(st.session_state, 'forecast_product') and st.session_state.forecast_product in all_products:
        default_forecast_index = list(all_products).index(st.session_state.forecast_product)
    
    selected_product_forecast = st.sidebar.selectbox(
        "Select product for forecasting",
        all_products,
        index=default_forecast_index
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
        st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
        health_fig = create_inventory_health_gauge(latest_data)
        st.plotly_chart(health_fig, use_container_width=True)
    
    with col2:
        st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
        status_fig = create_status_distribution_chart(latest_data)
        st.plotly_chart(status_fig, use_container_width=True)
    
    # Row 2: Stock levels
    st.subheader("📦 Current Stock Levels")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    stock_fig = create_stock_levels_chart(latest_data)
    st.plotly_chart(stock_fig, use_container_width=True)

    # Additional visualization 1: Stacked Bar Chart (Current Stock vs. Reorder Point)
    st.subheader("Current Stock vs. Reorder Point (Stacked Bar)")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    stacked_fig = go.Figure()
    stacked_fig.add_trace(go.Bar(
        x=latest_data['product_name'],
        y=latest_data['current_stock'],
        name='Current Stock',
        marker_color='blue',
    ))
    stacked_fig.add_trace(go.Bar(
        x=latest_data['product_name'],
        y=latest_data['reorder_point'],
        name='Reorder Point',
        marker_color='red',
    ))
    stacked_fig.update_layout(
        barmode='stack',
        title="Current Stock vs. Reorder Point",
        xaxis_title="Product",
        yaxis_title="Quantity",
        height=400
    )
    st.plotly_chart(stacked_fig, use_container_width=True)

    # Additional visualization 2: Heatmap of Stock Status
    st.subheader("Stock Status Heatmap")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    heatmap_data = latest_data.copy()
    heatmap_data['status_code'] = heatmap_data['status'].map({
        'Critical': 0,
        'Low': 1,
        'Good': 2,
        'High': 3
    })
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[heatmap_data['status_code']],
        x=heatmap_data['product_name'],
        y=['Status'],
        colorscale=[
            [0.0, 'red'],
            [0.25, 'orange'],
            [0.5, 'green'],
            [0.75, 'blue'],
            [1.0, 'blue']
        ],
        colorbar=dict(
            tickvals=[0,1,2,3],
            ticktext=['Critical','Low','Good','High'],
            title='Status'
        )
    ))
    heatmap_fig.update_layout(
        title="Stock Status by Product (Heatmap)",
        xaxis_title="Product",
        yaxis_title="Status",
        height=500
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Additional visualization 3: Inventory Metrics Heatmap
    st.subheader("Inventory Metrics Heatmap")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    metrics = ['current_stock', 'reorder_point', 'max_stock']
    metrics_heatmap = latest_data.set_index('product_name')[metrics].T
    metrics_fig = go.Figure(data=go.Heatmap(
        z=metrics_heatmap.values,
        x=metrics_heatmap.columns,
        y=metrics_heatmap.index,
        colorscale='Viridis',
        colorbar=dict(title='Quantity')
    ))
    metrics_fig.update_layout(
        title="Inventory Metrics Heatmap",
        xaxis_title="Product",
        yaxis_title="Metric",
        height=600
    )
    st.plotly_chart(metrics_fig, use_container_width=True)

    # Additional visualization 4: Days Until Stockout Heatmap
    st.subheader("Days Until Stockout Heatmap")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    # Calculate days until stockout (simple estimate: current_stock / avg daily demand)
    days_heatmap_data = latest_data.copy()
    # Merge with average daily demand per product
    avg_demand = forecaster.df.groupby('product_id')['daily_demand'].mean().reset_index()
    avg_demand.columns = ['product_id', 'avg_daily_demand']
    days_heatmap_data = days_heatmap_data.merge(avg_demand, on='product_id', how='left')
    days_heatmap_data['days_until_stockout'] = days_heatmap_data.apply(
        lambda row: row['current_stock'] / row['avg_daily_demand'] if row['avg_daily_demand'] > 0 else None, axis=1
    )
    days_fig = go.Figure(data=go.Heatmap(
        z=[days_heatmap_data['days_until_stockout']],
        x=days_heatmap_data['product_name'],
        y=['Days Until Stockout'],
        colorscale='RdYlGn',
        colorbar=dict(title='Days')
    ))
    days_fig.update_layout(
        title="Days Until Stockout Heatmap",
        xaxis_title="Product",
        yaxis_title="Metric",
        height=500
    )
    st.plotly_chart(days_fig, use_container_width=True)

    # Additional visualization: Days Until Stockout Horizontal Bar Chart
    st.subheader("Days Until Stockout by Product (Bar Chart)")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    bar_fig = px.bar(
        days_heatmap_data,
        x='days_until_stockout',
        y='product_name',
        orientation='h',
        color='days_until_stockout',
        color_continuous_scale='RdYlGn',
        labels={'days_until_stockout': 'Days Until Stockout', 'product_name': 'Product'},
        title='Days Until Stockout by Product'
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Additional visualization: Days Until Stockout Risk Pie Chart
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    def risk_category(days):
        if days is None:
            return 'Unknown'
        elif days < 7:
            return 'Critical'
        elif days < 14:
            return 'Low'
        else:
            return 'Safe'
    days_heatmap_data['risk'] = days_heatmap_data['days_until_stockout'].apply(risk_category)
    risk_counts = days_heatmap_data['risk'].value_counts()
    pie_fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=['red', 'orange', 'green', 'gray'])
    )])
    pie_fig.update_layout(title='Stockout Risk Distribution')
    st.plotly_chart(pie_fig, use_container_width=True)

    # Additional visualization: Days Until Stockout Table with Conditional Formatting
    st.subheader("Days Until Stockout Table")
    st.markdown("<span style='color:gray;font-size:small;'>(Pandas Styler)</span>", unsafe_allow_html=True)
    def color_days(val):
        if pd.isnull(val):
            return 'background-color: #cccccc'
        elif val < 7:
            return 'background-color: #ffcccc'
        elif val < 14:
            return 'background-color: #ffe6cc'
        else:
            return 'background-color: #ccffcc'
    styled_days = days_heatmap_data[['product_name', 'days_until_stockout']].style.applymap(color_days, subset=['days_until_stockout'])
    st.dataframe(styled_days, use_container_width=True)

    # Additional visualization: Days Until Stockout Scatter Plot
    st.subheader("Days Until Stockout vs. Current Stock (Scatter Plot)")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    scatter_fig = px.scatter(
        days_heatmap_data,
        x='current_stock',
        y='days_until_stockout',
        color='risk',
        labels={'current_stock': 'Current Stock', 'days_until_stockout': 'Days Until Stockout'},
        title='Days Until Stockout vs. Current Stock'
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Additional visualization: Days Until Stockout Pareto Chart
    st.subheader("Days Until Stockout Pareto Chart")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    # Sort products by ascending days until stockout
    pareto_data = days_heatmap_data[['product_name', 'days_until_stockout']].dropna().sort_values('days_until_stockout')
    pareto_fig = go.Figure()
    # Bar chart for days until stockout
    pareto_fig.add_trace(go.Bar(
        x=pareto_data['product_name'],
        y=pareto_data['days_until_stockout'],
        name='Days Until Stockout',
        marker_color='blue'
    ))
    pareto_fig.update_layout(
        title='Days Until Stockout Pareto Chart',
        xaxis_title='Product',
        yaxis_title='Days Until Stockout',
        height=500
    )
    st.plotly_chart(pareto_fig, use_container_width=True)

    # Row 3: Demand trends
    if selected_products_trends:
        st.subheader("📊 Daily Demand Trends")
        st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
        trends_fig, spike_rows = create_demand_trends_chart(forecaster.df, selected_products_trends)
        st.plotly_chart(trends_fig, use_container_width=True)
        # Show spike table and alert if any spikes detected
        if spike_rows:
            st.warning(f"Usage spikes detected: {len(spike_rows)} event(s)")
            st.subheader("⚡ Usage Spike Events Table")
            st.markdown("<span style='color:gray;font-size:small;'>(Pandas DataFrame)</span>", unsafe_allow_html=True)
            spike_df = pd.DataFrame(spike_rows)
            spike_df['date'] = pd.to_datetime(spike_df['date']).dt.strftime('%Y-%m-%d')
            st.dataframe(spike_df, use_container_width=True)
        else:
            st.info("No usage spikes detected in selected products.")
    
    # Row 4: Forecasting
    st.subheader("🔮 Demand Forecasting")
    st.markdown("<span style='color:gray;font-size:small;'>(Prophet, Plotly)</span>", unsafe_allow_html=True)
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
    
    # Additional visualization: Inventory Forecast Gauges
    st.subheader("Inventory Forecast Gauges")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    # Example gauges: Average, Min, Max forecasted demand for selected product
    if selected_product_forecast in forecaster.forecasts:
        forecast_data = forecaster.forecasts[selected_product_forecast]['forecast']
        future_demand = forecast_data[forecast_data['ds'] > forecaster.df['date'].max()]
        avg_forecast = future_demand['yhat'].mean()
        min_forecast = future_demand['yhat'].min()
        max_forecast = future_demand['yhat'].max()
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_forecast,
                title={'text': "Avg Forecasted Demand"},
                gauge={'axis': {'range': [0, max_forecast]}, 'bar': {'color': "blue"}}
            ))
            avg_gauge.update_layout(height=250)
            st.plotly_chart(avg_gauge, use_container_width=True)
        with col2:
            min_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min_forecast,
                title={'text': "Min Forecasted Demand"},
                gauge={'axis': {'range': [0, max_forecast]}, 'bar': {'color': "green"}}
            ))
            min_gauge.update_layout(height=250)
            st.plotly_chart(min_gauge, use_container_width=True)
        with col3:
            max_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_forecast,
                title={'text': "Max Forecasted Demand"},
                gauge={'axis': {'range': [0, max_forecast]}, 'bar': {'color': "red"}}
            ))
            max_gauge.update_layout(height=250)
            st.plotly_chart(max_gauge, use_container_width=True)
    
    # Product details table
    st.subheader("📋 Product Details")
    st.markdown("<span style='color:gray;font-size:small;'>(Pandas Styler)</span>", unsafe_allow_html=True)
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

    # Additional visualization: Days Until Stockout Gauges (10, 20, 30 days)
    st.subheader("Days Until Stockout Gauges")
    st.markdown("<span style='color:gray;font-size:small;'>(Plotly)</span>", unsafe_allow_html=True)
    # Create bins for 10, 20, 30 days
    bins = [0, 10, 20, 30, np.inf]
    labels = ['<=10 days', '11-20 days', '21-30 days', '>30 days']
    days_heatmap_data['stockout_bin'] = pd.cut(days_heatmap_data['days_until_stockout'], bins=bins, labels=labels, right=True)
    bin_counts = days_heatmap_data['stockout_bin'].value_counts().reindex(labels, fill_value=0)
    col1, col2, col3, col4 = st.columns(4)
    gauge_bins = {}
    for i, (label, col) in enumerate(zip(labels, [col1, col2, col3, col4])):
        with col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bin_counts[label],
                title={'text': label},
                gauge={'axis': {'range': [0, len(days_heatmap_data)]}, 'bar': {'color': ["red", "orange", "yellow", "green"][i]}}
            ))
            gauge.update_layout(height=200)
            st.plotly_chart(gauge, use_container_width=True)
            # Add clickable button for each gauge
            if st.button(f"Show parts for {label}", key=f"show_{label}"):
                selected_bin = label
                show_table = True
                gauge_bins[label] = True
            else:
                gauge_bins[label] = False
    # Show table if any gauge button was clicked
    if any(gauge_bins.values()):
        st.markdown("<span style='color:gray;font-size:small;'>(Pandas DataFrame)</span>", unsafe_allow_html=True)
        selected_bin = [label for label, clicked in gauge_bins.items() if clicked][0]
        st.subheader(f"Parts with Days Until Stockout: {selected_bin}")
        filtered = days_heatmap_data[days_heatmap_data['stockout_bin'] == selected_bin]
        if not filtered.empty:
            table_cols = ['product_id', 'product_name', 'current_stock', 'reorder_point', 'max_stock', 'avg_daily_demand', 'days_until_stockout', 'status']
            st.dataframe(filtered[table_cols], use_container_width=True)
        else:
            st.info("No parts in this category.")

if __name__ == "__main__":
    main()