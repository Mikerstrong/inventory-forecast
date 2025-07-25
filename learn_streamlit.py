#!/usr/bin/env python3
"""
Interactive Prophet Inventory Learning - Streamlit App

This Streamlit application provides an interactive learning environment
for Prophet inventory forecasting with 250 parts and various scenarios.

Run with: streamlit run learn_streamlit.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
from datetime import datetime, timedelta
import random

# Set random seeds for consistent results across sessions
np.random.seed(42)
random.seed(42)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Prophet Inventory Learning Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProphetInventoryLearning:
    def __init__(self):
        """Initialize the learning environment."""
        self.df = None
        self.parts_data = None
        
    @st.cache_data
    def load_sample_data(_self):
        """Load sample data from CSV file."""
        try:
            # Load the sample data
            df = pd.read_csv('data/sample_data.csv')
            
            # Ensure proper data types
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'])
            df['current_stock'] = pd.to_numeric(df['current_stock'])
            df['reorder_point'] = pd.to_numeric(df['reorder_point'])
            df['max_stock'] = pd.to_numeric(df['max_stock'])
            df['lead_time'] = pd.to_numeric(df['lead_time'])
            
            # Create parts metadata from the data
            parts_data = df.groupby('part_id').agg({
                'category': 'first',
                'reorder_point': 'first',
                'max_stock': 'first',
                'lead_time': 'first',
                'y': 'mean'  # Use mean demand as base_demand
            }).reset_index()
            
            parts_data = parts_data.rename(columns={'y': 'base_demand'})
            
            # Add additional metadata based on category
            parts_data['seasonality'] = parts_data['category'].map({
                'Fast_Moving': 0.3,
                'Medium_Moving': 0.2,
                'Slow_Moving': 0.1
            })
            
            parts_data['trend'] = parts_data['category'].map({
                'Fast_Moving': 0.02,
                'Medium_Moving': 0.01,
                'Slow_Moving': -0.005
            })
            
            parts_data['noise'] = parts_data['category'].map({
                'Fast_Moving': 0.1,
                'Medium_Moving': 0.15,
                'Slow_Moving': 0.3
            })
            
            return df, parts_data
            
        except FileNotFoundError:
            st.error("Sample data file not found. Please ensure 'data/sample_data.csv' exists.")
            return None, None
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return None, None
    
    def create_basic_forecast_chart(self, part_data, part_id, forecast_days=30):
        """Create basic Prophet forecast chart."""
        try:
            # Prepare data for Prophet with proper data types
            prophet_data = part_data[['ds', 'y']].copy()
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')
            
            # Remove any NaN values and ensure data is sorted
            prophet_data = prophet_data.dropna().sort_values('ds').reset_index(drop=True)
            
            if len(prophet_data) < 2:
                st.warning(f"Not enough data for {part_id}. Need at least 2 data points.")
                return None, None, None
            
            # Ensure we have a proper daily frequency
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds']).dt.date
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            # Create and fit Prophet model with minimal settings to avoid timestamp issues
            model = Prophet(
                weekly_seasonality=False,  # Disable to avoid potential issues
                yearly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Suppress Prophet's verbose output
            with st.spinner("Training Prophet model..."):
                model.fit(prophet_data)
            
            # Create future dataframe manually to avoid make_future_dataframe issues
            start_date = prophet_data['ds'].min()
            end_date = prophet_data['ds'].max()
            
            # Generate future dates manually
            all_dates = []
            current_date = start_date
            
            # Historical dates
            for _, row in prophet_data.iterrows():
                all_dates.append(row['ds'])
            
            # Future dates
            last_date = prophet_data['ds'].max()
            for i in range(1, forecast_days + 1):
                future_date = last_date + pd.Timedelta(days=i)
                all_dates.append(future_date)
            
            future = pd.DataFrame({'ds': all_dates})
            future['ds'] = pd.to_datetime(future['ds'])
            future = future.drop_duplicates().sort_values('ds').reset_index(drop=True)
            
            with st.spinner("Generating predictions..."):
                forecast = model.predict(future)
            
            # Create the chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'],
                y=prophet_data['y'],
                mode='markers',
                name='Historical Demand',
                marker=dict(color='blue', size=4)
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2)
            ))
            
            # Confidence intervals - using separate traces to avoid concatenation issues
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                showlegend=False,
                name='Confidence Interval'
            ))
            
            # Add vertical line to separate historical from forecast
            historical_end = prophet_data['ds'].max()
            fig.add_vline(
                x=historical_end,
                line_dash="dash",
                line_color="gray",
                annotation_text="Forecast Start"
            )
            
            fig.update_layout(
                title=f"Prophet Forecast - {part_id}",
                xaxis_title="Date",
                yaxis_title="Daily Demand",
                height=500
            )
            
            return fig, forecast, model
            
        except Exception as e:
            st.error(f"Error creating forecast for {part_id}: {str(e)}")
            st.error("This might be due to insufficient data or data quality issues.")
            st.info("Try selecting a different part or check the data generation process.")
            return None, None, None
    
    def create_seasonality_chart(self, model, forecast):
        """Create seasonality decomposition chart."""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Trend', 'Weekly Seasonality', 'Overall Forecast'],
                vertical_spacing=0.08
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['trend'], 
                          mode='lines', name='Trend', line=dict(color='green')),
                row=1, col=1
            )
            
            # Weekly seasonality
            if 'weekly' in forecast.columns:
                fig.add_trace(
                    go.Scatter(x=forecast['ds'], y=forecast['weekly'], 
                              mode='lines', name='Weekly', line=dict(color='orange')),
                    row=2, col=1
                )
            
            # Overall forecast
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                          mode='lines', name='Forecast', line=dict(color='red')),
                row=3, col=1
            )
            
            fig.update_layout(height=600, title_text="Prophet Seasonality Decomposition")
            return fig
            
        except Exception as e:
            st.error(f"Error creating seasonality chart: {e}")
            return None
    
    def create_stock_analysis_chart(self, part_data, part_id):
        """Create stock level analysis chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Daily Demand Over Time', 'Stock Levels Over Time'],
            vertical_spacing=0.1
        )
        
        # Daily demand
        fig.add_trace(
            go.Scatter(x=part_data['ds'], y=part_data['y'], 
                      mode='lines', name='Daily Demand', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Stock levels
        fig.add_trace(
            go.Scatter(x=part_data['ds'], y=part_data['current_stock'], 
                      mode='lines', name='Current Stock', line=dict(color='green')),
            row=2, col=1
        )
        
        # Reorder point
        reorder_point = part_data['reorder_point'].iloc[0]
        fig.add_hline(
            y=reorder_point, line_dash="dash", line_color="red",
            annotation_text=f"Reorder Point ({reorder_point})",
            row=2, col=1
        )
        
        # Max stock
        max_stock = part_data['max_stock'].iloc[0]
        fig.add_hline(
            y=max_stock, line_dash="dash", line_color="orange",
            annotation_text=f"Max Stock ({max_stock})",
            row=2, col=1
        )
        
        fig.update_layout(height=500, title_text=f"Stock Analysis - {part_id}")
        return fig
    
    def create_uncertainty_chart(self, part_data, part_id):
        """Create uncertainty analysis with different confidence intervals."""
        prophet_data = part_data[['ds', 'y']].copy()
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) < 2:
            st.warning(f"Not enough data for uncertainty analysis of {part_id}")
            return None
        
        intervals = [0.8, 0.95, 0.99]
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=prophet_data['ds'],
            y=prophet_data['y'],
            mode='markers',
            name='Historical Demand',
            marker=dict(color='blue', size=4)
        ))
        
        for i, interval in enumerate(intervals):
            try:
                model = Prophet(
                    interval_width=interval,
                    weekly_seasonality=False,  # Disable to avoid issues
                    yearly_seasonality=False,
                    daily_seasonality=False
                )
                model.fit(prophet_data)
                
                # Create future dates manually
                last_date = prophet_data['ds'].max()
                future_dates = []
                
                # Add historical dates
                for _, row in prophet_data.iterrows():
                    future_dates.append(row['ds'])
                
                # Add 30 future dates
                for i in range(1, 31):
                    future_date = last_date + pd.Timedelta(days=i)
                    future_dates.append(future_date)
                
                future = pd.DataFrame({'ds': future_dates})
                future = future.drop_duplicates().sort_values('ds').reset_index(drop=True)
                
                forecast = model.predict(future)
                
                # Only show forecast part for clarity
                forecast_part = forecast.tail(30)
                
                fig.add_trace(go.Scatter(
                    x=forecast_part['ds'],
                    y=forecast_part['yhat_upper'],
                    mode='lines',
                    name=f'{int(interval*100)}% Upper',
                    line=dict(color=colors[i], dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_part['ds'],
                    y=forecast_part['yhat_lower'],
                    mode='lines',
                    name=f'{int(interval*100)}% Lower',
                    line=dict(color=colors[i], dash='dash')
                ))
                
            except Exception as e:
                st.warning(f"Error with {interval} confidence interval: {e}")
        
        fig.update_layout(
            title=f"Uncertainty Analysis - {part_id}",
            xaxis_title="Date",
            yaxis_title="Daily Demand",
            height=500
        )
        
        return fig
    
    def display_part_statistics(self, part_data, part_id):
        """Display comprehensive part statistics."""
        
        # Basic statistics
        demand_stats = part_data['y'].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Demand", f"{demand_stats['mean']:.1f}")
            st.metric("Min Demand", f"{demand_stats['min']:.0f}")
        
        with col2:
            st.metric("Max Demand", f"{demand_stats['max']:.0f}")
            st.metric("Std Deviation", f"{demand_stats['std']:.1f}")
        
        with col3:
            current_stock = part_data['current_stock'].iloc[-1]
            reorder_point = part_data['reorder_point'].iloc[-1]
            st.metric("Current Stock", f"{current_stock:.0f}")
            st.metric("Reorder Point", f"{reorder_point:.0f}")
        
        with col4:
            max_stock = part_data['max_stock'].iloc[-1]
            lead_time = part_data['lead_time'].iloc[-1]
            st.metric("Max Stock", f"{max_stock:.0f}")
            st.metric("Lead Time", f"{lead_time} days")
        
        # Status analysis
        status_counts = part_data['status'].value_counts()
        
        st.subheader("📊 Historical Status Distribution")
        
        # Create status chart
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker=dict(colors=['green', 'orange', 'red', 'blue'])
        )])
        
        fig.update_layout(title="Stock Status Over Time", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics table
        st.subheader("📋 Detailed Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': ['Total Days', 'Days Critical', 'Days Low', 'Days Good', 'Days High'],
            'Value': [
                len(part_data),
                status_counts.get('Critical', 0),
                status_counts.get('Low', 0),
                status_counts.get('Good', 0),
                status_counts.get('High', 0)
            ],
            'Percentage': [
                100.0,
                (status_counts.get('Critical', 0) / len(part_data)) * 100,
                (status_counts.get('Low', 0) / len(part_data)) * 100,
                (status_counts.get('Good', 0) / len(part_data)) * 100,
                (status_counts.get('High', 0) / len(part_data)) * 100
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True)

def main():
    """Main Streamlit application."""
    st.title("🎓 Prophet Inventory Learning Lab")
    st.markdown("Interactive learning environment for Prophet inventory forecasting using sample inventory data")
    
    # Initialize the learning class
    learner = ProphetInventoryLearning()
    
    # Sidebar
    st.sidebar.header("🔧 Learning Controls")
    
    # Generate data
    with st.spinner("Loading sample inventory data..."):
        df, parts_data = learner.load_sample_data()
    
    if df is None or parts_data is None:
        st.error("Failed to load sample data. Please check the data file.")
        return
    
    st.sidebar.success(f"✅ Loaded {len(df):,} data points for {len(parts_data)} parts")
    
    # Part selection
    st.sidebar.subheader("📦 Part Selection")
    
    # Category filter
    categories = df['category'].unique()
    selected_category = st.sidebar.selectbox(
        "Filter by Category",
        ['All'] + list(categories)
    )
    
    # Filter parts by category
    if selected_category == 'All':
        available_parts = df['part_id'].unique()
    else:
        available_parts = df[df['category'] == selected_category]['part_id'].unique()
    
    selected_part = st.sidebar.selectbox(
        "Select Part for Analysis",
        available_parts
    )
    
    # Analysis options
    st.sidebar.subheader("📊 Analysis Options")
    
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        [
            "📈 Basic Forecasting",
            "🌊 Seasonality Analysis",
            "📦 Stock Level Analysis",
            "📊 Uncertainty Analysis",
            "📋 Part Statistics",
            "🔍 Model Comparison"
        ]
    )
    
    # Forecast parameters
    if analysis_type in ["📈 Basic Forecasting", "🌊 Seasonality Analysis", "📊 Uncertainty Analysis"]:
        forecast_days = st.sidebar.slider(
            "Forecast Days",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
    
    # Get selected part data
    part_data = df[df['part_id'] == selected_part].copy()
    part_info = parts_data[parts_data['part_id'] == selected_part].iloc[0]
    
    # Main content area
    st.header(f"Analysis for Part: {selected_part}")
    
    # Display part information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Category:** {part_info['category']}")
    with col2:
        st.info(f"**Base Demand:** {part_info['base_demand']}")
    with col3:
        st.info(f"**Lead Time:** {part_info['lead_time']} days")
    with col4:
        current_status = part_data['status'].iloc[-1]
        status_color = {
            'Critical': '🔴',
            'Low': '🟠', 
            'Good': '🟢',
            'High': '🔵'
        }
        st.info(f"**Status:** {status_color.get(current_status, '⚪')} {current_status}")
    
    # Analysis based on selection
    if analysis_type == "📈 Basic Forecasting":
        st.subheader("Basic Prophet Forecasting")
        st.markdown("Learn how Prophet predicts future demand using historical patterns.")
        
        with st.spinner("Generating forecast..."):
            fig, forecast, model = learner.create_basic_forecast_chart(part_data, selected_part, forecast_days)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            future_forecast = forecast.tail(forecast_days)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = future_forecast['yhat'].mean()
                st.metric("Average Forecast", f"{avg_forecast:.1f} units/day")
            
            with col2:
                total_forecast = future_forecast['yhat'].sum()
                st.metric(f"Total {forecast_days}-day Demand", f"{total_forecast:.0f} units")
            
            with col3:
                uncertainty = future_forecast['yhat_upper'].mean() - future_forecast['yhat_lower'].mean()
                st.metric("Avg Uncertainty Range", f"±{uncertainty/2:.1f} units")
            
            # Learning points
            st.info("""
            **🎓 Learning Points:**
            - Prophet uses historical patterns to predict future demand
            - Red shaded area shows uncertainty (confidence intervals)
            - Vertical line separates historical data from predictions
            - Weekly patterns are automatically detected and projected forward
            """)
    
    elif analysis_type == "🌊 Seasonality Analysis":
        st.subheader("Seasonality Decomposition")
        st.markdown("Understand how Prophet breaks down demand into trend and seasonal components.")
        
        with st.spinner("Analyzing seasonality..."):
            fig, forecast, model = learner.create_basic_forecast_chart(part_data, selected_part, forecast_days)
            
        if fig and model:
            seasonality_fig = learner.create_seasonality_chart(model, forecast)
            if seasonality_fig:
                st.plotly_chart(seasonality_fig, use_container_width=True)
            
            st.info("""
            **🎓 Learning Points:**
            - **Trend**: Long-term increase or decrease in demand
            - **Weekly Seasonality**: Regular patterns within each week
            - **Overall Forecast**: Combination of trend + seasonality + noise
            - Understanding components helps explain forecast behavior
            """)
    
    elif analysis_type == "📦 Stock Level Analysis":
        st.subheader("Stock Level vs Demand Analysis")
        st.markdown("Examine the relationship between demand patterns and inventory levels.")
        
        stock_fig = learner.create_stock_analysis_chart(part_data, selected_part)
        st.plotly_chart(stock_fig, use_container_width=True)
        
        # Stock statistics
        stockouts = (part_data['current_stock'] <= part_data['reorder_point']).sum()
        overstocks = (part_data['current_stock'] >= part_data['max_stock'] * 0.9).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Days Below Reorder Point", stockouts)
        with col2:
            st.metric("Days Near Max Stock", overstocks)
        with col3:
            avg_stock = part_data['current_stock'].mean()
            st.metric("Average Stock Level", f"{avg_stock:.0f}")
        
        st.info("""
        **🎓 Learning Points:**
        - Top chart shows demand volatility over time
        - Bottom chart shows how stock levels respond to demand
        - Red dashed line = Reorder point (time to restock)
        - Orange dashed line = Maximum stock capacity
        - Goal: Minimize time below reorder point and above max stock
        """)
    
    elif analysis_type == "📊 Uncertainty Analysis":
        st.subheader("Uncertainty & Confidence Intervals")
        st.markdown("Learn how Prophet quantifies prediction uncertainty for safety stock planning.")
        
        with st.spinner("Analyzing uncertainty..."):
            uncertainty_fig = learner.create_uncertainty_chart(part_data, selected_part)
        
        if uncertainty_fig:
            st.plotly_chart(uncertainty_fig, use_container_width=True)
        
        st.info("""
        **🎓 Learning Points:**
        - **80% Confidence**: Conservative estimate, lower safety stock needed
        - **95% Confidence**: Standard approach, balanced risk/cost
        - **99% Confidence**: High certainty, higher safety stock required
        - Wider intervals = More uncertainty = More safety stock needed
        - Use confidence intervals to calculate optimal safety stock levels
        """)
    
    elif analysis_type == "📋 Part Statistics":
        st.subheader("Comprehensive Part Statistics")
        st.markdown("Detailed statistical analysis of the selected part.")
        
        learner.display_part_statistics(part_data, selected_part)
        
        st.info("""
        **🎓 Learning Points:**
        - Statistics help understand demand patterns and variability
        - Status distribution shows inventory health over time
        - High variability (std dev) suggests need for higher safety stock
        - Lead time affects how quickly you can respond to demand changes
        """)
    
    elif analysis_type == "🔍 Model Comparison":
        st.subheader("Prophet Model Comparison")
        st.markdown("Compare different Prophet configurations to find the best approach.")
        
        with st.spinner("Comparing different Prophet models..."):
            prophet_data = part_data[['ds', 'y']].copy()
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 10:
                st.warning("Not enough data for model comparison. Need at least 10 data points.")
                return
            
            # Split data for validation
            split_point = int(len(prophet_data) * 0.8)
            train_data = prophet_data[:split_point]
            test_data = prophet_data[split_point:]
            
            models = {
                'Basic': Prophet(weekly_seasonality=False, yearly_seasonality=False),
                'With Weekly Seasonality': Prophet(weekly_seasonality=True, yearly_seasonality=False),
                'Conservative': Prophet(changepoint_prior_scale=0.001, weekly_seasonality=False),
                'Flexible': Prophet(changepoint_prior_scale=0.1, weekly_seasonality=False)
            }
            
            results = []
            
            for name, model in models.items():
                try:
                    model.fit(train_data)
                    
                    # Create future dates manually
                    future_dates = []
                    
                    # Add training dates
                    for _, row in train_data.iterrows():
                        future_dates.append(row['ds'])
                    
                    # Add test period dates
                    last_train_date = train_data['ds'].max()
                    for i in range(1, len(test_data) + 1):
                        future_date = last_train_date + pd.Timedelta(days=i)
                        future_dates.append(future_date)
                    
                    future = pd.DataFrame({'ds': future_dates})
                    future = future.drop_duplicates().sort_values('ds').reset_index(drop=True)
                    
                    forecast = model.predict(future)
                    
                    # Calculate metrics
                    test_predictions = forecast.tail(len(test_data))['yhat'].values
                    test_actual = test_data['y'].values
                    
                    mae = np.mean(np.abs(test_predictions - test_actual))
                    mape = np.mean(np.abs((test_actual - test_predictions) / (test_actual + 1e-8))) * 100
                    
                    results.append({
                        'Model': name,
                        'MAE': mae,
                        'MAPE': mape
                    })
                except Exception as e:
                    st.warning(f"Error with {name} model: {e}")
            
            if results:
                results_df = pd.DataFrame(results)
                
                # Display results
                st.dataframe(results_df.round(2), use_container_width=True)
                
                # Find best model
                best_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']
                st.success(f"🏆 Best performing model: **{best_model}**")
                
                # Create comparison chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results_df['Model'],
                    y=results_df['MAE'],
                    name='MAE',
                    marker_color='blue'
                ))
                
                fig.update_layout(
                    title="Model Performance Comparison (Lower is Better)",
                    xaxis_title="Model Type",
                    yaxis_title="Mean Absolute Error (MAE)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **🎓 Learning Points:**
        - **MAE**: Average prediction error in units
        - **MAPE**: Error as percentage of actual demand
        - **Conservative**: Less sensitive to changes, more stable
        - **Flexible**: More sensitive to changes, captures patterns better
        - Choose model based on your inventory tolerance for forecast errors
        """)
    
    # Footer with learning resources
    st.markdown("---")
    st.markdown("""
    ### 🎯 Key Prophet Learning Takeaways:
    
    1. **Data Structure**: Prophet needs 'ds' (dates) and 'y' (values) columns
    2. **Seasonality**: Prophet automatically detects weekly, yearly patterns
    3. **Uncertainty**: Confidence intervals help calculate safety stock
    4. **Trends**: Prophet identifies and projects demand trends
    5. **Validation**: Always test model performance before deploying
    
    **📚 Next Steps**: Try different parts, categories, and analysis types to deepen your understanding!
    """)

if __name__ == "__main__":
    main()
