# Inventory Management and Forecasting System

A Python-based inventory management system with an interactive **Streamlit web application** that provides real-time demand forecasting and dashboard visualization using Prophet and Plotly.

## Features

- 🌐 **Interactive Web Dashboard**: Real-time Streamlit interface with live controls
- 📊 **Dynamic Visualizations**: Interactive inventory status visualization with Plotly
- 🔮 **Demand Forecasting**: Uses Facebook Prophet for accurate demand prediction
- 🚦 **Status Indicators**: Color-coded inventory levels (Green/Yellow/Red)
- 📈 **Trend Analysis**: Historical demand patterns and future projections
- 🎯 **Reorder Alerts**: Automatic low-stock notifications with real-time metrics
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Real-time Updates**: Interactive controls for product selection and forecast periods

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd inventory-forecast
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit Web Application

Launch the interactive dashboard with:

```bash
streamlit run inventory_forecast.py
```

This will:
- Start a local web server (typically at `http://localhost:8501`)
- Load inventory data from `data/sample_inventory.csv`
- Generate real-time interactive dashboard
- Provide controls for product selection and forecasting parameters

### Using the Dashboard

The Streamlit interface provides several interactive components:

#### 📋 Sidebar Controls
- **Data Summary**: Overview of records, products, and date range
- **Product Selection**: Multi-select for trend analysis
- **Forecast Controls**: Choose product and forecast period (7-90 days)

#### 📊 Dashboard Sections
1. **Status Summary Cards**: Quick overview with metrics
2. **Inventory Health Gauge**: Overall system health percentage
3. **Stock Status Distribution**: Pie chart of inventory categories
4. **Current Stock Levels**: Interactive bar chart with reorder points
5. **Daily Demand Trends**: Multi-product trend analysis
6. **Demand Forecasting**: Prophet-generated forecasts with confidence intervals
7. **Product Details Table**: Sortable table with color-coded status

## File Structure

```
inventory-forecast/
├── requirements.txt              # Python dependencies (includes Streamlit)
├── inventory_forecast.py         # Streamlit web application
├── README.md                    # This documentation
└── data/
    └── sample_inventory.csv     # Sample inventory data
```

## Data Format

The system expects CSV data with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `date` | Date of record | 2024-01-01 |
| `product_id` | Unique product identifier | P001 |
| `product_name` | Product display name | Widget A |
| `current_stock` | Current inventory level | 500 |
| `daily_demand` | Daily demand quantity | 25 |
| `reorder_point` | Minimum stock threshold | 100 |
| `max_stock` | Maximum stock capacity | 1000 |
| `supplier_lead_time` | Days to receive new stock | 7 |

## Dashboard Components

### 1. Inventory Health Gauge
- Overall system health percentage
- Color-coded status indicator
- Performance against target thresholds
- Real-time updates based on current data

### 2. Status Summary Cards
- **Critical (🔴)**: Products at or below reorder point
- **Low Stock (🟠)**: Products within 1.5x of reorder point
- **Good (🟢)**: Products in healthy stock range
- **High Stock (🔵)**: Products at 90%+ of maximum capacity

### 3. Stock Levels by Product
- Interactive bar chart with current stock levels
- Color-coded status bars
- Visual reorder point reference line
- Hover information with detailed data

### 4. Daily Demand Trends
- Multi-product trend visualization
- Interactive product selection
- Time series analysis
- Trend identification and patterns

### 5. Demand Forecasting
- Prophet-generated demand forecasts
- Confidence intervals visualization
- Adjustable forecast periods (7-90 days)
- Forecast summary metrics

### 6. Product Details Table
- Sortable and filterable table
- Color-coded status rows
- Complete product information
- Export capabilities

## Interactive Features

### Real-time Controls
- **Product Selection**: Choose which products to analyze
- **Forecast Periods**: Adjust prediction timeframe
- **Multi-product Trends**: Compare demand patterns
- **Dynamic Updates**: All charts update automatically

### Visual Interactions
- **Zoom and Pan**: Interactive chart navigation
- **Hover Details**: Detailed information on hover
- **Toggle Series**: Show/hide data series
- **Download Options**: Export charts as images

## Status Color Coding

- 🔴 **Critical (Red)**: Stock at or below reorder point
- 🟠 **Low (Orange)**: Stock within 1.5x of reorder point
- 🔵 **High (Blue)**: Stock at 90%+ of maximum capacity
- 🟢 **Good (Green)**: Healthy stock levels

## Customization

### Adding New Products

Add new entries to `data/sample_inventory.csv` with the required columns. The Streamlit app automatically detects and processes new products on refresh.

### Modifying Forecast Parameters

Edit the `forecast_demand()` method in `inventory_forecast.py`:

```python
model = Prophet(
    daily_seasonality=False,    # Enable/disable daily patterns
    weekly_seasonality=True,    # Enable/disable weekly patterns
    yearly_seasonality=False    # Enable/disable yearly patterns
)
```

### Customizing the Dashboard

Modify various functions in `inventory_forecast.py` to adjust:
- Chart types and configurations
- Color schemes and themes
- Layout and component arrangement
- Interactive controls and widgets

## Streamlit Configuration

### Page Configuration
The app is configured with:
- Wide layout for optimal dashboard viewing
- Custom page title and icon
- Expandable sidebar for controls

### Performance Optimization
- Caching for data loading
- Efficient chart rendering
- Responsive design elements

## Dependencies

- **pandas** (≥1.5.0): Data manipulation and analysis
- **plotly** (≥5.0.0): Interactive visualization
- **prophet** (≥1.1.0): Time series forecasting
- **streamlit** (≥1.28.0): Web application framework

## Deployment Options

### Local Development
```bash
streamlit run inventory_forecast.py
```

### Streamlit Cloud
1. Connect your GitHub repository to Streamlit Cloud
2. Deploy directly from the web interface
3. Automatic updates on code changes

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "inventory_forecast.py"]
```

### Heroku Deployment
Add `setup.sh` and `Procfile` for Heroku deployment:

```bash
# setup.sh
mkdir -p ~/.streamlit/
echo "[server]" > ~/.streamlit/config.toml
echo "port = $PORT" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml

# Procfile
web: sh setup.sh && streamlit run inventory_forecast.py
```

## Troubleshooting

### Common Issues

1. **Streamlit Installation Error**:
   ```bash
   pip install --upgrade streamlit
   ```

2. **Prophet Installation Error**:
   ```bash
   # On macOS with Apple Silicon
   conda install -c conda-forge prophet
   
   # Alternative installation
   pip install pystan==2.19.1.1
   pip install prophet
   ```

3. **Port Already in Use**:
   ```bash
   streamlit run inventory_forecast.py --server.port 8502
   ```

4. **Missing Data Error**:
   - Ensure `data/sample_inventory.csv` exists
   - Verify CSV has all required columns
   - Check date format (YYYY-MM-DD)

5. **Charts Not Loading**:
   - Check browser console for errors
   - Refresh the page
   - Clear browser cache

### Performance Optimization

For large datasets:
- Use data sampling in the sidebar controls
- Limit the number of products in trend analysis
- Reduce default forecast periods
- Implement data caching with `@st.cache_data`

## Streamlit-Specific Features

### Widgets Used
- `st.sidebar` for controls
- `st.multiselect` for product selection
- `st.selectbox` for single product choice
- `st.slider` for forecast periods
- `st.metric` for status cards
- `st.plotly_chart` for visualizations
- `st.dataframe` for data tables

### Layout Components
- `st.columns` for responsive layout
- `st.container` for organized sections
- `st.expander` for collapsible content
- `st.tabs` for organized views

## Future Enhancements

- 📧 Email alerts integration with Streamlit
- 📊 Advanced analytics dashboard pages
- 🔄 Real-time data source integration
- 📱 Enhanced mobile responsiveness
- 🤖 Machine learning model comparison
- 💾 Data upload and download features
- 🔐 User authentication and permissions
- 📈 Advanced forecasting models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `streamlit run inventory_forecast.py`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section
- Review Streamlit documentation: https://docs.streamlit.io/

---

**Happy forecasting with Streamlit!** 🚀📊🌐