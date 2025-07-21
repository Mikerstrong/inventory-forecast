# Inventory Management & Forecasting System

A comprehensive Python-based inventory management system that provides demand forecasting and interactive dashboards with visual stock level indicators.

## Features

- 📊 **Demand Forecasting**: Uses Facebook Prophet for time-series forecasting
- 🚦 **Stock Level Indicators**: Green/Yellow/Red status system for inventory levels
- 📈 **Interactive Dashboard**: Plotly-based HTML dashboard with multiple visualizations
- 🔍 **Inventory Analysis**: Real-time stock status monitoring and reorder point management
- 📋 **CSV Data Support**: Easy data import from CSV files

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mikerstrong/inventory-forecast.git
cd inventory-forecast
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Run the forecasting system**:
```bash
python inventory_forecast.py
```

2. **View the dashboard**:
Open `templates/dashboard.html` in your web browser to view the interactive dashboard.

3. **Customize your data**:
Replace `data/sample_inventory.csv` with your own inventory data following the same format.

## Data Format

The system expects CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `date` | Date in YYYY-MM-DD format |
| `product_id` | Unique product identifier |
| `product_name` | Human-readable product name |
| `demand` | Daily demand quantity |
| `current_stock` | Current stock level |
| `reorder_point` | Minimum stock threshold |
| `lead_time_days` | Supplier lead time in days |

### Sample Data Structure
```csv
date,product_id,product_name,demand,current_stock,reorder_point,lead_time_days
2024-01-01,P001,Widget A,120,500,100,7
2024-01-02,P001,Widget A,105,495,100,7
```

## Dashboard Components

The generated dashboard includes four key visualizations:

### 1. Demand Forecast
- Historical demand patterns
- Future demand predictions with Prophet
- Confidence intervals for forecasts

### 2. Current Stock Levels
- Bar chart showing current inventory
- Reorder point markers
- Color-coded status indicators

### 3. Stock Status Overview
- Gauge showing critical inventory items
- Real-time status summary

### 4. Forecast Accuracy
- Model performance metrics
- Accuracy tracking over time

## Stock Status Indicators

| Status | Color | Condition |
|--------|-------|-----------|
| 🟢 **Good** | Green | Stock > Reorder Point + Lead Time Demand |
| 🟡 **Warning** | Yellow | Stock ≤ Reorder Point + Lead Time Demand |
| 🔴 **Critical** | Red | Stock ≤ Reorder Point |

## Dependencies

- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualization
- **prophet**: Time series forecasting
- **flask**: Web framework (for future web interface)
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities

## File Structure

```
inventory-forecast/
│
├── data/
│   └── sample_inventory.csv    # Sample inventory data
├── templates/
│   └── dashboard.html          # Generated dashboard
├── inventory_forecast.py       # Main forecasting script
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Customization

### Adding New Products
Simply add new rows to your CSV file with the same column structure.

### Adjusting Forecast Parameters
Modify the Prophet model parameters in `inventory_forecast.py`:

```python
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=False,
    interval_width=0.80
)
```

### Changing Dashboard Layout
Edit the subplot configuration in the `generate_dashboard()` method:

```python
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Custom Title 1', 'Custom Title 2', ...)
)
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'prophet'**
- Solution: Install dependencies with `pip install -r requirements.txt`

**Empty dashboard or no visualizations**
- Check that your CSV file follows the correct format
- Ensure date column is in YYYY-MM-DD format
- Verify there are at least 3 data points per product

**Prophet warnings**
- These are normal and suppressed by default
- They don't affect functionality

### Getting Help

1. Check the console output for detailed error messages
2. Verify your data format matches the expected structure
3. Ensure all dependencies are properly installed

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Future Enhancements

- 🌐 Web interface with Flask
- 📧 Email alerts for critical stock levels
- 📱 Mobile-responsive dashboard
- 🔄 Real-time data integration
- 📊 Advanced analytics and reporting
- 🤖 Machine learning optimization

---

**Built with ❤️ for better inventory management**