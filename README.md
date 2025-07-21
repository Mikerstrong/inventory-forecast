# Inventory Management and Forecasting System

A Python-based inventory management system that provides demand forecasting and interactive dashboard visualization using Prophet and Plotly.

## Features

- 📊 **Interactive Dashboard**: Real-time inventory status visualization
- 🔮 **Demand Forecasting**: Uses Facebook Prophet for accurate demand prediction
- 🚦 **Status Indicators**: Color-coded inventory levels (Green/Yellow/Red)
- 📈 **Trend Analysis**: Historical demand patterns and future projections
- 🎯 **Reorder Alerts**: Automatic low-stock notifications
- 📱 **Responsive Design**: Works on desktop and mobile devices

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

### Running the Forecasting System

Execute the main script to generate the dashboard:

```bash
python inventory_forecast.py
```

This will:
- Load inventory data from `data/sample_inventory.csv`
- Generate demand forecasts for all products
- Create an interactive HTML dashboard at `templates/dashboard.html`

### Viewing the Dashboard

After running the script, open the generated dashboard:

```bash
# Open in your default browser (macOS)
open templates/dashboard.html

# Open in your default browser (Linux)
xdg-open templates/dashboard.html

# Open in your default browser (Windows)
start templates/dashboard.html
```

## File Structure

```
inventory-forecast/
├── requirements.txt              # Python dependencies
├── inventory_forecast.py         # Main forecasting script
├── README.md                    # This documentation
├── data/
│   └── sample_inventory.csv     # Sample inventory data
└── templates/
    └── dashboard.html           # Generated HTML dashboard
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

### 2. Stock Levels by Product
- Current stock vs reorder points
- Color-coded status bars (Red: Critical, Orange: Low, Green: Good, Blue: High)
- Visual reorder point reference line

### 3. Daily Demand Trends
- Historical demand patterns for top products
- Time series visualization
- Trend identification

### 4. Demand Forecasting
- Prophet-generated demand forecasts
- Confidence intervals
- 30-day forward predictions

### 5. Status Distribution
- Pie chart of inventory status categories
- Quick overview of system health

## Status Color Coding

- 🔴 **Critical (Red)**: Stock at or below reorder point
- 🟠 **Low (Orange)**: Stock within 1.5x of reorder point
- 🔵 **High (Blue)**: Stock at 90%+ of maximum capacity
- 🟢 **Good (Green)**: Healthy stock levels

## Customization

### Adding New Products

Add new entries to `data/sample_inventory.csv` with the required columns. The system automatically detects and processes new products.

### Modifying Forecast Parameters

Edit the `forecast_demand()` method in `inventory_forecast.py`:

```python
model = Prophet(
    daily_seasonality=False,    # Enable/disable daily patterns
    weekly_seasonality=True,    # Enable/disable weekly patterns
    yearly_seasonality=False    # Enable/disable yearly patterns
)
```

### Changing Dashboard Layout

Modify the `create_dashboard()` method to adjust:
- Number of subplots
- Chart types and configurations
- Color schemes and themes

## Dependencies

- **pandas** (≥1.5.0): Data manipulation and analysis
- **plotly** (≥5.0.0): Interactive visualization
- **prophet** (≥1.1.0): Time series forecasting
- **flask** (≥2.0.0): Web framework (for future web integration)

## Troubleshooting

### Common Issues

1. **Prophet Installation Error**:
   ```bash
   # On macOS with Apple Silicon
   conda install -c conda-forge prophet
   
   # Alternative installation
   pip install pystan==2.19.1.1
   pip install prophet
   ```

2. **Missing Data Error**:
   - Ensure `data/sample_inventory.csv` exists
   - Verify CSV has all required columns
   - Check date format (YYYY-MM-DD)

3. **Dashboard Not Loading**:
   - Check console for JavaScript errors
   - Ensure all Plotly CDN resources load
   - Try opening in different browser

### Performance Optimization

For large datasets:
- Limit the number of products in trend analysis
- Reduce forecast periods
- Use data sampling for visualization

## Future Enhancements

- 🌐 Web-based Flask interface
- 📧 Email alerts for critical stock levels
- 📊 Advanced analytics and reporting
- 🔄 Real-time data integration
- 📱 Mobile app development
- 🤖 Machine learning optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Happy forecasting!** 🚀📊