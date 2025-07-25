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
   git clone https://github.com/Mikerstrong/inventory-forecast.git
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

## 🐳 Docker & Portainer Deployment (Recommended)

Deploy the entire system using Docker and manage it with Portainer's web interface. This is the **recommended production deployment method**.

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mikerstrong/inventory-forecast.git
   cd inventory-forecast
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker compose up -d
   ```

3. **Access the applications:**
   - 📊 **Inventory Dashboard**: [http://localhost:7342](http://localhost:7342)
   - 🐳 **Portainer Management**: [http://localhost:9000](http://localhost:9000)

### Portainer Deployment Methods

#### Method 1: Deploy Portainer + App Together (Recommended)

The provided `docker-compose.yml` deploys both services together:

```bash
# Clone and deploy
git clone https://github.com/Mikerstrong/inventory-forecast.git
cd inventory-forecast
docker compose up -d

# Verify deployment
docker ps
```

#### Method 2: Deploy via Portainer Stacks

1. **Install Portainer first:**
   ```bash
   docker volume create portainer_data
   docker run -d -p 9000:9000 -p 9443:9443 --name portainer --restart=always \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v portainer_data:/data portainer/portainer-ce:latest
   ```

2. **Access Portainer**: Visit [http://localhost:9000](http://localhost:9000)

3. **Create a Stack:**
   - Go to **Stacks** → **Add Stack**
   - Name: `inventory-forecast`
   - Copy the docker-compose.yml content (excluding the Portainer service)
   - Click **Deploy the stack**

#### Method 3: Remote Repository Deployment

Deploy directly from GitHub in Portainer:

1. **In Portainer**, go to **Stacks** → **Add Stack**
2. **Choose "Repository"**
3. **Repository URL**: `https://github.com/Mikerstrong/inventory-forecast`
4. **Compose file path**: `docker-compose.yml`
5. **Click Deploy**

### Portainer Stack Configuration

For Portainer deployment, use this stack configuration:

```yaml
version: '3.8'
services:
  inventory-app:
    image: python:3.11-slim
    container_name: inventory-forecast-app
    working_dir: /app
    volumes:
      - ./:/app
    ports:
      - "7342:7342"
    command: >
      sh -c "pip install --upgrade pip && \
             pip install -r requirements.txt && \
             streamlit run inventory_forecast.py --server.port 7342 --server.address 0.0.0.0"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    networks:
      - inventory-network

networks:
  inventory-network:
    driver: bridge
```

#### Minimal Portainer Stack Example

If you already have Portainer running, use this minimal stack file for your inventory app:

```yaml
version: '3.8'
services:
  inventory-app:
    image: python:3.11-slim
    container_name: inventory-forecast-app
    working_dir: /app
    volumes:
      - ./:/app
    ports:
      - "7342:7342"
    command: >
      sh -c "pip install --upgrade pip && \
             pip install -r requirements.txt && \
             streamlit run inventory_forecast.py --server.port 7342 --server.address 0.0.0.0"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    networks:
      - inventory-network

networks:
  inventory-network:
    driver: bridge
```

Paste this into Portainer's stack editor (Stacks → Add Stack) to deploy only the inventory app.

### Managing with Portainer

After deployment, use Portainer's web interface to:

#### 📊 **Dashboard Management**
- **View Containers**: Monitor app health and resource usage
- **Logs**: Real-time log viewing for debugging
- **Console Access**: Terminal access to containers
- **Resource Monitoring**: CPU, memory, and network usage

#### 🔄 **Application Updates**
- **Stack Updates**: Redeploy when code changes
- **Image Updates**: Pull latest base images
- **Configuration Changes**: Update environment variables

#### 🛠️ **Maintenance Tasks**
- **Backup Volumes**: Backup inventory data
- **Scale Services**: Increase/decrease replicas
- **Network Management**: Configure container networking

### Environment Variables

Configure the app through Portainer environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Server bind address |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |

### Production Deployment Tips

#### 🔒 **Security**
- Use HTTPS with reverse proxy (nginx/traefik)
- Change default Portainer admin password
- Configure firewall rules for ports 8501 and 9000

#### 📈 **Performance**
- Allocate sufficient memory (2GB+ recommended)
- Use SSD storage for better I/O performance
- Monitor container resource usage in Portainer

#### 🔄 **Backup & Recovery**
```bash
# Backup Portainer data
docker run --rm -v portainer_data:/data -v $(pwd):/backup alpine tar czf /backup/portainer-backup.tar.gz -C /data .

# Restore Portainer data
docker run --rm -v portainer_data:/data -v $(pwd):/backup alpine tar xzf /backup/portainer-backup.tar.gz -C /data
```

### Troubleshooting Portainer Deployment

1. **Port Conflicts**: Change ports in docker-compose.yml if needed
2. **Permission Issues**: Ensure Docker daemon is accessible
3. **Container Won't Start**: Check logs in Portainer container view
4. **Update Issues**: Use Portainer's "Recreate" option for clean redeployment

---

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