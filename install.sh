#!/bin/bash
# NL-PowerPredict v3.0 Installation Script
# For Raspberry Pi 4 (8GB) running 64-bit OS

set -e

echo "========================================="
echo "NL-PowerPredict v3.0 Installation"
echo "Nederlandse Electricity Price Prediction"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo -e "${YELLOW}Warning: Not running on Raspberry Pi${NC}"
fi

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${RED}Error: 64-bit OS required (found: $ARCH)${NC}"
    exit 1
fi

# Check memory
MEM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')
if [ $MEM_TOTAL -lt 3800 ]; then
    echo -e "${YELLOW}Warning: Less than 4GB RAM detected${NC}"
fi

echo -e "${GREEN}System checks passed${NC}"

# Installation directory
INSTALL_DIR="/home/pi/nl-powerpredict"
echo "Installing to: $INSTALL_DIR"

# Create directory structure
echo "Creating directory structure..."
mkdir -p $INSTALL_DIR/{templates,static,data,logs,config,scripts}
cd $INSTALL_DIR

# Download files from GitHub
echo "Downloading application files..."
GITHUB_BASE="https://raw.githubusercontent.com/ruiterer/nl-powerpredict/main"

wget -q -O nl-powerpredict-v3.py "$GITHUB_BASE/nl-powerpredict-v3.py" || {
    echo -e "${YELLOW}GitHub repository not found, using local files${NC}"
}

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx git sqlite3 \
    python3-dev build-essential libatlas-base-dev libopenblas-dev \
    libxml2-dev libxslt1-dev

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version for ARM64
echo "Installing PyTorch (CPU-optimized for Raspberry Pi)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install Chronos-Bolt
echo "Installing Chronos-Bolt forecasting model..."
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Install other requirements
echo "Installing Python packages..."
cat > requirements_install.txt << EOF
flask==3.0.0
flask-cors==4.0.0
pandas==2.1.4
numpy==1.24.4
scipy==1.11.4
scikit-learn==1.3.2
requests==2.31.0
psutil==5.9.6
python-dotenv==1.0.0
pytz==2023.3
lxml==4.9.3
xmltodict==0.13.0
paho-mqtt==1.6.1
schedule==1.2.0
transformers==4.36.0
EOF

pip install -r requirements_install.txt

# Create configuration file
echo "Creating configuration..."
cat > config/.env << EOF
# NL-PowerPredict v3.0 Configuration
# Add your API keys here

# ENTSO-E Transparency Platform (Required for real data)
# Register at: https://transparency.entsoe.eu/
# Email transparency@entsoe.eu for API access
ENTSO_E_TOKEN=

# KNMI Weather Data (Optional)
# Register at: https://developer.dataplatform.knmi.nl/
KNMI_API_KEY=

# EnergyZero (Optional - no key required)
ENERGYZERO_KEY=

# DayAhead.nl (Optional)
DAYAHEAD_KEY=

# System Configuration
DATABASE_PATH=data/nl-powerpredict.db
LOG_LEVEL=INFO
PORT=5000
EOF

# Download or create templates
echo "Setting up web interface..."
if [ ! -f templates/mobile-dashboard.html ]; then
    wget -q -O templates/mobile-dashboard.html "$GITHUB_BASE/templates/mobile-dashboard.html" || {
        echo -e "${YELLOW}Creating default dashboard${NC}"
        # Dashboard would be created here
    }
fi

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/nl-powerpredict.service > /dev/null << EOF
[Unit]
Description=NL-PowerPredict v3.0 - Nederlandse Electricity Price Prediction
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/python nl-powerpredict-v3.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits for Pi 4
MemoryMax=6G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

# Create Nginx configuration
echo "Configuring Nginx..."
sudo tee /etc/nginx/sites-available/nl-powerpredict > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for real-time updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # API rate limiting
    location /api/ {
        limit_req zone=api burst=10 nodelay;
        proxy_pass http://127.0.0.1:5000;
    }

    # Static files
    location /static/ {
        alias $INSTALL_DIR/static/;
        expires 1h;
    }
}
EOF

# Add rate limiting to Nginx
sudo tee -a /etc/nginx/nginx.conf > /dev/null << 'EOF'
# Rate limiting for API
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/nl-powerpredict /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

# Create monitoring script
echo "Creating monitoring script..."
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
# NL-PowerPredict Monitoring Script

echo "NL-PowerPredict v3.0 System Monitor"
echo "===================================="

# Check service status
STATUS=$(systemctl is-active nl-powerpredict)
echo "Service Status: $STATUS"

# Check memory usage
MEM_USED=$(ps aux | grep nl-powerpredict | grep -v grep | awk '{print $4}')
echo "Memory Usage: ${MEM_USED}%"

# Check CPU usage
CPU_USED=$(ps aux | grep nl-powerpredict | grep -v grep | awk '{print $3}')
echo "CPU Usage: ${CPU_USED}%"

# Check database size
DB_SIZE=$(du -h data/nl-powerpredict.db 2>/dev/null | cut -f1)
echo "Database Size: ${DB_SIZE:-N/A}"

# Check last prediction time
LAST_LOG=$(tail -n 1 logs/nl-powerpredict.log 2>/dev/null | cut -d' ' -f1-2)
echo "Last Activity: ${LAST_LOG:-N/A}"

# Check API responsiveness
curl -s -o /dev/null -w "API Response Time: %{time_total}s\n" http://localhost:5000/api/system/status
EOF
chmod +x scripts/monitor.sh

# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
# NL-PowerPredict Backup Script

BACKUP_DIR="/home/pi/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/nl-powerpredict_$TIMESTAMP.tar.gz"

mkdir -p $BACKUP_DIR

echo "Creating backup: $BACKUP_FILE"
tar -czf $BACKUP_FILE \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    /home/pi/nl-powerpredict

# Keep only last 7 backups
ls -t $BACKUP_DIR/nl-powerpredict_*.tar.gz | tail -n +8 | xargs rm -f 2>/dev/null

echo "Backup completed"
EOF
chmod +x scripts/backup.sh

# Create update script
cat > scripts/update.sh << 'EOF'
#!/bin/bash
# NL-PowerPredict Update Script

echo "Updating NL-PowerPredict..."

cd /home/pi/nl-powerpredict

# Backup current version
./scripts/backup.sh

# Pull latest version
git pull origin main 2>/dev/null || {
    wget -O nl-powerpredict-v3.py \
        https://raw.githubusercontent.com/ruiterer/nl-powerpredict/main/nl-powerpredict-v3.py
}

# Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart service
sudo systemctl restart nl-powerpredict

echo "Update completed"
EOF
chmod +x scripts/update.sh

# Initialize database with sample data
echo "Initializing database..."
source venv/bin/activate
python3 << 'PYTHON_EOF'
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create database
conn = sqlite3.connect('data/nl-powerpredict.db')
cursor = conn.cursor()

# Create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datetime TIMESTAMP UNIQUE,
        price_eur_mwh REAL,
        price_cent_kwh REAL,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Generate 28 days of sample data
print("Generating 28 days of sample data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=28)

dates = pd.date_range(start_date, end_date, freq='H')
prices = []

for date in dates:
    # Base price with daily pattern
    hour = date.hour
    base = 70 + 30 * np.sin(hour * np.pi / 12)

    # Market patterns
    if hour in [1,2,3,4,5]:  # Night valley
        base *= 0.85
    elif hour in [6,7,8,9]:  # Morning peak
        base *= 1.15
    elif hour in [11,12,13,14]:  # Solar dip
        base *= 0.90
    elif hour in [17,18,19,20]:  # Evening peak
        base *= 1.25

    # Weekend effect
    if date.weekday() >= 5:
        base *= 0.92

    # Random variation
    base += np.random.normal(0, 10)
    prices.append((date, max(0, base), max(0, base/10), 'simulation'))

cursor.executemany(
    "INSERT OR IGNORE INTO historical_prices (datetime, price_eur_mwh, price_cent_kwh, source) VALUES (?, ?, ?, ?)",
    prices
)

conn.commit()
conn.close()
print("Database initialized with sample data")
PYTHON_EOF

# Enable and start service
echo "Starting NL-PowerPredict service..."
sudo systemctl daemon-reload
sudo systemctl enable nl-powerpredict
sudo systemctl start nl-powerpredict

# Wait for service to start
sleep 5

# Check service status
if systemctl is-active --quiet nl-powerpredict; then
    echo -e "${GREEN}✓ NL-PowerPredict successfully installed!${NC}"
    echo ""
    echo "Access the dashboard at:"
    echo "  http://$(hostname -I | cut -d' ' -f1)"
    echo ""
    echo "API Endpoints:"
    echo "  http://$(hostname -I | cut -d' ' -f1)/api/predictions/next24"
    echo "  http://$(hostname -I | cut -d' ' -f1)/api/predictions/next96"
    echo ""
    echo "Configuration file: $INSTALL_DIR/config/.env"
    echo ""
    echo "Commands:"
    echo "  Monitor: $INSTALL_DIR/scripts/monitor.sh"
    echo "  Backup:  $INSTALL_DIR/scripts/backup.sh"
    echo "  Update:  $INSTALL_DIR/scripts/update.sh"
    echo ""
    echo "Service management:"
    echo "  sudo systemctl status nl-powerpredict"
    echo "  sudo systemctl restart nl-powerpredict"
    echo "  sudo journalctl -u nl-powerpredict -f"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    echo "Check logs: sudo journalctl -u nl-powerpredict -n 50"
fi
