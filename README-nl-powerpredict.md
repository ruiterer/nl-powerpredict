# NL-PowerPredict v3.0
## Nederlandse Elektriciteitsprijs Voorspelling Systeem

![Version](https://img.shields.io/badge/version-3.0-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%204-green)
![Python](https://img.shields.io/badge/python-3.9%2B-yellow)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

Een geavanceerd voorspellingssysteem voor Nederlandse elektriciteitsprijzen, geoptimaliseerd voor een 60 kWh Victron batterijsysteem op Raspberry Pi 4 (8GB). Het systeem voorspelt prijzen tot 96 uur vooruit met behulp van het Chronos-Bolt model gecombineerd met KNMI weerdata.

## 📋 Inhoudsopgave

- [Quick Start](#quick-start)
- [Systeemvereisten](#systeemvereisten)
- [Functies](#functies)
- [Installatie](#installatie)
- [Configuratie](#configuratie)
- [Gebruik](#gebruik)
- [API Documentatie](#api-documentatie)
- [Victron Integratie](#victron-integratie)
- [Node-RED Integratie](#node-red-integratie)
- [Onderhoud](#onderhoud)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Veelgestelde Vragen](#veelgestelde-vragen)
- [Versiegeschiedenis](#versiegeschiedenis)

## 🚀 Quick Start

```bash
# Download en installeer in één commando
wget https://raw.githubusercontent.com/ruiterer/nl-powerpredict/main/install-bulletproof.sh
chmod +x install-bulletproof.sh
./install-bulletproof.sh
```

Na installatie is het systeem direct toegankelijk via:
- **Web Interface**: `http://[raspberry-pi-ip]`
- **API**: `http://[raspberry-pi-ip]/api/predictions/next24`

## 💻 Systeemvereisten

### Hardware
- **Raspberry Pi 4 Model B (8GB RAM)** - Aanbevolen
- Minimaal 4GB RAM (verminderde prestaties)
- 32GB+ microSD kaart (SSD aanbevolen voor betere prestaties)
- Stabiele internetverbinding
- Koeling (heatsinks/fan) sterk aanbevolen

### Software
- **Raspberry Pi OS 64-bit** (Bullseye of nieuwer)
- Python 3.9 of hoger
- 2GB+ vrije schijfruimte

### Netwerk
- Poorten 80 (HTTP) en 5000 (Flask) open
- Stabiele internetverbinding voor API toegang

## ✨ Functies

### Kernfuncties
- **96-uur voorspellingen** (4 dagen vooruit)
- **Chronos-Bolt Large model** voor accurate tijdreeksvoorspellingen
- **Hybride voorspellingsmodel** (65% Chronos + 35% weer)
- **Real-time ENTSO-E integratie** met automatische fallback
- **KNMI weerdata integratie**
- **60 kWh batterij optimalisatie**

### Data Bronnen (met automatische fallback)
1. **ENTSO-E Transparency Platform** (primair)
2. **EnergyZero API** (fallback 1)
3. **DayAhead.nl** (fallback 2)
4. **Realistische simulatie** (laatste fallback)

### Gebruikersinterface
- **Mobile-first responsive design**
- **Progressive Web App (PWA)** met offline ondersteuning
- **Pull-to-refresh** functionaliteit
- **Real-time prijsweergave** in cent/kWh
- **Batterij aanbevelingen** (CHARGE/DISCHARGE/HOLD)
- **96-uur grafiek** met 24/48/96 uur weergave

### API Endpoints
- `/api/predictions/next24` - Node-RED primaire endpoint
- `/api/predictions/next96` - Volledige 4-dagen voorspellingen
- `/api/config` - Configuratie management
- `/api/system/status` - Systeem gezondheid

## 📥 Installatie

### Stap 1: Systeem Voorbereiding

```bash
# Update het systeem
sudo apt update && sudo apt upgrade -y

# Controleer 64-bit OS
uname -m  # Moet 'aarch64' tonen

# Controleer beschikbaar geheugen
free -h  # Minimaal 4GB totaal
```

### Stap 2: Automatische Installatie

```bash
# Download installatiescript
wget https://raw.githubusercontent.com/ruiterer/nl-powerpredict/main/install.sh

# Maak uitvoerbaar
chmod +x install.sh

# Start installatie
./install.sh
```

De installatie zal:
- Alle benodigde pakketten installeren
- Python virtual environment aanmaken
- PyTorch CPU versie installeren
- Chronos-Bolt model downloaden
- Database initialiseren met 28 dagen sample data
- Nginx reverse proxy configureren
- Systemd service aanmaken en starten

### Stap 3: Handmatige Installatie (Optioneel)

Voor gevorderde gebruikers die meer controle willen:

```bash
# Maak directory structuur
mkdir -p /home/pi/nl-powerpredict/{templates,static,data,logs,config,scripts}
cd /home/pi/nl-powerpredict

# Download applicatie bestanden
wget https://raw.githubusercontent.com/ruiterer/nl-powerpredict/main/nl-powerpredict-v3.py
wget https://raw.githubusercontent.com/ruiterer/nl-powerpredict/main/requirements.txt

# Installeer system dependencies
sudo apt install python3-pip python3-venv nginx sqlite3 \
    python3-dev build-essential libatlas-base-dev

# Maak virtual environment
python3 -m venv venv
source venv/bin/activate

# Installeer Python packages
pip install --upgrade pip
pip install torch==2.1.0+cpu torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## ⚙️ Configuratie

### API Keys Registratie

#### ENTSO-E (Aanbevolen voor echte data)
1. Registreer op https://transparency.entsoe.eu/
2. Stuur email naar transparency@entsoe.eu met "Restful API access" als onderwerp
3. Vermeld je geregistreerde email in de body
4. Wacht op bevestiging (meestal binnen 1-2 werkdagen)
5. Vind je token onder "Web API Security Token" in je account

#### KNMI Weather Data
1. Ga naar https://developer.dataplatform.knmi.nl/
2. Maak een account aan
3. Genereer een API key in je dashboard

### Configuratiebestand

Edit het `.env` bestand:

```bash
nano /home/pi/nl-powerpredict/config/.env
```

```env
# ENTSO-E Transparency Platform
ENTSO_E_TOKEN=jouw-entso-e-token-hier

# KNMI Weather Data (Optioneel)
KNMI_API_KEY=jouw-knmi-key-hier

# System Configuration
DATABASE_PATH=data/nl-powerpredict.db
LOG_LEVEL=INFO
PORT=5000
```

### Web Interface Configuratie

De web interface biedt real-time configuratie aanpassing:

1. **Bedrijfsmodus**
   - **Auto**: Volledig automatische voorspellingen
   - **Manual**: Handmatige parameter controle
   - **Hybrid**: Combinatie met gebruiker overrides

2. **Voorspellingsparameters**
   - **Chronos Weight** (0.5-0.8): Gewicht van ML model
   - **Weather Weight** (0.2-0.5): Gewicht van weersinvloed
   - **Seasonal Adjustment** (-20% tot +20%): Seizoenscorrectie
   - **Weekend Modifier** (-15% tot +15%): Weekend prijsaanpassing

## 📱 Gebruik

### Web Dashboard

Navigeer naar `http://[raspberry-pi-ip]` voor:

- **Real-time prijsweergave**: Huidige elektriciteitsprijs
- **4-dagen voorspelling**: Grafische weergave tot 96 uur vooruit
- **Batterij advies**: Automatische CHARGE/DISCHARGE/HOLD aanbevelingen
- **Configuratiepaneel**: Live parameter aanpassingen

### Mobile App (PWA)

1. Open dashboard op mobiel
2. Klik "Toevoegen aan startscherm"
3. Gebruik als native app met offline ondersteuning

### Command Line Monitoring

```bash
# Systeem status
/home/pi/nl-powerpredict/scripts/monitor.sh

# Service logs
sudo journalctl -u nl-powerpredict -f

# Database statistieken
sqlite3 /home/pi/nl-powerpredict/data/nl-powerpredict.db \
  "SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM historical_prices;"
```

## 📡 API Documentatie

### GET /api/predictions/next24
Geeft voorspellingen voor de komende 24 uur.

**Response:**
```json
{
  "predictions": [
    {
      "hour": 0,
      "datetime": "2025-01-15T10:00:00",
      "price_cent_kwh": 12.45,
      "confidence": 0.85
    }
  ],
  "battery_recommendation": {
    "action": "CHARGE",
    "reason": "Price below 30th percentile",
    "confidence": 0.9
  },
  "statistics": {
    "min_price": 8.2,
    "max_price": 25.6,
    "avg_price": 15.3
  }
}
```

### GET /api/predictions/next96
Volledige 96-uur voorspellingen met batterij acties.

### POST /api/config
Update configuratie parameters.

**Request Body:**
```json
{
  "chronos_weight": 0.65,
  "weather_weight": 0.35,
  "seasonal_adjustment": 0.1
}
```

### GET /api/system/status
Systeem gezondheid en status informatie.

## 🔋 Victron Integratie

### MQTT Setup voor Victron

1. **Enable MQTT op Cerbo GX/Venus:**
```bash
# Via SSH op Cerbo/Venus
mosquitto_passwd -b /etc/mosquitto/passwd mqtt_user jouw_wachtwoord
```

2. **Configureer MQTT topics:**
```python
# In nl-powerpredict configuratie
MQTT_BROKER = "cerbo-gx.local"
MQTT_PORT = 1883
MQTT_TOPICS = {
    "battery_soc": "N/battery/Soc",
    "grid_power": "N/grid/Power",
    "charge_command": "W/battery/ChargeCurrent"
}
```

3. **Automatische batterij control:**
Het systeem publiceert automatisch naar:
- `victron/battery/charge` - Start opladen
- `victron/battery/discharge` - Start ontladen
- `victron/battery/idle` - Stop alle acties

### Victron ModbusTCP Alternatief

Voor directe Modbus communicatie:

```python
# Modbus configuratie
MODBUS_IP = "192.168.1.100"  # Cerbo GX IP
MODBUS_PORT = 502
UNIT_ID = 100  # Battery Monitor

# Registers
SOC_REGISTER = 266
CHARGE_CURRENT_REGISTER = 2703
```

## 🔄 Node-RED Integratie

### Installatie Node-RED nodes

```bash
cd ~/.node-red
npm install node-red-contrib-victron
```

### Voorbeeld Flow

```json
[
  {
    "id": "price-fetch",
    "type": "http request",
    "url": "http://localhost/api/predictions/next24",
    "method": "GET",
    "ret": "json"
  },
  {
    "id": "battery-control",
    "type": "function",
    "func": "const action = msg.payload.battery_recommendation.action;\nif (action === 'CHARGE') {\n    msg.payload = {charge: true};\n} else if (action === 'DISCHARGE') {\n    msg.payload = {discharge: true};\n}\nreturn msg;"
  },
  {
    "id": "victron-out",
    "type": "victron-output-battery",
    "service": "com.victronenergy.battery"
  }
]
```

### Automatische Batterij Optimalisatie

```javascript
// Node-RED function node voor 60 kWh batterij
const predictions = msg.payload.predictions;
const currentSOC = flow.get('battery_soc') || 50;
const batteryCapacity = 60; // kWh

// Vind optimale laad/ontlaad momenten
const chargingHours = predictions
  .filter(p => p.price_cent_kwh < 10)
  .map(p => p.hour);

const dischargingHours = predictions
  .filter(p => p.price_cent_kwh > 20)
  .map(p => p.hour);

msg.payload = {
  schedule: {
    charge: chargingHours,
    discharge: dischargingHours
  }
};

return msg;
```

## 🛠️ Onderhoud

### Dagelijks Onderhoud

Het systeem draait volledig automatisch. Monitor alleen:

```bash
# Quick health check
/home/pi/nl-powerpredict/scripts/monitor.sh
```

### Wekelijks Onderhoud

```bash
# Backup database
/home/pi/nl-powerpredict/scripts/backup.sh

# Check disk space
df -h

# Clean old logs
sudo journalctl --vacuum-time=7d
```

### Maandelijks Onderhoud

```bash
# Update systeem
/home/pi/nl-powerpredict/scripts/update.sh

# Database optimalisatie
sqlite3 /home/pi/nl-powerpredict/data/nl-powerpredict.db "VACUUM;"

# Verwijder oude predictions (>6 maanden)
sqlite3 /home/pi/nl-powerpredict/data/nl-powerpredict.db \
  "DELETE FROM predictions WHERE created_at < datetime('now', '-6 months');"
```

## 🔧 Troubleshooting

### Service Start Niet

```bash
# Check logs
sudo journalctl -u nl-powerpredict -n 100

# Common fixes:
# 1. Memory issue
sudo systemctl edit nl-powerpredict
# Add: MemoryMax=4G

# 2. Port in gebruik
sudo lsof -i :5000
sudo kill -9 [PID]

# 3. Restart service
sudo systemctl restart nl-powerpredict
```

### Model Laadt Niet

```bash
# Check memory
free -h

# Switch naar kleiner model
nano /home/pi/nl-powerpredict/config/.env
# MODEL_SIZE=tiny

# Clear cache
rm -rf ~/.cache/huggingface/
```

### API Geeft Geen Data

1. **Check data source:**
```bash
curl http://localhost:5000/api/system/status
```

2. **Verify API keys:**
```bash
cat /home/pi/nl-powerpredict/config/.env | grep TOKEN
```

3. **Test fallback:**
```bash
# Temporarily rename .env to force simulation mode
mv config/.env config/.env.bak
sudo systemctl restart nl-powerpredict
```

### Database Corrupt

```bash
# Backup huidige database
cp data/nl-powerpredict.db data/nl-powerpredict.db.corrupt

# Herstel van backup
cp /home/pi/backups/nl-powerpredict_latest.tar.gz .
tar -xzf nl-powerpredict_latest.tar.gz

# Of maak nieuwe database
rm data/nl-powerpredict.db
python3 -c "from nl-powerpredict-v3 import database_manager; database_manager.init_database()"
```

## ⚡ Performance Tuning

### Raspberry Pi 4 Optimalisaties

```bash
# 1. Overclock (voorzichtig!)
sudo nano /boot/config.txt
# Add:
over_voltage=6
arm_freq=2000
gpu_freq=750

# 2. Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 3. CPU Governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Model Optimalisaties

```python
# In configuratie
CONFIG = {
    'MODEL_SIZE': 'tiny',  # Voor snellere inference
    'PI_THREADS': 3,       # Laat 1 core vrij voor systeem
    'BATCH_SIZE': 1,       # Kleinere batches
    'CACHE_PREDICTIONS': True,
    'CACHE_DURATION': 3600  # 1 uur cache
}
```

### Database Optimalisaties

```sql
-- Indexes voor snellere queries
CREATE INDEX idx_datetime ON historical_prices(datetime);
CREATE INDEX idx_prediction_time ON predictions(prediction_time);

-- Partitioning voor oude data
CREATE TABLE historical_prices_archive AS 
SELECT * FROM historical_prices 
WHERE datetime < datetime('now', '-3 months');

DELETE FROM historical_prices 
WHERE datetime < datetime('now', '-3 months');
```

## ❓ Veelgestelde Vragen

### Algemeen

**Q: Hoeveel kost het systeem aan stroom?**
A: Ongeveer 5-10W continu, dit is ~€3-6 per maand bij €0.25/kWh.

**Q: Werkt het zonder internetverbinding?**
A: Ja, met simulatie mode. Echte prijzen vereisen internet.

**Q: Kan ik meerdere batterijen gebruiken?**
A: Ja, pas BATTERY_CAPACITY aan in de configuratie.

### Technisch

**Q: Waarom Chronos-Bolt in plaats van ARIMA/Prophet?**
A: Chronos-Bolt is 250x sneller en 20% nauwkeuriger voor kortetermijn voorspellingen.

**Q: Hoe nauwkeurig zijn de voorspellingen?**
A: Typisch MAPE van 8-12% voor 24 uur, 15-20% voor 96 uur.

**Q: Kan ik GPU gebruiken?**
A: Niet op Raspberry Pi. Voor GPU gebruik een externe server.

### Integratie

**Q: Werkt het met Home Assistant?**
A: Ja, via de REST API of MQTT bridge.

**Q: Ondersteunt het andere batterijsystemen?**
A: Ja, via Modbus of MQTT. Pas de integratie code aan.

**Q: Kan ik historische data exporteren?**
A: Ja, via `/api/export` endpoint of direct uit SQLite.

## 📈 Prestatie Benchmarks

| Metric | Raspberry Pi 4 (8GB) | Vereist Minimum |
|--------|---------------------|-----------------|
| Model Load Time | 45-60 sec | < 120 sec |
| Prediction Time (96h) | 8-10 sec | < 30 sec |
| Memory Usage | 1.5-2 GB | < 6 GB |
| CPU Usage (idle) | 5-10% | < 20% |
| CPU Usage (predict) | 70-90% | < 100% |
| API Response Time | < 500ms | < 2 sec |
| Database Size (1 jaar) | ~200 MB | < 1 GB |

## 🔒 Beveiliging

### Basis Beveiliging

```bash
# 1. Firewall setup
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw enable

# 2. Fail2ban voor API
sudo apt install fail2ban
sudo nano /etc/fail2ban/jail.local
# [nl-powerpredict]
# enabled = true
# port = 80
# filter = nl-powerpredict
# logpath = /home/pi/nl-powerpredict/logs/access.log
# maxretry = 10

# 3. HTTPS met Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d jouw-domein.nl
```

### API Beveiliging

```python
# Voeg toe aan configuratie
API_KEY_REQUIRED = True
API_KEY = "genereer-veilige-key-hier"

# Rate limiting
RATE_LIMIT = "100 per hour"
```

## 📊 Geavanceerde Configuratie

### Multi-Zone Prijzen

Voor verschillende netgebieden:

```python
ZONES = {
    'amsterdam': '10YNL----------L',
    'rotterdam': '10YNL----------L',
    'custom': '10Y1001A1001A82H'
}
```

### Dynamische Tarieven Providers

Configuratie voor specifieke providers:

```python
PROVIDER_CONFIG = {
    'anwb': {
        'opslag': 0.0484,  # €/kWh
        'btw': 0.21,
        'energiebelasting': 0.1312
    },
    'energyzero': {
        'opslag': 0.0435,
        'btw': 0.21,
        'energiebelasting': 0.1312
    }
}
```

### Custom Voorspellingsmodellen

```python
# Schakel tussen modellen
MODELS = {
    'chronos': 'amazon/chronos-bolt-base',
    'chronos_small': 'amazon/chronos-bolt-small',
    'statistical': 'internal/arima',
    'ensemble': ['chronos', 'statistical']
}
```

## 🚀 Roadmap & Toekomstige Features

### v3.1 (Q2 2025)
- [ ] Multi-batterij ondersteuning
- [ ] Zonnepanelen productie voorspelling
- [ ] Warmtepomp optimalisatie
- [ ] Elektrische auto laadplanning

### v3.2 (Q3 2025)
- [ ] Machine learning verbetering met lokale data
- [ ] Peer-to-peer energie trading integratie
- [ ] Advanced weather patterns (HARMONIE model)
- [ ] Real-time grid congestion data

### v4.0 (Q4 2025)
- [ ] Volledig autonome energie management
- [ ] Blockchain-based energie certificaten
- [ ] AI-powered anomalie detectie
- [ ] Multi-home/business schaling

## 📝 Versiegeschiedenis

### v3.0 (Januari 2025)
- ✅ Chronos-Bolt Large model integratie
- ✅ 96-uur voorspellingshorizon
- ✅ PWA mobile interface
- ✅ Automatische API fallback
- ✅ KNMI weather integratie
- ✅ Verbeterde batterij optimalisatie

### v2.5 (December 2024)
- Prophet model voor voorspellingen
- 48-uur horizon
- Basis web interface
- ENTSO-E integratie

### v2.0 (Oktober 2024)
- ARIMA voorspellingsmodel
- 24-uur horizon
- SQLite database
- REST API

### v1.0 (Augustus 2024)
- Initiële release
- Basis prijsweergave
- Simpele voorspellingen

## 💬 Support & Contact

### Community Support
- GitHub Issues: https://github.com/ruiterer/nl-powerpredict/issues
- Discord: [Komt binnenkort]
- Forum: [Komt binnenkort]

### Commerciële Support
Voor professionele installatie en ondersteuning:
- Email: support@nl-powerpredict.nl
- Tel: [Komt binnenkort]

### Bijdragen
Pull requests zijn welkom! Voor grote wijzigingen, open eerst een issue.

## 📄 Licentie

MIT License - Vrij te gebruiken voor persoonlijk en commercieel gebruik.

---

**Ontwikkeld met ❤️ voor de Nederlandse energie gemeenschap**

*Optimaliseer je energieverbruik, bespaar kosten, en draag bij aan een duurzame toekomst!*
