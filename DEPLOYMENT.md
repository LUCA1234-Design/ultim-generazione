# Deployment (AWS Tokyo Colocation Helper)

## Obiettivo
Ridurre latenza verso Binance Futures deployando il bot in `ap-northeast-1` (Tokyo).

## EC2 consigliate
- **t3.medium**: ambiente iniziale / paper trading
- **c6i.xlarge**: produzione con più simboli e workload più intensivo

## Variabili ambiente principali
```bash
export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."
export DB_BACKEND="sqlite"   # oppure timescaledb
export DB_TIMESCALE_URL="postgresql://user:pass@host:5432/trading_db"
```

## Esecuzione con systemd
1. Copia il repo in `/opt/ultim-generazione`
2. Crea venv e installa dipendenze
3. Crea `/etc/systemd/system/ultim-generazione.service`:

```ini
[Unit]
Description=V18 Agentic AI Trading Bot
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ultim-generazione
ExecStart=/opt/ultim-generazione/.venv/bin/python /opt/ultim-generazione/main.py
Restart=always
RestartSec=5
User=ubuntu
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

4. Avvio:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ultim-generazione
sudo systemctl start ultim-generazione
sudo systemctl status ultim-generazione
```

## Verifica latenza
Controlla heartbeat Telegram: deve includere RTT e WS delay.
Se `p95 RTT > 100ms`, valutare upgrade istanza/networking.
