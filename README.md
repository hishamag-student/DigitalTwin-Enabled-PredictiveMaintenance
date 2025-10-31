# DigitalTwin-Enabled Predictive Maintenance Dashboard

This project implements a **Digital Twin system** for the Noor Ouarzazate Solar Complex in Morocco.  
It integrates **IoT data**, **machine learning**, and **real-time visualization** to predict equipment faults — including **molten salt leaks** — and support proactive maintenance decisions.

---

## 🚀 Features
- Real-time monitoring of temperature, vibration, and pressure.  
- ML-based condition classification: *Normal*, *Maintenance Advised*, *Critical Fault / Molten Salt Leak*.  
- Dynamic Streamlit dashboard with heatmaps, charts, and KPIs.  
- CO₂ saving and life-extension estimation.  
- MQTT broker-ready for live data integration.  

---

## 🛠️ Installation

```bash
git clone https://github.com/hishamag-student/DigitalTwin-Enabled-PredictiveMaintenance.git
cd DigitalTwin-Enabled-PredictiveMaintenance
pip install -r requirements.txt
streamlit run noor_digitalTwin.py
