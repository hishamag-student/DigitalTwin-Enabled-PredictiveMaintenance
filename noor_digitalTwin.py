import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load or generate sensor data
# -------------------------------
@st.cache_data
def load_sensor_data():
    timestamps = pd.date_range(start='2025-01-01', periods=200, freq='h')
    components = ['CSP Loop', 'PV Array', 'Salt Tank', 'Pump', 'Heliostat']
    data = []

    for comp in components:
        temp = np.random.uniform(20, 120, len(timestamps))
        vib = np.random.uniform(0, 5, len(timestamps))
        press = np.random.uniform(1, 10, len(timestamps))

        # Special leak event injection for molten salt system
        if comp == "Salt Tank":
            leak_start = 120
            leak_end = 160
            temp[leak_start:leak_end] += np.linspace(0, 80, leak_end - leak_start)   # rising temp
            press[leak_start:leak_end] -= np.linspace(0, 5, leak_end - leak_start)   # dropping pressure
            vib[leak_start:leak_end] += np.linspace(0, 1.5, leak_end - leak_start)   # increased vibration

        df_comp = pd.DataFrame({
            'timestamp': timestamps,
            'component': comp,
            'temperature': temp,
            'vibration': vib,
            'pressure': press
        })
        data.append(df_comp)

    df = pd.concat(data, ignore_index=True)
    return df

# -------------------------------
# Load predictive model
# -------------------------------
def load_model():
    model = RandomForestClassifier()
    return model

# -------------------------------
# Predict health / leak detection
# -------------------------------
def predict_health(model, df):
    # Ensure continuous indexing (avoids KeyError)
    df = df.reset_index(drop=True)
    X = df[['temperature', 'vibration', 'pressure']].copy()

    # Initialize with normal states
    state = np.zeros(len(X), dtype=int)

    # Molten salt leak detection logic
    if df['component'].iloc[0] == 'Salt Tank':
        for i in range(1, len(df)):
            temp_diff = df.loc[i, 'temperature'] - df.loc[i-1, 'temperature']
            pressure_diff = df.loc[i, 'pressure'] - df.loc[i-1, 'pressure']

            # Simple rule-based leak detection
            if df.loc[i, 'temperature'] > 150 or (temp_diff > 10 and pressure_diff < -1):
                state[i] = 2  # Critical Leak Detected
            elif df.loc[i, 'temperature'] > 100 or pressure_diff < -0.5:
                state[i] = 1  # Early Leak Warning
            else:
                state[i] = 0  # Normal
    else:
        # Random simulated health for other components
        state = np.random.choice([0, 1, 2], size=len(X), p=[0.7, 0.2, 0.1])

    return state


# -------------------------------
# Compute KPIs safely
# -------------------------------
def compute_kpis(df):
    if df.empty:
        return pd.Series({'temperature': np.nan}), 0, 0, 0
    latest = df.iloc[-1]
    co2_savings = round(np.random.uniform(0.5, 2.0) * 1000, 0)  # tonnes
    lifetime_extension = round(np.random.uniform(5, 20), 1)     # months
    dispatch_efficiency = round(np.random.uniform(85, 99), 1)   # %
    return latest, co2_savings, lifetime_extension, dispatch_efficiency

# -------------------------------
# Risk heatmap generation
# -------------------------------
def generate_risk_heatmap(df):
    if 'predicted_state' not in df.columns or df.empty:
        return None
    pivot = df.pivot_table(index='timestamp', columns='component', values='predicted_state', fill_value=0)
    return pivot

# -------------------------------
# Streamlit Dashboard
# -------------------------------
st.set_page_config(page_title="Noor Digital Twin Dashboard", layout="wide")
st.title("Noor Ouarzazate Predictive Maintenance Dashboard")

# Image of Noor
st.subheader("Noor Ouarzazate Solar Power Station")
st.image(
    "https://fr.le360.ma/resizer/v2/DWIEVAM4AFF6PLGZFCB7XQUUJM.jpg?auth=9aa2172b342008bb4b96f5bd3c2f5a4130000ee726bf573b99c246664a8edb96&smart=true&width=1216&height=684",
    caption="Noor Ouarzazate Solar Complex, Morocco",
    use_container_width=True
)

# Live webcam
st.subheader("Live Solar Energy Station Stream (EarthCam)")
live_cam_url = "https://www.earthcam.com/cams/solar-energy/"
st.markdown(
    f'<iframe src="{live_cam_url}" width="700" height="500" frameborder="0" allowfullscreen></iframe>',
    unsafe_allow_html=True
)

# Load data
data = load_sensor_data()

# Sidebar filters
st.sidebar.header("Data Filtering")
component = st.sidebar.selectbox("Component", data['component'].unique())
filtered_data = data.loc[data['component'] == component].copy()

# Warn if no data
if filtered_data.empty:
    st.warning(f"No data available for component: {component}")
else:
    # Sensor table
    st.header(f"Sensor Data for {component}")
    st.write(filtered_data.tail())

    # Live KPIs
    st.subheader("Live KPIs")
    latest, co2_savings, lifetime_ext, dispatch_eff = compute_kpis(filtered_data)
    st.metric("Latest Temperature", f"{latest['temperature']:.2f} °C")
    st.metric("Latest Vibration", f"{latest['vibration']:.2f} mm/s")
    st.metric("CO₂ Savings", f"{co2_savings} tonnes")
    st.metric("Equipment Lifetime Extension", f"{lifetime_ext} months")
    st.metric("Dispatch Efficiency", f"{dispatch_eff}%")

    # Sensor trends
    st.subheader("Sensor Trends Over Time")
    fig = px.line(filtered_data, x='timestamp', y=['temperature', 'vibration'])
    st.plotly_chart(fig)

    # Health prediction
    model = load_model()
    filtered_data['predicted_state'] = predict_health(model, filtered_data)
    state_map = {
        0: "Normal",
        1: "Maintenance Recommended",
        2: "Critical Fault / Molten Salt Leak"
    }
    filtered_data['state_label'] = filtered_data['predicted_state'].map(state_map)

    st.subheader("Health State Predictions")
    fig2 = px.scatter(filtered_data, x='timestamp', y='temperature', color='state_label',
                      title="Predicted Maintenance States Over Time")
    st.plotly_chart(fig2)

    # Alerts
    last_state = filtered_data['predicted_state'].iloc[-1]
    if last_state == 2:
        if component == "Salt Tank":
            st.error("🚨 Critical Molten Salt Leak Detected! Immediate shutdown and maintenance required.")
        else:
            st.error("🚨 Critical Fault detected! Immediate maintenance required.")
    elif last_state == 1:
        if component == "Salt Tank":
            st.warning("⚠️ Possible early leak warning detected in molten salt loop.")
        else:
            st.warning("Maintenance recommended soon.", icon="🛠️")
    else:
        st.success("All systems normal.")

    # Risk heatmap
    st.subheader("Component Risk Heatmap")
    heatmap_data = generate_risk_heatmap(filtered_data)
    if heatmap_data is not None:
        fig3 = px.imshow(heatmap_data.T, labels=dict(x="Time", y="Component", color="State"))
        st.plotly_chart(fig3)

    # Download CSV
    st.download_button("Download latest report", filtered_data.to_csv(index=False).encode(),
                       file_name="noor_maintenance_report.csv")

st.caption("This dashboard includes molten salt leak detection for the Salt Tank component using synthetic Digital Twin data.")
