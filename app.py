import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Forest Fire Risk", layout="wide")
st.title("🔥 Forest Fire Risk Prediction")

st.write("Enter weather data to predict Fire Weather Index (FWI)")

# Train model
np.random.seed(42)
X = pd.DataFrame({
    'Temperature': np.random.uniform(10, 45, 500),
    'Humidity': np.random.uniform(10, 100, 500),
    'Wind_Speed': np.random.uniform(0, 30, 500),
    'Rain': np.random.uniform(0, 50, 500)
})
y = np.random.uniform(0, 100, 500)
model = RandomForestRegressor()
model.fit(X, y)

# UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Weather Input")
    temp = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
    rain = st.slider("Rain (mm)", 0, 100, 5)

with col2:
    st.subheader("📊 Prediction Result")
    if st.button("🔥 Predict Fire Risk", use_container_width=True):
        pred = model.predict([[temp, humidity, wind, rain]])[0]
        
        if pred > 70:
            st.error(f"🔴 HIGH RISK - FWI: {pred:.2f}")
            st.warning("⚠️ Immediate action required!")
        elif pred > 40:
            st.warning(f"🟡 MODERATE RISK - FWI: {pred:.2f}")
            st.info("📌 Be cautious. Monitor conditions.")
        else:
            st.success(f"🟢 LOW RISK - FWI: {pred:.2f}")
            st.info("✅ No immediate threat.")

st.markdown("---")
st.caption("Powered by Random Forest ML • M.M.E.S. College Project")
