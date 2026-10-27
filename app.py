import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")

@st.cache_resource
def load():
    model = joblib.load("crop_yield_model.joblib")
    features = joblib.load("feature_columns.joblib")
    hist = pd.read_csv("model_data.csv")
    return model, features, hist

model, features, hist = load()

st.title("🌾 Crop Yield Predictor (India)")
st.caption("Machine-learning model trained on FAO crop data and NASA POWER weather data.")

crop = st.selectbox("Crop", sorted(hist["Item"].unique()))
h = hist[hist["Item"] == crop].sort_values("Year")
last = h.iloc[-1]
next_year = int(last["Year"]) + 1

# "Normal" weather = average of last 10 years
recent = h.tail(10)
weather_cols = ["Monsoon_rain_mm", "Monsoon_temp", "Rabi_rain_mm",
                "Rabi_temp", "Annual_rain_mm", "Annual_temp"]
normal = recent[weather_cols].mean()

st.subheader("Scenario")
c1, c2 = st.columns(2)
rain_change = c1.slider("Rainfall change vs. normal (%)", -40, 40, 0, step=5)
temp_change = c2.slider("Temperature change vs. normal (°C)", -2.0, 3.0, 0.0, step=0.5)
area_change = st.slider("Area harvested change vs. last year (%)", -20, 20, 0, step=1)

# Build the input row
row = {
    "Year": next_year,
    "Area_ha": last["Area_ha"] * (1 + area_change / 100),
    "Yield_lag1": last["Yield_t_ha"],
    "Yield_lag2": h.iloc[-2]["Yield_t_ha"],
    "Yield_roll3": h.tail(3)["Yield_t_ha"].mean(),
    "Area_lag1": last["Area_ha"],
}
for col in weather_cols:
    if "rain" in col:
        row[col] = normal[col] * (1 + rain_change / 100)
    else:
        row[col] = normal[col] + temp_change

X = pd.DataFrame([row])
for f in features:                       # crop dummy columns
    if f.startswith("Item_"):
        X[f] = 1 if f == f"Item_{crop}" else 0
X = X[features]                          # same order as training

pred = float(model.predict(X)[0])
baseline = float(last["Yield_t_ha"])

st.subheader("Prediction")
m1, m2 = st.columns(2)
m1.metric(f"Predicted yield ({next_year})", f"{pred:.2f} t/ha",
          delta=f"{(pred - baseline) / baseline * 100:+.1f}% vs last known")
m2.metric(f"Last known yield ({int(last['Year'])})", f"{baseline:.2f} t/ha")

chart = h[["Year", "Yield_t_ha"]].set_index("Year").rename(columns={"Yield_t_ha": "Actual"})
chart.loc[next_year, "Predicted"] = pred
st.line_chart(chart)

st.caption("Predictions are for learning and demonstration only, not for farming or financial decisions.")
