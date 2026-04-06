import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ---------- Load data ----------
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "processed" / "final_model_data.csv"
data = pd.read_csv(data_path)
data["Date"] = pd.to_datetime(data["Date"])

# ---------- Train model ----------
X = data[["ZHVI", "MORTGAGE30US", "ZHVI_Lag1", "Rate_Lag1"]]
y = data["Target_Next_Month_ZHVI"]

model = LinearRegression()
model.fit(X, y)

# ---------- Title ----------
st.title("Los Angeles House Price Predictor")
st.write("Predict next month's Los Angeles home value using mortgage rates and previous housing values.")

# ---------- Sidebar ----------
st.sidebar.header("Enter Input Values")

current_zhvi = st.sidebar.number_input("Current ZHVI", value=float(data["ZHVI"].iloc[-1]))
current_rate = st.sidebar.number_input("Current Mortgage Rate", value=float(data["MORTGAGE30US"].iloc[-1]))
lag_zhvi = st.sidebar.number_input("Previous Month ZHVI", value=float(data["ZHVI_Lag1"].iloc[-1]))
lag_rate = st.sidebar.number_input("Previous Month Mortgage Rate", value=float(data["Rate_Lag1"].iloc[-1]))

prediction = None
if st.sidebar.button("Predict Next Month Home Value"):
    input_data = pd.DataFrame({
        "ZHVI": [current_zhvi],
        "MORTGAGE30US": [current_rate],
        "ZHVI_Lag1": [lag_zhvi],
        "Rate_Lag1": [lag_rate]
    })
    prediction = model.predict(input_data)[0]

# ---------- Top metrics ----------
col1, col2, col3 = st.columns(3)

col1.metric("Latest ZHVI", f"${data['ZHVI'].iloc[-1]:,.2f}")
col2.metric("Latest Mortgage Rate", f"{data['MORTGAGE30US'].iloc[-1]:.2f}%")

if prediction is not None:
    col3.metric("Predicted Next Month Value", f"${prediction:,.2f}")
else:
    col3.metric("Predicted Next Month Value", "Click Predict")

# ---------- Prediction message ----------
if prediction is not None:
    st.success(f"Predicted Next Month Value: ${prediction:,.2f}")


#        Predictor
pred_all = model.predict(X)

results = data.copy()
results["Predicted_Next_Month_ZHVI"] = pred_all

st.subheader("Linear Regression Prediction Trend")
fig4, ax4 = plt.subplots(figsize=(10, 3.8))
ax4.plot(results["Date"], results["Target_Next_Month_ZHVI"], label="Actual")
ax4.plot(results["Date"], results["Predicted_Next_Month_ZHVI"], label="Predicted")
ax4.set_xlabel("Date")
ax4.set_ylabel("Home Value")
ax4.set_title("Actual vs Predicted Home Value")
ax4.legend()
plt.xticks(rotation=45)
st.pyplot(fig4)


# ---------- Charts ----------
st.subheader("Home Value Trend")
fig1, ax1 = plt.subplots(figsize=(10, 3.8))
ax1.plot(data["Date"], data["ZHVI"])
ax1.set_xlabel("Date")
ax1.set_ylabel("ZHVI")
ax1.set_title("Los Angeles Home Values Over Time")
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Mortgage Rate Trend")
fig2, ax2 = plt.subplots(figsize=(10, 3.8))
ax2.plot(data["Date"], data["MORTGAGE30US"])
ax2.set_xlabel("Date")
ax2.set_ylabel("Mortgage Rate")
ax2.set_title("Mortgage Rates Over Time")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("Mortgage Rate vs Home Value")
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.scatter(data["MORTGAGE30US"], data["ZHVI"])
ax3.set_xlabel("Mortgage Rate")
ax3.set_ylabel("ZHVI")
ax3.set_title("Mortgage Rate vs Home Value")
st.pyplot(fig3)

# ---------- Recent data ----------
st.subheader("Recent Data Preview")

display_data = data.tail().copy()
display_data["Date"] = pd.to_datetime(display_data["Date"]).dt.strftime("%Y-%m")
st.dataframe(display_data)

# ---------- Model performance ----------
st.subheader("Model Performance")
m1, m2, m3 = st.columns(3)
m1.metric("Linear Regression MAE", "2365.27")
m2.metric("Linear Regression RMSE", "2856.68")
m3.metric("Linear Regression R²", "0.9982")

st.info("Linear Regression was chosen because it captured the time trend much better than Random Forest.")

# ---------- Summary ----------
st.subheader("Project Summary")
st.write("""
- Dataset 1: Zillow Metro ZHVI data
- Dataset 2: FRED mortgage interest rate data
- Metro used: Los Angeles, CA
- Final model used in the app: Linear Regression
- Goal: Predict next month's home value
""")

st.download_button(
    label="Download Processed Data",
    data=data.to_csv(index=False),
    file_name="final_model_data.csv",
    mime="text/csv"
)

with st.expander("About this project"):
    st.write("This project predicts next month's Los Angeles home value using Zillow and FRED datasets.")


