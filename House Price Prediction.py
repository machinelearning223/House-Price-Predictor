# %%
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Split features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🏠 California House Price Predictor")

st.write("Enter house details below to estimate the median house value:")

inputs = {}
for col in X.columns:
    inputs[col] = st.number_input(
        label=col,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean())
    )

# Convert user input to DataFrame
input_df = pd.DataFrame([inputs])

# Predict
pred = model.predict(input_df)[0]
st.success(f"💰 Predicted Median House Value: ${pred * 100000:.2f}")



