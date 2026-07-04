import streamlit as st
import pandas as pd

df = pd.read_csv("outputs/traffic_report.csv")

st.title("Traffic Analytics Dashboard")

st.metric("Total Vehicles", df["total_vehicles"].max())

st.bar_chart(
    df[["cars", "buses", "trucks", "motorcycles"]].iloc[-1]
)