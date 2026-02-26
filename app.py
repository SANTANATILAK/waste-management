import streamlit as st
import pandas as pd
import numpy as np

st.title("Smart Waste Management Dashboard")

st.write("This is a placeholder Streamlit application for the Smart AI-Based Domestic Waste Management System.")

# Example data
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame({
        "Bin ID": [1, 2, 3],
        "Fill Level (%)": [30, 75, 50],
        "Last Collected": pd.to_datetime(["2025-02-01", "2025-02-20", "2025-02-25"]),
    })

st.dataframe(st.session_state.data)

# Simple interaction
def add_random_entry():
    new_id = st.session_state.data["Bin ID"].max() + 1
    new_entry = {
        "Bin ID": new_id,
        "Fill Level (%)": np.random.randint(0, 101),
        "Last Collected": pd.Timestamp.today(),
    }
    st.session_state.data = st.session_state.data.append(new_entry, ignore_index=True)

if st.button("Add random bin"): 
    add_random_entry()

