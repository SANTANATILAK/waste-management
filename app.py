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

# Simple interaction and bin management

def add_random_entry():
    # automatically pick next Bin ID
    new_id = int(st.session_state.data["Bin ID"].max() + 1)
    new_row = pd.DataFrame([{  # create a tiny single-row DataFrame
        "Bin ID": new_id,
        "Fill Level (%)": np.random.randint(0, 101),
        "Last Collected": pd.Timestamp.today(),
    }])
    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)

if st.button("Add random bin"):
    add_random_entry()

# allow user to add a custom bin entry
with st.expander("Add custom bin"):
    with st.form("custom_bin_form"):
        fill = st.number_input("Fill level (%)", min_value=0, max_value=100, value=0)
        collected = st.date_input("Last collected date", value=pd.Timestamp.today())
        submitted = st.form_submit_button("Add bin")
        if submitted:
            new_id = int(st.session_state.data["Bin ID"].max() + 1)
            new_row = pd.DataFrame([
                {
                    "Bin ID": new_id,
                    "Fill Level (%)": fill,
                    "Last Collected": pd.to_datetime(collected),
                }
            ])
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.success(f"Bin {new_id} added.")