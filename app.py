import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# ===== AUTHENTICATION SECTION =====
def check_password():
    """Returns `True` if the user had a correct password."""
    
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        # Login form
        st.markdown("### 🔐 Smart Waste Management Login")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("")
        
        with col2:
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            # Sample credentials (in production, use a database)
            valid_users = {
                "admin": "admin123",
                "user": "user123",
                "demo": "demo123"
            }
            
            if st.button("Login"):
                if username in valid_users and valid_users[username] == password:
                    st.session_state.password_correct = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        return False
    else:
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.password_correct = False
            st.session_state.username = ""
            st.rerun()
        return True


# ===== GENERATE TRAINING DATA =====
@st.cache_resource
def generate_training_data(n_samples=60000):
    """Generate 60,000 synthetic waste management data points."""
    np.random.seed(42)
    
    # Generate features
    data = pd.DataFrame({
        'Bin_ID': np.random.randint(1, 100, n_samples),
        'Hour_of_Day': np.random.randint(0, 24, n_samples),
        'Day_of_Week': np.random.randint(0, 7, n_samples),
        'Temperature': np.random.uniform(5, 35, n_samples),
        'Previous_Fill_Level': np.random.uniform(0, 100, n_samples),
        'Days_Since_Collection': np.random.randint(0, 30, n_samples),
        'Waste_Type_Code': np.random.randint(1, 5, n_samples),  # 1=organic, 2=recyclable, 3=plastic, 4=mixed
    })
    
    # Generate target variable with some realistic correlation
    data['Fill_Level'] = (
        0.5 * data['Previous_Fill_Level'] +
        0.3 * data['Days_Since_Collection'] * 3 +
        0.1 * data['Temperature'] +
        0.1 * data['Hour_of_Day'] * 2 +
        np.random.normal(0, 5, n_samples)
    )
    
    # Clip to valid range
    data['Fill_Level'] = data['Fill_Level'].clip(0, 100)
    
    return data


# ===== TRAIN ML MODEL =====
@st.cache_resource
def train_ml_model(training_data):
    """Train a Random Forest model on 60,000 data points."""
    X = training_data.drop('Fill_Level', axis=1)
    y = training_data['Fill_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X, y, mse, r2, X_test, y_test, y_pred


# ===== MAIN APP =====
if check_password():
    # Page configuration
    st.set_page_config(page_title="Smart Waste Management", layout="wide")
    
    st.title("🗑️ Smart Waste Management Dashboard")
    st.markdown(f"**Welcome, {st.session_state.username}!** 👋")
    
    st.write("AI-Based Domestic Waste Management System with ML Predictions")
    
    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame({
            "Bin ID": [1, 2, 3, 4, 5],
            "Fill Level (%)": [30, 75, 50, 85, 20],
            "Last Collected": pd.to_datetime(["2025-02-01", "2025-02-20", "2025-02-25", "2025-02-15", "2025-02-26"]),
        })
    
    # ===== TABS =====
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 ML Models", "📈 Analytics", "⚙️ Bin Management"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.header("Dashboard Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_fill = st.session_state.data["Fill Level (%)"].mean()
            st.metric("Avg Fill Level", f"{avg_fill:.1f}%", delta=f"{avg_fill-50:.1f}%")
        
        with col2:
            max_fill = st.session_state.data["Fill Level (%)"].max()
            st.metric("Max Fill Level", f"{max_fill:.1f}%", delta="Critical" if max_fill > 80 else "Ok")
        
        with col3:
            bins_full = len(st.session_state.data[st.session_state.data["Fill Level (%)"] > 80])
            st.metric("Bins > 80%", bins_full, delta=f"{bins_full} need collection")
        
        with col4:
            total_bins = len(st.session_state.data)
            st.metric("Total Bins", total_bins)
        
        # Chart 1: Fill Level Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                st.session_state.data,
                x="Bin ID",
                y="Fill Level (%)",
                title="Current Bin Fill Levels",
                color="Fill Level (%)",
                color_continuous_scale=["green", "yellow", "red"],
            )
            fig_bar.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Alert Level")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['< 50%', '50-80%', '> 80%'],
                values=[
                    len(st.session_state.data[st.session_state.data["Fill Level (%)"] < 50]),
                    len(st.session_state.data[(st.session_state.data["Fill Level (%)"] >= 50) & (st.session_state.data["Fill Level (%)"] <= 80)]),
                    len(st.session_state.data[st.session_state.data["Fill Level (%)"] > 80])
                ],
                marker=dict(colors=['green', 'yellow', 'red'])
            )])
            fig_pie.update_layout(title="Bin Status Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("📋 Bin Details")
        st.dataframe(st.session_state.data, use_container_width=True)
    
    # TAB 2: ML MODELS
    with tab2:
        st.header("🤖 Machine Learning Model")
        st.write("Trained on 60,000 synthetic waste management data points")
        
        # Generate training data and train model
        st.info("⏳ Loading and training ML model on 60,000 data points...")
        training_data = generate_training_data(n_samples=60000)
        model, X, y, mse, r2, X_test, y_test, y_pred = train_ml_model(training_data)
        
        # Display model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Data Points", "60,000")
        
        with col2:
            st.metric("Model R² Score", f"{r2:.4f}", delta="High Accuracy" if r2 > 0.8 else "Moderate")
        
        with col3:
            rmse = np.sqrt(mse)
            st.metric("RMSE", f"{rmse:.2f}%")
        
        st.success("✅ Model trained successfully!")
        
        # Prediction vs Actual
        st.subheader("Predictions vs Actual Values")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            y=y_test[:500],
            name='Actual',
            mode='markers',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig_pred.add_trace(go.Scatter(
            y=y_pred[:500],
            name='Predicted',
            mode='markers',
            marker=dict(color='red', opacity=0.6)
        ))
        fig_pred.update_layout(title="Predictions vs Actual (First 500 Test Samples)", 
                              xaxis_title="Sample", yaxis_title="Fill Level (%)")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            title='Feature Importance in ML Model',
            orientation='h'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Make predictions on current bins
        st.subheader("Predict Next Fill Levels")
        if st.button("🔮 Predict Fill Levels"):
            st.write("Creating predictions for current bins...")
            # Simple predictions based on current data
            for idx, row in st.session_state.data.iterrows():
                sample = np.array([[
                    row["Bin ID"],
                    12,  # Hour
                    3,   # Day of week
                    20,  # Temperature
                    row["Fill Level (%)"],
                    2,   # Days since collection
                    2    # Waste type
                ]])
                prediction = model.predict(sample)[0]
                st.write(f"**Bin {int(row['Bin ID'])}**: Current: {row['Fill Level (%)']:.1f}% → Predicted: {prediction:.1f}%")
    
    # TAB 3: ANALYTICS
    with tab3:
        st.header("📈 Analytics & Insights")
        
        # Trend analysis
        st.subheader("Fill Level Trends")
        dates = pd.date_range(start='2025-01-15', periods=30, freq='D')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Avg Fill Level': np.cumsum(np.random.randn(30)) + 50
        })
        trend_data['Avg Fill Level'] = trend_data['Avg Fill Level'].clip(20, 90)
        
        fig_trend = px.line(trend_data, x='Date', y='Avg Fill Level', 
                           title='Average Fill Level Over Time',
                           markers=True)
        fig_trend.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Alert")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Collection optimization
        st.subheader("Collection Recommendations")
        bins_to_collect = st.session_state.data[st.session_state.data["Fill Level (%)"] > 75]
        
        if len(bins_to_collect) > 0:
            st.warning(f"⚠️ {len(bins_to_collect)} bins need collection soon!")
            st.dataframe(bins_to_collect, use_container_width=True)
        else:
            st.success("✅ All bins are within safe levels!")
    
    # TAB 4: BIN MANAGEMENT
    with tab4:
        st.header("⚙️ Manage Bins")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add Random Bin")
            if st.button("➕ Add Random Bin"):
                new_id = int(st.session_state.data["Bin ID"].max() + 1)
                new_row = pd.DataFrame([{
                    "Bin ID": new_id,
                    "Fill Level (%)": np.random.randint(10, 80),
                    "Last Collected": pd.Timestamp.today(),
                }])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.success(f"✅ Bin {new_id} added!")
                st.rerun()
        
        with col2:
            st.subheader("Add Custom Bin")
            with st.form("custom_bin_form"):
                fill = st.slider("Fill level (%)", 0, 100, 50)
                collected = st.date_input("Last collected date", value=pd.Timestamp.today())
                submitted = st.form_submit_button("✅ Add Bin")
                if submitted:
                    new_id = int(st.session_state.data["Bin ID"].max() + 1)
                    new_row = pd.DataFrame([{
                        "Bin ID": new_id,
                        "Fill Level (%)": fill,
                        "Last Collected": pd.to_datetime(collected),
                    }])
                    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                    st.success(f"✅ Bin {new_id} added successfully!")
                    st.rerun()
        
        st.divider()
        st.subheader("Delete Bin")
        bin_id = st.selectbox("Select Bin to Delete", st.session_state.data["Bin ID"].unique())
        if st.button("🗑️ Delete Bin"):
            st.session_state.data = st.session_state.data[st.session_state.data["Bin ID"] != bin_id]
            st.success(f"✅ Bin {bin_id} deleted!")
            st.rerun()