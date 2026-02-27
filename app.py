import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from io import BytesIO

# ------------------ authentication ------------------
VALID_USERS = {"admin": "password"}

def login():
    st.subheader("Login to Waste Management System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.user = username
        else:
            st.error("Please enter both username and password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------ page setup ------------------
st.set_page_config(page_title="Smart Waste Management Dashboard", layout="wide")
st.sidebar.title(f"Welcome {st.session_state.user}")
page = st.sidebar.radio("Go to", [
    "Dashboard",
    "Waste Data Analysis",
    "Clustering",
    "Route Optimization",
    "Model Evaluation",
    "Alerts",
    "About",
])

# ------------------ data utilities ------------------
@st.cache_data
def generate_sample_dataset(num_records: int = 600):
    """Generate approximately `num_records` rows by varying number of wards and weeks."""
    # choose reasonable number of wards and weeks to reach target
    wards = [f"Ward {i}" for i in range(1, 11)]
    weeks = int(np.ceil(num_records / len(wards)))
    dates = pd.date_range("2025-01-01", periods=weeks, freq="W")
    rows = []
    for d in dates:
        for w in wards:
            rows.append({
                "date": d,
                "ward": w,
                "waste_volume": np.random.randint(50, 500),
                "recycling_rate": np.random.rand(),
                "population_density": np.random.randint(1000, 10000),
            })
    df = pd.DataFrame(rows)
    if len(df) > num_records:
        df = df.sample(num_records, random_state=42).reset_index(drop=True)
    return df

# allow user to choose how much synthetic data to produce (around 600 by default)
num_samples = st.sidebar.number_input("Dataset size", min_value=100, max_value=2000, value=600, step=100)
df = generate_sample_dataset(int(num_samples))

# download helper

def make_downloadable(df, filename="report.csv"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ------------------ functionality pages ------------------

def show_dashboard():
    st.title("Smart Waste Management Dashboard")
    st.markdown("This system offers data analysis, clustering, route optimization, and more.")
    st.dataframe(df.head())


def show_data_analysis():
    st.header("Waste Data Analysis")
    st.subheader("Waste distribution by ward")
    chart = alt.Chart(df).mark_bar().encode(
        x="ward", y="waste_volume", color="ward"
    )
    st.altair_chart(chart, use_container_width=True)
    st.subheader("Time trends")
    ts = df.groupby("date")["waste_volume"].sum().reset_index()
    st.line_chart(ts.rename(columns={"date": "index"}).set_index("index"))
    st.subheader("Recycling rates")
    st.bar_chart(df.groupby("ward")["recycling_rate"].mean())
    st.subheader("Correlation matrix")
    corr = df[["waste_volume", "recycling_rate", "population_density"]].corr()
    st.write(corr)


def show_clustering():
    st.header("Zone Clustering")
    features = df[["waste_volume", "population_density"]]
    k = st.slider("Number of clusters", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(features)
    df_plot = df.copy()
    df_plot["cluster"] = labels
    scatter = alt.Chart(df_plot).mark_circle(size=60).encode(
        x="waste_volume", y="population_density", color="cluster:N", tooltip=["ward", "cluster"]
    )
    st.altair_chart(scatter, use_container_width=True)
    st.write("Cluster centers:")
    centers = pd.DataFrame(model.cluster_centers_, columns=features.columns)
    st.write(centers)


def nearest_neighbor(coords):
    unvisited = coords.copy()
    route = [unvisited.pop(0)]
    while unvisited:
        last = route[-1]
        distances = [np.linalg.norm(np.array(last) - np.array(u)) for u in unvisited]
        idx = int(np.argmin(distances))
        route.append(unvisited.pop(idx))
    return route


def route_length(route):
    return sum(np.linalg.norm(np.array(route[i]) - np.array(route[i - 1])) for i in range(1, len(route)))


def two_opt(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_length(new_route) < route_length(best):
                    best = new_route
                    improved = True
        route = best
    return best


def show_route_optimization():
    st.header("Route Optimization")
    n = st.number_input("Number of stops", 5, 50, 20)
    coords = [(np.random.rand(), np.random.rand()) for _ in range(n)]
    nn = nearest_neighbor(coords.copy())
    opt = two_opt(nn.copy())
    st.write(f"Nearest neighbor distance: {route_length(nn):.2f}")
    st.write(f"2-opt distance: {route_length(opt):.2f}")


def show_model_evaluation():
    st.header("Model Evaluation & Stress Testing")
    X = np.random.randn(1000, 5)
    y = (np.sum(X, axis=1) + np.random.randn(1000) > 0).astype(int)
    y_pred = y.copy()
    if st.checkbox("Add noise to labels"):
        idx = np.random.choice(len(y), size=100, replace=False)
        y_pred[idx] = 1 - y_pred[idx]
    cm = confusion_matrix(y, y_pred)
    st.write("Confusion matrix:")
    st.write(cm)
    st.write("Precision", precision_score(y, y_pred))
    st.write("Recall", recall_score(y, y_pred))
    st.write("F1-score", f1_score(y, y_pred))


def show_alerts():
    st.header("Smart Alerts & Recommendations")
    alerts = []
    high_vol = df[df.waste_volume > 450]
    for _, row in high_vol.iterrows():
        alerts.append(f"{row['ward']} on {row['date'].date()} has high waste volume ({row['waste_volume']})")
    if alerts:
        st.warning("\n".join(alerts))
    else:
        st.success("No alerts!")
    if st.button("Download report"):
        buf = make_downloadable(high_vol)
        st.download_button("Download CSV", data=buf, file_name="alerts.csv", mime="text/csv")


def show_about():
    st.write("This prototype demonstrates analysis, clustering, routing, evaluation, and alert features for a smart waste management system.")

# main dispatcher
def main():
    if page == "Dashboard":
        show_dashboard()
    elif page == "Waste Data Analysis":
        show_data_analysis()
    elif page == "Clustering":
        show_clustering()
    elif page == "Route Optimization":
        show_route_optimization()
    elif page == "Model Evaluation":
        show_model_evaluation()
    elif page == "Alerts":
        show_alerts()
    else:
        show_about()

if __name__ == "__main__":
    main()
