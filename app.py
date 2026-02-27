import streamlit as st
import pandas as pd
import numpy as np

# simple in-memory credentials (replace with secure auth in production)
VALID_USERS = {"admin": "password", "user": "1234"}

st.set_page_config(page_title="Smart Waste Management Dashboard")
st.title("Smart Waste Management Dashboard")

# --- authentication -------------------------------------------------

def exchange_code_for_user_info(code: str):
    """Use Google's OAuth token endpoint to exchange authorization code for user info."""
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": st.secrets.google.client_id,
        "client_secret": st.secrets.google.client_secret,
        "redirect_uri": st.secrets.google.redirect_uri,
        "grant_type": "authorization_code",
    }
    resp = requests.post(token_url, data=data)
    resp.raise_for_status()
    tok = resp.json()
    # id_token contains JWT with user info; alternatively call userinfo endpoint
    userinfo = requests.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {tok['access_token']}"},
    )
    userinfo.raise_for_status()
    return userinfo.json()


def handle_google_callback():
    # check for code in query params and perform exchange
    params = st.experimental_get_query_params()
    if "code" in params and has_google_secrets():
        try:
            info = exchange_code_for_user_info(params["code"][0])
            # use email address as username
            st.session_state.logged_in = True
            st.session_state.user = info.get("email") or info.get("sub")
            st.success(f"Logged in as {st.session_state.user} via Google")
            # clear query params to avoid re-processing
            st.experimental_set_query_params()
        except Exception as e:
            st.error(f"Google login failed: {e}")


def login():
    # attempt google callback first
    handle_google_callback()

    if st.session_state.get("logged_in"):
        return

    st.subheader("Please log in")

    # traditional credential form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Logged in successfully")
        else:
            st.error("Invalid credentials")

    st.markdown("---")
    st.write("*Or use your Google account to sign in:*")

    # Google OAuth link (requires config in Streamlit secrets)
    # safely access secrets; return False if missing
    def has_google_secrets():
        try:
            return "google" in st.secrets and st.secrets.google.get("client_id")
        except Exception:
            return False

    if has_google_secrets():
        from urllib.parse import urlencode
        params = {
            "client_id": st.secrets.google.client_id,
            "redirect_uri": st.secrets.google.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account",
        }
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
        st.markdown(f"[Sign in with Google]({auth_url})")
    else:
        st.info("Google sign-in not configured. Add credentials to Streamlit secrets (local: .streamlit/secrets.toml)")

# show login form if not logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    st.stop()

st.write(f"Welcome, **{st.session_state.user}**! This is a placeholder Streamlit application for the Smart AI-Based Domestic Waste Management System.")

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

# ---------------------------------------------------------------
# model training / data simulation section

def generate_training_data(n=60000):
    # create simple linear data with noise
    X = np.random.rand(n, 1)
    y = 3 * X.squeeze() + np.random.randn(n) * 0.1
    return X, y

st.markdown("---")
st.header("Model training / data simulation")
if st.button("Generate 60 000 samples and train"):
    with st.spinner("Generating data and fitting model..."):
        X, y = generate_training_data()
        # fit a linear model using least squares
        A = np.hstack([X, np.ones((len(X), 1))])
        coef, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        # simulate a training loss curve
        epochs = np.arange(1, 51)
        losses = np.exp(-epochs / 10) + np.random.rand(len(epochs)) * 0.02
        st.write(f"Fitted coefficient: **{coef:.3f}**, intercept: **{intercept:.3f}**")
        st.line_chart(losses, height=300, use_container_width=True)
        # show a small sample scatter to visualize data
        sample_idx = np.random.choice(len(X), size=500, replace=False)
        df_scatter = pd.DataFrame({"X": X[sample_idx].flatten(), "y": y[sample_idx]})
        import altair as alt
        st.altair_chart(
            alt.Chart(df_scatter).mark_circle(size=10, opacity=0.4)
            .encode(x="X", y="y"), use_container_width=True
        )
