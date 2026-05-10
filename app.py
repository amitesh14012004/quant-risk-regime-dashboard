"""
Multi-Asset Volatility Regime Detection & Risk Dashboard
=========================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility & Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Multi-Asset Volatility Regime Detection & Risk Dashboard")
st.markdown("*GARCH Volatility Modeling · HMM Regime Detection · Dynamic VaR/CVaR*")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

TICKERS = {
    "Nifty 50":   "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT":   "^CNXIT",
}

ticker_name = st.sidebar.selectbox("Select Index", list(TICKERS.keys()))
start_date  = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date    = st.sidebar.date_input("End Date",   pd.to_datetime("2024-12-31"))
confidence  = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, step=0.01)
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", 
                                         value=1_000_000, step=100_000)
n_regimes   = st.sidebar.radio("Number of HMM Regimes", [2, 3], index=1)

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# DATA & COMPUTATION (cached)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_and_compute(ticker_sym, start, end, confidence, n_regimes):
    """All heavy computation cached so UI stays responsive."""
    
    # 1. Download data
    df = yf.download(ticker_sym, start=str(start), end=str(end), progress=False)
    prices = df["Close"].squeeze()
    returns = (np.log(prices / prices.shift(1)) * 100).dropna()
    
    # 2. GARCH(1,1)
    garch_mdl = arch_model(returns, vol="Garch", p=1, q=1, 
                           dist="normal", mean="Constant")
    garch_res  = garch_mdl.fit(disp="off")
    cond_vol   = garch_res.conditional_volatility
    annual_vol = cond_vol * np.sqrt(252)
    
    # GARCH parameters
    params = {
        "omega": garch_res.params["omega"],
        "alpha": garch_res.params["alpha[1]"],
        "beta":  garch_res.params["beta[1]"],
        "persistence": garch_res.params["alpha[1]"] + garch_res.params["beta[1]"]
    }
    
    # 3. HMM Regime Detection
    aligned = pd.DataFrame({"ret": returns, "vol": cond_vol}).dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(aligned[["ret", "vol"]])
    
    hmm = GaussianHMM(n_components=n_regimes, covariance_type="full",
                      n_iter=2000, random_state=42)
    hmm.fit(X)
    states = hmm.predict(X)
    
    # Label states by mean return
    state_order = np.argsort(hmm.means_[:, 0])
    if n_regimes == 2:
        labels_map = {state_order[0]: "Bear 🔴", state_order[1]: "Bull 🟢"}
        color_map  = {"Bear 🔴": "#ef4444", "Bull 🟢": "#22c55e"}
    else:
        labels_map = {state_order[0]: "Bear 🔴", 
                      state_order[1]: "Sideways 🟡", 
                      state_order[2]: "Bull 🟢"}
        color_map  = {"Bear 🔴": "#ef4444", "Sideways 🟡": "#f59e0b", "Bull 🟢": "#22c55e"}
    
    regime_labels = pd.Series(states, index=aligned.index).map(labels_map)
    
    # 4. GARCH VaR / CVaR
    mu = returns.mean()
    z  = stats.norm.ppf(1 - confidence)
    
    var_pct  = mu + z * cond_vol
    cvar_pct = mu - cond_vol * stats.norm.pdf(z) / (1 - confidence)
    var_inr  = abs(var_pct  / 100) * portfolio_val
    cvar_inr = abs(cvar_pct / 100) * portfolio_val
    
    # 5. Compile into one DataFrame
    main_df = pd.DataFrame({
        "price":      prices,
        "returns":    returns,
        "cond_vol":   cond_vol,
        "annual_vol": annual_vol,
        "var_pct":    var_pct,
        "cvar_pct":   cvar_pct,
        "var_inr":    var_inr,
        "cvar_inr":   cvar_inr,
    }).dropna()
    
    main_df["regime"] = regime_labels.reindex(main_df.index)
    
    return main_df, params, hmm, labels_map, color_map, returns


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
if run_button:
    with st.spinner("Downloading data and running models..."):
        try:
            df, params, hmm, labels_map, color_map, returns = load_and_compute(
                TICKERS[ticker_name], start_date, end_date, confidence, n_regimes
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    
    st.success(f"✓ Analysis complete | {len(df)} trading days | {ticker_name}")
    
    # ── KPI METRICS ──────────────────────────────────────
    st.subheader("📌 Key Risk Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    latest_var  = df["var_inr"].iloc[-1]
    latest_cvar = df["cvar_inr"].iloc[-1]
    latest_vol  = df["annual_vol"].iloc[-1]
    current_regime = df["regime"].iloc[-1]
    persistence = params["persistence"]
    
    k1.metric("Current Regime",    current_regime)
    k2.metric("Annualized Vol",    f"{latest_vol:.1f}%")
    k3.metric(f"{confidence*100:.0f}% 1-Day VaR",  f"₹{latest_var:,.0f}")
    k4.metric(f"{confidence*100:.0f}% 1-Day CVaR", f"₹{latest_cvar:,.0f}")
    k5.metric("Vol Persistence (α+β)", f"{persistence:.4f}")
    
    st.divider()
    
    # ── CHART 1: Price + Regime Background ───────────────
    st.subheader("📈 Price Chart with Regime Detection")
    
    fig1 = go.Figure()
    
    # Price line
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["price"],
        name="Price", line=dict(color="#3b82f6", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>"
    ))
    
    # Regime color bands (group consecutive same-regime days)
    if "regime" in df.columns:
        regime_groups = (df["regime"] != df["regime"].shift()).cumsum()
        for _, group_df in df.groupby(regime_groups):
            if len(group_df) > 0:
                regime = group_df["regime"].iloc[0]
                color  = color_map.get(regime, "gray")
                fig1.add_vrect(
                    x0=group_df.index[0],
                    x1=group_df.index[-1],
                    fillcolor=color,
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    annotation_text="" 
                )
    
    fig1.update_layout(
        height=350, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Index Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # ── CHART 2: GARCH Volatility ─────────────────────────
    st.subheader("🌋 GARCH Conditional Volatility (Annualized)")
    
    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=["Annualized Volatility (%)", "Daily Returns (%)"]
    )
    
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["annual_vol"],
        name="GARCH Volatility", fill="tozeroy",
        line=dict(color="#f59e0b", width=1.5),
        fillcolor="rgba(245,158,11,0.2)"
    ), row=1, col=1)
    
    fig2.add_trace(go.Bar(
        x=df.index, y=df["returns"],
        name="Daily Returns",
        marker_color=np.where(df["returns"] >= 0, "#22c55e", "#ef4444")
    ), row=2, col=1)
    
    fig2.update_layout(height=450, hovermode="x unified",
                       showlegend=True)
    fig2.add_annotation(
        text=f"α={params['alpha']:.4f}  β={params['beta']:.4f}  α+β={params['persistence']:.4f}",
        xref="paper", yref="paper", x=0.01, y=0.99,
        showarrow=False,
        bgcolor="rgba(0,0,0,0.5)",
        font=dict(color="white", size=11)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # ── CHART 3: Dynamic VaR/CVaR ────────────────────────
    st.subheader(f"🛡️ Dynamic GARCH-VaR & CVaR (₹{portfolio_val:,.0f} Portfolio)")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df.index, y=df["cvar_inr"],
        name=f"CVaR ({confidence*100:.0f}%)",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)"
    ))
    fig3.add_trace(go.Scatter(
        x=df.index, y=df["var_inr"],
        name=f"VaR ({confidence*100:.0f}%)",
        line=dict(color="#f59e0b", width=1.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.2)"
    ))
    fig3.update_layout(
        height=350, hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Loss Estimate (₹)",
        yaxis_tickformat=",",
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # ── CHART 4: Regime Distribution ─────────────────────
    st.subheader("🥧 Regime Distribution & Statistics")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        regime_counts = df["regime"].value_counts()
        fig4 = px.pie(
            values=regime_counts.values,
            names=regime_counts.index,
            color=regime_counts.index,
            color_discrete_map=color_map,
            title="Days in Each Regime"
        )
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col_r:
        st.markdown("**Regime Summary Statistics**")
        regime_stats = df.groupby("regime")["returns"].agg([
            ("Mean Return (%)", lambda x: x.mean().round(3)),
            ("Volatility (%)",  lambda x: x.std().round(3)),
            ("Worst Day (%)",   lambda x: x.min().round(3)),
            ("Best Day (%)",    lambda x: x.max().round(3)),
            ("Days",            "count")
        ])
        st.dataframe(regime_stats, use_container_width=True)
    
    # ── CHART 5: Return Distribution by Regime ──────────
    st.subheader("📉 Return Distribution by Regime")
    
    fig5 = go.Figure()
    for regime, color in color_map.items():
        subset = df[df["regime"] == regime]["returns"]
        if len(subset) > 10:
            fig5.add_trace(go.Histogram(
                x=subset, name=regime,
                marker_color=color,
                opacity=0.6,
                nbinsx=50,
                histnorm="probability density"
            ))
    fig5.update_layout(
        barmode="overlay", height=300,
        xaxis_title="Daily Return (%)",
        yaxis_title="Density",
        title="Return Distributions are Distinctly Different Across Regimes"
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # ── GARCH PARAMS BOX ─────────────────────────────────
    st.subheader("🔢 GARCH Model Parameters")
    
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("ω (Omega)", f"{params['omega']:.6f}", 
              help="Baseline variance — long-run floor of volatility")
    p2.metric("α (Alpha)",  f"{params['alpha']:.4f}",  
              help="ARCH term: how much recent shocks move volatility")
    p3.metric("β (Beta)",   f"{params['beta']:.4f}",   
              help="GARCH term: how much past variance persists")
    p4.metric("α+β Persistence", f"{params['persistence']:.4f}", 
              help="Values near 1 = long memory. >1 = explosive (bad)")
    
    st.divider()
    st.caption("Built with GARCH (arch library) · HMM (hmmlearn) · VaR/CVaR (scipy) · Dashboard: Streamlit + Plotly")

else:
    # Landing page
    st.info("👈 Configure the settings in the sidebar and click **Run Analysis** to start.")
    
    st.markdown("""
    ### What this dashboard does:
    
    | Component | Method | Output |
    |---|---|---|
    | **Volatility Modeling** | GARCH(1,1) | Time-varying σ(t) for each trading day |
    | **Regime Detection** | Hidden Markov Model | Bull / Bear / Sideways labels |
    | **Risk Estimation** | GARCH-VaR & CVaR | Daily loss estimate in ₹ |
    | **Visualization** | Plotly + Streamlit | Interactive charts & KPIs |
    
    ### Indexes covered:
    - **Nifty 50** — Broad Indian market
    - **Nifty Bank** — Banking sector
    - **Nifty IT** — Technology sector
    """)
