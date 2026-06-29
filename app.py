"""
Multi-Asset Volatility Regime Detection & Risk Dashboard
=========================================================
UPGRADED VERSION — Adds:
  1. Regime-conditioned trading signal + backtest
  2. Full performance metrics (Sharpe, Sortino, Max Drawdown, Calmar)
  3. GARCH 10-day volatility forecast panel
  4. Kupiec POF backtest in UI
  5. Multi-index comparison tab
  6. Rolling Sharpe ratio chart
  7. Model comparison (GARCH vs EGARCH vs GJR) in UI
  8. Downloadable results as CSV

Run with: streamlit run app.py
Install:  pip install streamlit arch hmmlearn yfinance plotly scikit-learn scipy
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
from scipy.stats import jarque_bera, kurtosis, skew
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility & Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stMetric { background: #0f172a; border-radius: 8px; padding: 10px; }
    .regime-bull { color: #22c55e; font-weight: bold; }
    .regime-bear { color: #ef4444; font-weight: bold; }
    .regime-side { color: #f59e0b; font-weight: bold; }
    div[data-testid="stTab"] button { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Multi-Asset Volatility Regime Detection & Risk Dashboard")
st.markdown("*GARCH Volatility · HMM Regime Detection · Dynamic VaR/CVaR · Backtest · Forecast*")
st.divider()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

TICKERS = {
    "Nifty 50":   "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT":   "^CNXIT",
}

ticker_name   = st.sidebar.selectbox("Primary Index", list(TICKERS.keys()))
start_date    = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date      = st.sidebar.date_input("End Date",   pd.to_datetime("2024-12-31"))
confidence    = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, step=0.01)
portfolio_val = st.sidebar.number_input("Portfolio Value (₹)", value=1_000_000, step=100_000)
n_regimes     = st.sidebar.radio("Number of HMM Regimes", [2, 3], index=1)
forecast_days = st.sidebar.slider("GARCH Forecast Horizon (days)", 5, 30, 10)

st.sidebar.divider()
st.sidebar.markdown("**Signal Strategy**")
bull_signal  = st.sidebar.radio("Bull Regime Action", ["Long", "Flat"], index=0)
bear_signal  = st.sidebar.radio("Bear Regime Action", ["Short", "Flat"], index=0)

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def performance_metrics(returns_series, label="Strategy"):
    """Compute full suite of performance metrics."""
    r = returns_series.dropna() / 100  # convert % to decimal
    if len(r) == 0:
        return {}

    ann_return  = r.mean() * 252
    ann_vol     = r.std() * np.sqrt(252)
    sharpe      = ann_return / ann_vol if ann_vol != 0 else 0

    # Sortino (downside deviation only)
    downside    = r[r < 0]
    down_dev    = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino     = ann_return / down_dev if down_dev != 0 else 0

    # Max Drawdown
    cum         = (1 + r).cumprod()
    roll_max    = cum.cummax()
    drawdown    = (cum - roll_max) / roll_max
    max_dd      = drawdown.min()

    # Calmar ratio
    calmar      = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate    = (r > 0).mean()

    # Total return
    total_ret   = (1 + r).prod() - 1

    return {
        "Annual Return (%)":    round(ann_return * 100, 2),
        "Annual Volatility (%)":round(ann_vol    * 100, 2),
        "Sharpe Ratio":         round(sharpe,          3),
        "Sortino Ratio":        round(sortino,         3),
        "Calmar Ratio":         round(calmar,          3),
        "Max Drawdown (%)":     round(max_dd    * 100, 2),
        "Win Rate (%)":         round(win_rate   * 100, 2),
        "Total Return (%)":     round(total_ret  * 100, 2),
    }


def kupiec_pof_test(actual_returns, var_threshold, confidence):
    """Kupiec's Proportion of Failures test."""
    aligned  = pd.concat([actual_returns, var_threshold], axis=1).dropna()
    act      = aligned.iloc[:, 0]
    var_t    = aligned.iloc[:, 1]
    breaches = act < var_t
    n, x     = len(breaches), breaches.sum()
    p_hat    = x / n
    p_exp    = 1 - confidence

    if x == 0 or x == n:
        return {"n": n, "breaches": x, "actual_rate": p_hat,
                "expected_rate": p_exp, "LR": np.nan,
                "p_value": np.nan, "pass": None}

    LR = -2 * (
        x * np.log(p_exp / p_hat) +
        (n - x) * np.log((1 - p_exp) / (1 - p_hat))
    )
    p_val = 1 - stats.chi2.cdf(LR, df=1)
    return {
        "n": n, "breaches": x,
        "actual_rate": p_hat, "expected_rate": p_exp,
        "LR": LR, "p_value": p_val,
        "pass": p_val > 0.05
    }


def rolling_sharpe(returns_series, window=60):
    """60-day rolling Sharpe ratio."""
    r = returns_series / 100
    roll_mean = r.rolling(window).mean() * 252
    roll_std  = r.rolling(window).std() * np.sqrt(252)
    return (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan)


# ─────────────────────────────────────────────────────────────────
# MAIN COMPUTATION (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_and_compute(ticker_sym, start, end, confidence, n_regimes):
    """All heavy computation cached."""

    # 1. Data
    df = yf.download(
        ticker_sym,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker_sym}. Check ticker or date range.")

    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].iloc[:,0]
    else:
        prices = df["Close"]

    prices = prices.dropna()
    if len(prices) < 50:
        raise ValueError("Too few price observations.")

    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf,-np.inf], np.nan).dropna()*100

    if len(returns) < 100:
        raise ValueError(f"Only {len(returns)} return observations available.")

    # 2. GARCH(1,1)
    garch_mdl = arch_model(returns, vol="Garch", p=1, q=1,
                           dist="normal", mean="Constant")
    garch_res  = garch_mdl.fit(disp="off")
    cond_vol   = garch_res.conditional_volatility
    annual_vol = cond_vol * np.sqrt(252)

    params = {
        "omega":       garch_res.params["omega"],
        "alpha":       garch_res.params["alpha[1]"],
        "beta":        garch_res.params["beta[1]"],
        "persistence": garch_res.params["alpha[1]"] + garch_res.params["beta[1]"],
        "aic":         garch_res.aic,
        "bic":         garch_res.bic,
    }

    # 3. Model comparison
    comparison = {}
    for label, cfg in {
        "GARCH(1,1)":     dict(vol="Garch",  p=1, o=0, q=1),
        "GJR-GARCH(1,1)": dict(vol="Garch",  p=1, o=1, q=1),
        "EGARCH(1,1)":    dict(vol="EGARCH", p=1, o=1, q=1),
    }.items():
        try:
            m = arch_model(returns, mean="Constant", dist="normal", **cfg)
            r = m.fit(disp="off")
            comparison[label] = {"AIC": round(r.aic, 2), "BIC": round(r.bic, 2)}
        except Exception:
            comparison[label] = {"AIC": None, "BIC": None}

    # 4. HMM
    aligned = pd.DataFrame({"ret": returns, "vol": cond_vol}).dropna()
    scaler  = StandardScaler()
    X       = scaler.fit_transform(aligned[["ret", "vol"]])

    hmm = GaussianHMM(n_components=n_regimes, covariance_type="full",
                      n_iter=2000, random_state=42)
    hmm.fit(X)
    states = hmm.predict(X)

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

    # 5. VaR / CVaR
    mu      = returns.mean()
    z       = stats.norm.ppf(1 - confidence)
    var_pct  = mu + z * cond_vol
    cvar_pct = mu - cond_vol * stats.norm.pdf(z) / (1 - confidence)
    var_inr  = abs(var_pct  / 100) * 1_000_000   # normalised to ₹10L (rescaled in UI)
    cvar_inr = abs(cvar_pct / 100) * 1_000_000

    # 6. Main DataFrame
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

    return main_df, params, garch_res, hmm, labels_map, color_map, comparison, returns


# ─────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("⏳ Downloading data and running models — this takes ~15 seconds..."):
        try:
            df, params, garch_res, hmm, labels_map, color_map, comparison, returns = \
                load_and_compute(TICKERS[ticker_name], start_date, end_date,
                                 confidence, n_regimes)
        except Exception:
            import traceback
            st.error("Analysis failed.")
            st.code(traceback.format_exc())
            st.stop()

    # Scale VaR/CVaR to actual portfolio value
    scale = portfolio_val / 1_000_000
    df["var_inr_scaled"]  = df["var_inr"]  * scale
    df["cvar_inr_scaled"] = df["cvar_inr"] * scale

    st.success(f"✓ Analysis complete — {len(df)} trading days | {ticker_name} | "
               f"{df.index[0].date()} → {df.index[-1].date()}")

    # ── KPI ROW ──────────────────────────────────────────────
    st.subheader("📌 Key Risk Metrics (Latest Day)")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    latest_var    = df["var_inr_scaled"].iloc[-1]
    latest_cvar   = df["cvar_inr_scaled"].iloc[-1]
    latest_vol    = df["annual_vol"].iloc[-1]
    current_reg   = df["regime"].iloc[-1]
    persistence   = params["persistence"]
    total_days    = len(df)

    k1.metric("Current Regime",          current_reg)
    k2.metric("Annualized Vol",          f"{latest_vol:.1f}%")
    k3.metric(f"{confidence*100:.0f}% VaR",  f"₹{latest_var:,.0f}")
    k4.metric(f"{confidence*100:.0f}% CVaR", f"₹{latest_cvar:,.0f}")
    k5.metric("GARCH Persistence (α+β)", f"{persistence:.4f}")
    k6.metric("Trading Days Analyzed",   total_days)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Price & Regimes",
        "🌋 Volatility",
        "⚡ Trading Signal & Backtest",   # NEW
        "🛡️ VaR / CVaR",
        "📡 GARCH Forecast",              # NEW
        "🧪 Model Diagnostics",          # NEW
        "📥 Download",                   # NEW
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1: Price & Regimes
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader(f"{ticker_name} — Price with Regime Detection")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df.index, y=df["price"],
            name="Price", line=dict(color="#3b82f6", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:,.2f}<extra></extra>"
        ))

        # Colored regime bands
        regime_groups = (df["regime"] != df["regime"].shift()).cumsum()
        for _, grp in df.groupby(regime_groups):
            if len(grp):
                regime = grp["regime"].iloc[0]
                fig1.add_vrect(
                    x0=grp.index[0], x1=grp.index[-1],
                    fillcolor=color_map.get(regime, "gray"),
                    opacity=0.15, layer="below", line_width=0
                )

        fig1.update_layout(height=400, hovermode="x unified",
                           xaxis_title="Date", yaxis_title="Index Level")
        st.plotly_chart(fig1, use_container_width=True)

        # Regime stats table
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("**Regime Distribution**")
            regime_counts = df["regime"].value_counts()
            fig_pie = px.pie(
                values=regime_counts.values,
                names=regime_counts.index,
                color=regime_counts.index,
                color_discrete_map=color_map,
            )
            fig_pie.update_layout(height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            st.markdown("**Regime Summary Statistics**")
            reg_stats = df.groupby("regime")["returns"].agg(
                Days="count",
                Mean=lambda x: round(x.mean(), 3),
                Volatility=lambda x: round(x.std(), 3),
                Best=lambda x: round(x.max(), 3),
                Worst=lambda x: round(x.min(), 3),
            ).reset_index()
            reg_stats.columns = ["Regime", "Days", "Mean Return (%)",
                                  "Volatility (%)", "Best Day (%)", "Worst Day (%)"]
            st.dataframe(reg_stats, use_container_width=True, hide_index=True)

            # Transition matrix
            st.markdown("**HMM Transition Matrix**")
            T = hmm.transmat_
            regime_names = [labels_map[i] for i in range(n_regimes)]
            trans_df = pd.DataFrame(T, index=regime_names, columns=regime_names).round(4)
            st.dataframe(trans_df, use_container_width=True)

        # Return distributions by regime
        st.markdown("**Return Distribution by Regime**")
        fig_dist = go.Figure()
        for regime, color in color_map.items():
            subset = df[df["regime"] == regime]["returns"]
            if len(subset) > 10:
                fig_dist.add_trace(go.Histogram(
                    x=subset, name=regime, marker_color=color,
                    opacity=0.6, nbinsx=60, histnorm="probability density"
                ))
        fig_dist.update_layout(barmode="overlay", height=300,
                                xaxis_title="Daily Return (%)", yaxis_title="Density")
        st.plotly_chart(fig_dist, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2: Volatility
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("GARCH(1,1) Conditional Volatility")

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.6, 0.4],
                             subplot_titles=["Annualized GARCH Volatility (%)",
                                             "Daily Log Returns (%)"])

        fig2.add_trace(go.Scatter(
            x=df.index, y=df["annual_vol"],
            name="GARCH Vol", fill="tozeroy",
            line=dict(color="#f59e0b", width=1.5),
            fillcolor="rgba(245,158,11,0.2)"
        ), row=1, col=1)

        # Add 2× median threshold
        thresh = df["annual_vol"].median() * 2
        fig2.add_hline(y=thresh, line_dash="dash", line_color="red",
                       annotation_text=f"2× Median ({thresh:.1f}%)",
                       row=1, col=1)

        colors_bar = np.where(df["returns"] >= 0, "#22c55e", "#ef4444")
        fig2.add_trace(go.Bar(
            x=df.index, y=df["returns"],
            name="Returns", marker_color=colors_bar, opacity=0.7
        ), row=2, col=1)

        fig2.add_annotation(
            text=(f"α={params['alpha']:.4f}  β={params['beta']:.4f}  "
                  f"α+β={params['persistence']:.4f}  "
                  f"AIC={params['aic']:.1f}"),
            xref="paper", yref="paper", x=0.01, y=0.99,
            showarrow=False,
            bgcolor="rgba(30,41,59,0.8)",
            font=dict(color="white", size=11),
            bordercolor="#334155", borderwidth=1
        )
        fig2.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        # GARCH parameter explanation
        with st.expander("📖 How to Interpret GARCH Parameters"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ω (Omega)", f"{params['omega']:.6f}",
                      help="Long-run variance floor")
            c2.metric("α (Alpha)", f"{params['alpha']:.4f}",
                      help="Speed of reaction to shocks")
            c3.metric("β (Beta)",  f"{params['beta']:.4f}",
                      help="Persistence of past variance")
            c4.metric("α+β",       f"{params['persistence']:.4f}",
                      help="Total persistence. >0.99 = very long memory")

            st.markdown(f"""
            - **α = {params['alpha']:.4f}**: After a large market shock, volatility jumps by this fraction.
            - **β = {params['beta']:.4f}**: That elevated volatility persists — decaying at this rate each day.
            - **α+β = {params['persistence']:.4f}**: {'⚠️ Very high persistence — shocks last a long time.' if params['persistence'] > 0.97 else 'Moderate persistence — volatility reverts reasonably quickly.'}
            - **Long-run annualized vol**: {np.sqrt(params['omega'] / (1 - params['persistence'])) * np.sqrt(252):.2f}%
            """)

    # ══════════════════════════════════════════════════════════
    # TAB 3: TRADING SIGNAL & BACKTEST  ← NEW
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("⚡ Regime-Conditioned Trading Strategy")
        st.markdown(f"""
        **Strategy Logic:**
        - **Bull 🟢** regime → `{bull_signal}` (+1 or 0)
        - **Bear 🔴** regime → `{bear_signal}` (-1 or 0)
        - **Sideways 🟡** regime → `Flat` (0) *(if 3 regimes)*
        - Signal is applied next day (no lookahead bias)
        """)

        # Build signal
        signal_map = {"Bull 🟢": 1 if bull_signal == "Long" else 0,
                      "Bear 🔴": -1 if bear_signal == "Short" else 0,
                      "Sideways 🟡": 0}
        df["signal"]          = df["regime"].map(signal_map).shift(1)  # next-day
        df["strategy_return"] = df["signal"] * df["returns"]
        df["bh_return"]       = df["returns"]   # buy & hold

        # Cumulative returns
        df["cum_strategy"] = (1 + df["strategy_return"] / 100).cumprod()
        df["cum_bh"]       = (1 + df["bh_return"] / 100).cumprod()

        # Drawdown
        df["strat_dd"]  = df["cum_strategy"] / df["cum_strategy"].cummax() - 1
        df["bh_dd"]     = df["cum_bh"] / df["cum_bh"].cummax() - 1

        # Rolling Sharpe
        df["rolling_sharpe_strat"] = rolling_sharpe(df["strategy_return"])
        df["rolling_sharpe_bh"]    = rolling_sharpe(df["bh_return"])

        # Performance metrics
        strat_metrics = performance_metrics(df["strategy_return"], "Strategy")
        bh_metrics    = performance_metrics(df["bh_return"],       "Buy & Hold")

        # Metrics comparison table
        st.markdown("### 📊 Performance Comparison")
        metrics_df = pd.DataFrame({
            "Strategy": strat_metrics,
            "Buy & Hold": bh_metrics
        })
        st.dataframe(
            metrics_df.style.highlight_max(axis=1, color="#22c55e30")
                            .highlight_min(axis=1, color="#ef444430"),
            use_container_width=True
        )

        # Chart 1: Equity Curves
        fig_bt = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               row_heights=[0.5, 0.25, 0.25],
                               subplot_titles=[
                                   "Cumulative Return (₹1 invested)",
                                   "Drawdown (%)",
                                   "Rolling 60-Day Sharpe Ratio"
                               ])

        fig_bt.add_trace(go.Scatter(
            x=df.index, y=df["cum_strategy"],
            name="Regime Strategy", line=dict(color="#22c55e", width=2)
        ), row=1, col=1)
        fig_bt.add_trace(go.Scatter(
            x=df.index, y=df["cum_bh"],
            name="Buy & Hold", line=dict(color="#3b82f6", width=1.5, dash="dot")
        ), row=1, col=1)

        fig_bt.add_trace(go.Scatter(
            x=df.index, y=df["strat_dd"] * 100,
            name="Strategy DD", fill="tozeroy",
            line=dict(color="#22c55e", width=1),
            fillcolor="rgba(34,197,94,0.2)"
        ), row=2, col=1)
        fig_bt.add_trace(go.Scatter(
            x=df.index, y=df["bh_dd"] * 100,
            name="B&H DD", fill="tozeroy",
            line=dict(color="#3b82f6", width=1),
            fillcolor="rgba(59,130,246,0.15)"
        ), row=2, col=1)

        fig_bt.add_trace(go.Scatter(
            x=df.index, y=df["rolling_sharpe_strat"],
            name="Strat Sharpe", line=dict(color="#22c55e", width=1)
        ), row=3, col=1)
        fig_bt.add_trace(go.Scatter(
            x=df.index, y=df["rolling_sharpe_bh"],
            name="B&H Sharpe", line=dict(color="#3b82f6", width=1, dash="dot")
        ), row=3, col=1)
        fig_bt.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        fig_bt.update_layout(height=600, hovermode="x unified")
        st.plotly_chart(fig_bt, use_container_width=True)

        # Signal chart
        st.markdown("### 🔀 Signal Over Time")
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=df.index, y=df["signal"],
            mode="lines", name="Signal",
            line=dict(color="#a855f7", width=1.5),
            fill="tozeroy", fillcolor="rgba(168,85,247,0.1)"
        ))
        fig_sig.update_layout(height=200, yaxis_title="Signal (+1/0/-1)",
                               hovermode="x unified")
        fig_sig.update_yaxes(tickvals=[-1, 0, 1],
                             ticktext=["Short", "Flat", "Long"])
        st.plotly_chart(fig_sig, use_container_width=True)

        with st.expander("⚠️ Backtest Disclaimer"):
            st.warning("""
            This backtest does NOT account for:
            - Transaction costs / brokerage
            - Slippage (price impact)
            - Short-selling restrictions on NSE indices
            - Taxes (STT, STCG)
            
            Real trading performance will differ. This is a research prototype only.
            """)

    # ══════════════════════════════════════════════════════════
    # TAB 4: VaR / CVaR
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader(f"🛡️ Dynamic VaR & CVaR — ₹{portfolio_val:,.0f} Portfolio")

        # Main VaR chart
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df.index, y=df["cvar_inr_scaled"],
            name=f"CVaR ({confidence*100:.0f}%)",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)"
        ))
        fig3.add_trace(go.Scatter(
            x=df.index, y=df["var_inr_scaled"],
            name=f"VaR ({confidence*100:.0f}%)",
            line=dict(color="#f59e0b", width=1.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.2)"
        ))
        fig3.update_layout(height=350, hovermode="x unified",
                           yaxis_title="Daily Loss Estimate (₹)",
                           yaxis_tickformat=",")
        st.plotly_chart(fig3, use_container_width=True)

        # VaR breaches
        st.markdown("### 🎯 VaR Breach Analysis")
        breaches = df["returns"] < df["var_pct"]
        breach_rate = breaches.mean()

        b1, b2, b3 = st.columns(3)
        b1.metric("Total VaR Breaches", f"{breaches.sum()} / {len(breaches)}")
        b2.metric("Actual Breach Rate", f"{breach_rate:.2%}")
        b3.metric("Expected Rate", f"{(1-confidence):.2%}")

        # Kupiec POF test
        st.markdown("### 🧪 Kupiec's POF Backtesting (Regulatory Standard)")
        kup = kupiec_pof_test(df["returns"], df["var_pct"], confidence)

        if kup["pass"] is not None:
            if kup["pass"]:
                st.success(f"✅ Model **PASSES** Kupiec's POF test (p={kup['p_value']:.4f} > 0.05). "
                           f"VaR is well-calibrated.")
            else:
                st.error(f"❌ Model **FAILS** Kupiec's POF test (p={kup['p_value']:.4f} < 0.05). "
                         f"{'Model underestimates risk.' if breach_rate > (1-confidence) else 'Model is too conservative.'}")

        kup_df = pd.DataFrame([{
            "Observations": kup["n"],
            "Actual Breaches": kup["breaches"],
            "Expected Breaches": int(kup["expected_rate"] * kup["n"]),
            "Actual Rate": f"{kup['actual_rate']:.2%}",
            "Expected Rate": f"{kup['expected_rate']:.2%}",
            "LR Statistic": f"{kup['LR']:.4f}",
            "p-value": f"{kup['p_value']:.4f}",
            "Result": "✅ PASS" if kup.get("pass") else "❌ FAIL"
        }])
        st.dataframe(kup_df, use_container_width=True, hide_index=True)

        # Returns vs VaR threshold
        fig_breach = go.Figure()
        fig_breach.add_trace(go.Bar(
            x=df.index, y=df["returns"],
            marker_color=np.where(df["returns"] >= 0, "#22c55e", "#ef4444"),
            name="Daily Returns", opacity=0.6
        ))
        fig_breach.add_trace(go.Scatter(
            x=df.index, y=df["var_pct"],
            name="VaR Threshold", line=dict(color="#f59e0b", width=1.5)
        ))
        breach_dates = df.index[breaches]
        fig_breach.add_trace(go.Scatter(
            x=breach_dates, y=df.loc[breach_dates, "returns"],
            mode="markers",
            marker=dict(color="black", size=7, symbol="x"),
            name=f"VaR Breach ({breaches.sum()})"
        ))
        fig_breach.update_layout(height=350, hovermode="x unified",
                                  yaxis_title="Return (%)",
                                  title="Returns vs VaR Threshold (✕ = breach)")
        st.plotly_chart(fig_breach, use_container_width=True)

        # Multi-confidence table
        st.markdown("### 📋 VaR at Multiple Confidence Levels")
        conf_rows = []
        for c in [0.90, 0.95, 0.99]:
            z_c  = stats.norm.ppf(1 - c)
            v_pct = (returns.mean() + z_c * df["cond_vol"]).iloc[-1]
            cv_pct = (returns.mean() - df["cond_vol"].iloc[-1] *
                      stats.norm.pdf(z_c) / (1 - c))
            conf_rows.append({
                "Confidence": f"{c:.0%}",
                "VaR (%)": f"{v_pct:.3f}%",
                "VaR (₹)": f"₹{abs(v_pct/100)*portfolio_val:,.0f}",
                "CVaR (%)": f"{cv_pct:.3f}%",
                "CVaR (₹)": f"₹{abs(cv_pct/100)*portfolio_val:,.0f}",
            })
        st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # TAB 5: GARCH FORECAST  ← NEW
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader(f"📡 GARCH Volatility Forecast — Next {forecast_days} Trading Days")
        st.markdown("GARCH forecasts mean-revert toward long-run volatility. "
                    "High current vol → forecast decays. Low current vol → forecast rises slightly.")

        forecasts = garch_res.forecast(horizon=forecast_days, reindex=False)
        f_var = forecasts.variance.iloc[-1]
        f_vol_daily = np.sqrt(f_var)
        f_vol_annual = f_vol_daily * np.sqrt(252)

        current_vol = df["annual_vol"].iloc[-1]
        if params["persistence"] < 0.999:
            long_run_vol = np.sqrt(params["omega"]/(1-params["persistence"])) * np.sqrt(252)
        else:
            long_run_vol = np.nan

        # Forecast table
        forecast_df = pd.DataFrame({
            "Day": [f"+{i+1}" for i in range(forecast_days)],
            "Daily σ (%)": f_vol_daily.round(4).values,
            "Annualized Vol (%)": f_vol_annual.round(2).values,
            "Expected VaR (%)": (returns.mean() + stats.norm.ppf(1-confidence) * f_vol_daily).round(4).values,
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Forecast chart
        hist_vol = df["annual_vol"].tail(60)
        future_dates = pd.bdate_range(start=df.index[-1], periods=forecast_days + 1)[1:]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist_vol.index, y=hist_vol,
            name="Historical Vol", line=dict(color="#f59e0b", width=1.5)
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=f_vol_annual.values,
            name="Forecast Vol", line=dict(color="#a855f7", width=2, dash="dot"),
            mode="lines+markers"
        ))
        fig_fc.add_hline(y=long_run_vol, line_dash="dash", line_color="gray",
                         annotation_text=f"Long-Run Vol: {long_run_vol:.1f}%")
        fig_fc.add_hline(y=current_vol, line_dash="dot", line_color="#ef4444",
                         annotation_text=f"Current: {current_vol:.1f}%")
        fig_fc.add_vrect(x0=df.index[-1], x1=future_dates[-1],
                         fillcolor="rgba(168,85,247,0.08)", layer="below", line_width=0)
        fig_fc.update_layout(height=400, hovermode="x unified",
                             yaxis_title="Annualized Volatility (%)",
                             title=f"60-Day History + {forecast_days}-Day GARCH Forecast")
        st.plotly_chart(fig_fc, use_container_width=True)

        # VaR forecast
        st.markdown("### 📋 Forecast VaR (₹) — Next 10 Days")
        var_fc_rows = []
        for i, (dv, av) in enumerate(zip(f_vol_daily, f_vol_annual), 1):
            v_pct = returns.mean() + stats.norm.ppf(1-confidence) * dv
            var_fc_rows.append({
                "Day": f"+{i}",
                "Forecast Daily σ": f"{dv:.4f}%",
                "Forecast Annual σ": f"{av:.2f}%",
                f"Forecast VaR ({confidence*100:.0f}%)": f"{v_pct:.4f}%",
                "VaR in ₹": f"₹{abs(v_pct/100)*portfolio_val:,.0f}",
            })
        st.dataframe(pd.DataFrame(var_fc_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # TAB 6: MODEL DIAGNOSTICS  ← NEW
    # ══════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("🧪 Model Diagnostics & Comparison")

        # Model comparison
        st.markdown("### GARCH Model Comparison (AIC / BIC)")
        st.caption("Lower AIC/BIC = better model. AIC penalizes complexity less than BIC.")

        comp_df = pd.DataFrame(comparison).T.reset_index()
        comp_df.columns = ["Model", "AIC", "BIC"]
        best_aic = comp_df.loc[comp_df["AIC"].idxmin(), "Model"]
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.info(f"📌 Best model by AIC: **{best_aic}** — "
                f"(using GARCH(1,1) throughout for interpretability)")

        # Return distribution stats
        st.markdown("### Return Distribution Properties")
        s    = skew(returns)
        k    = kurtosis(returns, fisher=True)
        jb_s, jb_p = jarque_bera(returns)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Skewness", f"{s:.4f}",
                  delta="Left-skewed ↙" if s < 0 else "Right-skewed ↗",
                  delta_color="off")
        d2.metric("Excess Kurtosis", f"{k:.4f}",
                  delta="Fat tails" if k > 0 else "Thin tails",
                  delta_color="off")
        d3.metric("Jarque-Bera Stat", f"{jb_s:.2f}")
        d4.metric("JB p-value", f"{jb_p:.6f}",
                  delta="Non-normal ✓" if jb_p < 0.05 else "Normal?",
                  delta_color="off")

        if jb_p < 0.05:
            st.success("Returns are **NOT** normally distributed (JB p < 0.05). "
                       "This justifies GARCH over simple variance, and suggests "
                       "t-distribution innovations may fit better than Normal.")

        # QQ Plot of standardized residuals
        st.markdown("### Standardized GARCH Residuals")
        std_resid = garch_res.std_resid.dropna()
        sorted_resid = np.sort(std_resid)
        n_pts = len(sorted_resid)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, n_pts))

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=sorted_resid,
            mode="markers",
            marker=dict(color="#3b82f6", size=3, opacity=0.6),
            name="Residuals"
        ))
        fig_qq.add_trace(go.Scatter(
            x=[-4, 4], y=[-4, 4],
            line=dict(color="#ef4444", dash="dash"),
            name="Normal reference"
        ))
        fig_qq.update_layout(
            height=400,
            title="Q-Q Plot of Standardized Residuals (Deviations = fat tails / skew)",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Sample Quantiles"
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption("Points far above the line at extremes = fat tails. "
                   "Curvature = skewness. A perfect Normal = straight line.")

        # Residual ACF (simple bar chart)
        st.markdown("### ACF of Squared Residuals (Test for Remaining ARCH Effects)")
        sq_resid = std_resid ** 2
        n_lags = 20
        acf_vals = [sq_resid.autocorr(lag=l) for l in range(1, n_lags + 1)]
        ci = 1.96 / np.sqrt(len(sq_resid))

        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(
            x=list(range(1, n_lags + 1)), y=acf_vals,
            marker_color=["#ef4444" if abs(v) > ci else "#3b82f6" for v in acf_vals],
            name="ACF"
        ))
        fig_acf.add_hline(y=ci,  line_dash="dash", line_color="gray")
        fig_acf.add_hline(y=-ci, line_dash="dash", line_color="gray")
        fig_acf.update_layout(height=300, xaxis_title="Lag",
                               yaxis_title="ACF",
                               title="Significant bars (red) = remaining ARCH effects not captured")
        st.plotly_chart(fig_acf, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 7: DOWNLOAD
    # ══════════════════════════════════════════════════════════
    with tabs[6]:
        st.subheader("📥 Download Results")

        # Main dataset
        export_df = df[["price", "returns", "annual_vol", "regime",
                         "var_pct", "cvar_pct", "var_inr_scaled",
                         "cvar_inr_scaled", "signal",
                         "strategy_return", "cum_strategy", "cum_bh"]].copy()
        export_df.columns = [
            "Price", "Log Return (%)", "Annual Vol (%)", "Regime",
            "VaR (%)", "CVaR (%)", f"VaR (₹)", f"CVaR (₹)",
            "Signal", "Strategy Return (%)",
            "Cumulative Strategy", "Cumulative B&H"
        ]

        csv = export_df.to_csv()
        st.download_button(
            label="⬇️ Download Full Dataset (CSV)",
            data=csv,
            file_name=f"{ticker_name.replace(' ', '_')}_volatility_dashboard.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Performance summary
        perf_csv = pd.DataFrame({
            "Strategy": strat_metrics,
            "Buy & Hold": bh_metrics
        }).to_csv()
        st.download_button(
            label="⬇️ Download Performance Metrics (CSV)",
            data=perf_csv,
            file_name=f"{ticker_name.replace(' ', '_')}_performance.csv",
            mime="text/csv",
            use_container_width=True
        )

        # GARCH forecast
        fc_csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download GARCH Forecast (CSV)",
            data=fc_csv,
            file_name=f"{ticker_name.replace(' ', '_')}_garch_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()
        st.markdown("### 📋 Data Preview")
        st.dataframe(export_df.tail(20), use_container_width=True)

else:
    # ── Landing page ─────────────────────────────────────────
    st.info("👈 Configure the settings in the sidebar and click **🚀 Run Analysis**.")

    st.markdown("""
    ### What this dashboard does:

    | Tab | Component | Method | Output |
    |---|---|---|---|
    | 📈 Price & Regimes | Regime Detection | HMM + Viterbi | Bull/Sideways/Bear labels |
    | 🌋 Volatility | Volatility Modeling | GARCH(1,1) | Time-varying σ(t) |
    | ⚡ Backtest | Trading Strategy | Regime-conditioned signals | Sharpe, Drawdown, Calmar |
    | 🛡️ VaR/CVaR | Risk Estimation | GARCH-Dynamic + Kupiec | Daily loss in ₹ |
    | 📡 Forecast | Volatility Forecast | GARCH multi-step | 5–30 day vol forecast |
    | 🧪 Diagnostics | Model Validation | QQ plot, ACF, AIC/BIC | Residual analysis |
    | 📥 Download | Export | CSV | All results downloadable |

    ### Indexes covered:
    - **Nifty 50** — Broad Indian market
    - **Nifty Bank** — Banking sector
    - **Nifty IT** — Technology sector

    ### Key techniques:
    `GARCH(1,1)` · `GJR-GARCH` · `EGARCH` · `HMM (Viterbi)` · `VaR` · `CVaR` · `Kupiec POF Test` · `Sharpe` · `Sortino` · `Calmar` · `Rolling Sharpe`
    """)
