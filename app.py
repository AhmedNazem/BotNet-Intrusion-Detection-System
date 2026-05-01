import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BotNet IDS — LSTM Detector",
    page_icon="🛡️",
    layout="wide",
)

FEATURE_COLS = [
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
    'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot',
    'bwd_data_pkts_tot', 'flow_pkts_per_sec', 'down_up_ratio',
    'fwd_header_size_tot', 'fwd_header_size_min', 'fwd_header_size_max',
    'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max',
    'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count',
    'fwd_PSH_flag_count', 'bwd_PSH_flag_count', 'flow_ACK_flag_count',
    'fwd_URG_flag_count', 'bwd_URG_flag_count', 'flow_CWR_flag_count',
    'flow_ECE_flag_count', 'fwd_pkts_payload.min', 'fwd_pkts_payload.max',
    'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.std',
    'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot',
    'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min',
    'flow_pkts_payload.max', 'flow_pkts_payload.tot', 'flow_pkts_payload.avg',
    'flow_pkts_payload.std', 'fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot',
    'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot',
    'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max',
    'flow_iat.tot', 'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second',
    'fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes',
    'bwd_subflow_bytes', 'fwd_bulk_bytes', 'bwd_bulk_bytes', 'fwd_bulk_packets',
    'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate', 'active.min',
    'active.max', 'active.tot', 'active.avg', 'active.std', 'idle.min',
    'idle.max', 'idle.tot', 'idle.avg', 'idle.std', 'fwd_init_window_size',
    'bwd_init_window_size', 'fwd_last_window_size', 'bwd_last_window_size',
    'traffic',
]

ATTACK_TYPES = {
    'normal':       ('Normal Traffic',  '#22c55e'),
    'camoverflow':  ('Camera Overflow', '#ef4444'),
    'netscan':      ('Network Scan',    '#f97316'),
    'rudeadyet':    ('RUDeadYet DoS',   '#dc2626'),
    'apachekiller': ('Apache Killer',   '#b91c1c'),
    'mqttmalaria':  ('MQTT Malaria',    '#9333ea'),
    'slowloris':    ('Slowloris DoS',   '#ec4899'),
    'arpspoofing':  ('ARP Spoofing',    '#f59e0b'),
    'slowread':     ('Slow Read DoS',   '#6366f1'),
}


# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model  = keras.saving.load_model('ids_lstm_model_patched.keras')
    scaler = joblib.load('scaler.pkl')
    le     = joblib.load('label_encoder.pkl')
    return model, scaler, le


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset():
    if os.path.exists('archive/output.csv'):
        return pd.read_csv('archive/output.csv')
    return pd.read_csv('sample_data.csv')


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_batch(df_raw, model, scaler, le):
    df = df_raw[FEATURE_COLS].copy()
    df = df.replace('-', 0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    X = df.astype('float32').values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    probs = model.predict(X_reshaped, verbose=0).flatten()
    indices = (probs > 0.5).astype(int)
    labels = le.inverse_transform(indices)
    return labels, probs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=80)
    st.title("BotNet IDS")
    st.caption("LSTM-based Intrusion Detection System")
    st.divider()

    st.subheader("Model Architecture")
    st.markdown("""
    | Layer    | Details               |
    |----------|-----------------------|
    | Input    | (1, 90) — time step   |
    | LSTM     | 8 units, L2 reg       |
    | Dropout  | 30%                   |
    | Dense    | 1 unit, sigmoid       |
    """)
    st.divider()

    st.subheader("Attack Types in Dataset")
    for key, (name, color) in ATTACK_TYPES.items():
        if key == 'normal':
            st.markdown(f"🟢 **{name}**")
        else:
            st.markdown(f"🔴 {name}")

    st.divider()
    st.caption("College Project — IDS with LSTM")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ BotNet Intrusion Detection System")
st.markdown("Real-time network traffic classification using an **LSTM deep learning model**.")
st.divider()

model, scaler, le = load_artifacts()

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Type", "LSTM")
col2.metric("Input Features", "90")
col3.metric("Output Classes", "2")
col4.metric("Optimizer", "Adam")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dataset Demo", "📁 Upload CSV", "📈 Dataset Overview"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Demo
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Test with Real Dataset Samples")
    st.markdown("Pick how many rows to run through the model and see predictions vs ground truth.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_samples = st.slider("Number of samples", min_value=10, max_value=500, value=100, step=10)

    df_data = load_dataset()

    if os.path.exists('archive/output.csv'):
        st.info(f"Using full dataset — {len(df_data):,} rows loaded.")
    else:
        st.warning("Full dataset not found locally. Using sample_data.csv (1,350 rows). Download the full dataset to use all 3.2M records.")

    if st.button("▶ Run Predictions", type="primary", key="run_demo"):
        sample_df = df_data.sample(n=n_samples, random_state=42).reset_index(drop=True)

        with st.spinner("Running model inference…"):
            labels, probs = predict_batch(sample_df, model, scaler, le)

        sample_df['Predicted']   = labels
        sample_df['Confidence']  = (probs * 100).round(2)
        sample_df['Actual']      = sample_df['is_attack'].map({1: 'Attack', 0: 'Normal'})
        sample_df['Correct']     = sample_df['Predicted'] == sample_df['Actual']
        sample_df['Traffic Type'] = sample_df['traffic'].map(
            lambda x: ATTACK_TYPES.get(x, (x, '#888'))[0]
        )

        accuracy = sample_df['Correct'].mean() * 100

        # ── Summary metrics ──
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.1f}%")
        m2.metric("Attacks Detected",
                  int((sample_df['Predicted'] == 'Attack').sum()),
                  delta=None)
        m3.metric("Normal Traffic",
                  int((sample_df['Predicted'] == 'Normal').sum()))
        m4.metric("Samples Tested", n_samples)

        st.divider()

        # ── Charts row ──
        c1, c2 = st.columns(2)

        with c1:
            pred_counts = sample_df['Predicted'].value_counts().reset_index()
            pred_counts.columns = ['Label', 'Count']
            fig_pie = px.pie(
                pred_counts, names='Label', values='Count',
                title='Prediction Breakdown',
                color='Label',
                color_discrete_map={'Attack': '#ef4444', 'Normal': '#22c55e'},
                hole=0.4,
            )
            fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            traffic_counts = sample_df['Traffic Type'].value_counts().reset_index()
            traffic_counts.columns = ['Type', 'Count']
            fig_bar = px.bar(
                traffic_counts, x='Type', y='Count',
                title='Traffic Type Distribution',
                color='Count',
                color_continuous_scale='Reds',
            )
            fig_bar.update_layout(
                margin=dict(t=40, b=0),
                xaxis_tickangle=-30,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Confidence histogram ──
        fig_hist = px.histogram(
            sample_df, x='Confidence', color='Predicted',
            title='Model Confidence Distribution (%)',
            color_discrete_map={'Attack': '#ef4444', 'Normal': '#22c55e'},
            nbins=30, barmode='overlay', opacity=0.75,
        )
        fig_hist.update_layout(margin=dict(t=40))
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Results table ──
        st.subheader("Detailed Results")

        display_df = sample_df[['Traffic Type', 'Actual', 'Predicted', 'Confidence', 'Correct']].copy()

        def color_row(row):
            if not row['Correct']:
                return ['background-color: #fef2f2'] * len(row)
            if row['Predicted'] == 'Attack':
                return ['background-color: #fff7ed'] * len(row)
            return ['background-color: #f0fdf4'] * len(row)

        styled = display_df.style.apply(color_row, axis=1).format({'Confidence': '{:.2f}%'})
        st.dataframe(styled, use_container_width=True, height=400)

        st.caption("🟢 Green = Normal (correct)  |  🟠 Orange = Attack (correct)  |  🔴 Red = Misclassified")


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload CSV
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload Your Own CSV File")
    st.markdown(f"The CSV must contain these **{len(FEATURE_COLS)} feature columns** (same format as the training dataset).")

    with st.expander("Show required column names"):
        st.code(", ".join(FEATURE_COLS))

    n_upload = st.slider("Rows to sample from file", min_value=50, max_value=1000, value=200, step=50,
                         help="Large files are sampled to keep things fast.")

    uploaded = st.file_uploader("Drop a CSV file here", type=["csv"])

    if uploaded:
        try:
            with st.spinner("Reading file…"):
                up_df = pd.read_csv(uploaded, nrows=n_upload)
            st.success(f"Loaded {len(up_df):,} rows from file.")

            missing = [c for c in FEATURE_COLS if c not in up_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("▶ Run Predictions", type="primary", key="run_upload"):
                    with st.spinner("Running inference…"):
                        labels, probs = predict_batch(up_df, model, scaler, le)

                    up_df['Predicted']  = labels
                    up_df['Confidence'] = (probs * 100).round(2)

                    u1, u2 = st.columns(2)
                    u1.metric("Attacks Found", int((up_df['Predicted'] == 'Attack').sum()))
                    u2.metric("Normal Flows",  int((up_df['Predicted'] == 'Normal').sum()))

                    fig = px.pie(
                        up_df['Predicted'].value_counts().reset_index(),
                        names='Predicted', values='count',
                        color='Predicted',
                        color_discrete_map={'Attack': '#ef4444', 'Normal': '#22c55e'},
                        hole=0.4, title='Prediction Breakdown',
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        up_df[['Predicted', 'Confidence']].style.format({'Confidence': '{:.2f}%'}),
                        use_container_width=True,
                    )

                    csv_out = up_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇ Download Results CSV", csv_out, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error reading file: {e}")


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Overview
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Dataset Overview")
    st.markdown("Summary of the full training dataset (3.2M rows).")

    raw_counts = {
        'Camera Overflow (camoverflow)': 1640039,
        'Normal Traffic':                766915,
        'Network Scan (netscan)':        467093,
        'RUDeadYet DoS (rudeadyet)':     131081,
        'Apache Killer (apachekiller)':   84579,
        'MQTT Malaria (mqttmalaria)':     69623,
        'Slowloris DoS (slowloris)':      63608,
        'ARP Spoofing (arpspoofing)':     11236,
        'Slow Read DoS (slowread)':        9014,
    }
    total = sum(raw_counts.values())

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Records",  f"{total:,}")
    d2.metric("Attack Records", f"{total - 766915:,}")
    d3.metric("Normal Records", "766,915")

    counts_df = pd.DataFrame(list(raw_counts.items()), columns=['Type', 'Count'])
    counts_df['Percentage'] = (counts_df['Count'] / total * 100).round(2)

    colors = ['#22c55e' if 'Normal' in t else '#ef4444' for t in counts_df['Type']]

    fig_overview = px.bar(
        counts_df, x='Type', y='Count',
        title='Traffic Type Distribution in Dataset',
        text='Percentage',
        color='Type',
        color_discrete_sequence=colors,
    )
    fig_overview.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_overview.update_layout(
        xaxis_tickangle=-25,
        showlegend=False,
        margin=dict(t=60, b=120),
        yaxis_title='Number of Records',
    )
    st.plotly_chart(fig_overview, use_container_width=True)

    st.dataframe(counts_df.style.format({'Count': '{:,}', 'Percentage': '{:.2f}%'}),
                 use_container_width=True)
