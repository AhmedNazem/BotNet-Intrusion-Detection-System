# 🛡️ BotNet Intrusion Detection System (IDS)

A deep learning project that uses an **LSTM neural network** to detect botnet attacks in network traffic.  
Built with Python, Keras, and Streamlit. Designed for a college cybersecurity course.

---

## 📋 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [How Does It Work? (Simple Explanation)](#2-how-does-it-work-simple-explanation)
3. [Project Files Explained](#3-project-files-explained)
4. [Libraries Used](#4-libraries-used)
5. [The Dataset](#5-the-dataset)
6. [The Machine Learning Model](#6-the-machine-learning-model)
7. [The 90 Input Features Explained](#7-the-90-input-features-explained)
8. [The Attack Types](#8-the-attack-types)
9. [Code Walkthrough](#9-code-walkthrough)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [How to Use the Dashboard](#11-how-to-use-the-dashboard)
12. [Understanding the Results](#12-understanding-the-results)
13. [Common Questions & Answers](#13-common-questions--answers)

---

## 1. What Is This Project?

This project is an **Intrusion Detection System (IDS)** — a system that watches network traffic and decides whether it is **normal** (safe) or an **attack** (dangerous).

### Real-world analogy
Think of it like a security guard at a building entrance. Every person (network connection) that walks in is checked. The guard has learned from thousands of past cases what a suspicious person looks like vs. a normal visitor. Our model is that guard — trained on 3.2 million network connections.

### What problem does it solve?
Botnets are networks of infected computers controlled by hackers. They are used to:
- Crash websites (DoS attacks)
- Spy on users
- Spread malware

Our model detects when a computer is being used as part of a botnet attack — in real time.

---

## 2. How Does It Work? (Simple Explanation)

```
Raw Network Traffic
       ↓
Extract 90 measurements from each connection
(speed, size, timing, flags, etc.)
       ↓
Clean & normalize the numbers
       ↓
Feed into the LSTM neural network
       ↓
Model outputs a score (0.0 to 1.0)
       ↓
Score > 0.5  →  Normal
Score < 0.5  →  Attack
```

### What is an LSTM?
**LSTM** stands for **Long Short-Term Memory**. It is a type of neural network that is very good at learning patterns in sequences — like how network packets arrive over time.

Normal traffic has a rhythm. Attack traffic breaks that rhythm. The LSTM learns to spot the difference.

---

## 3. Project Files Explained

```
BotNet/
│
├── app.py                        ← Main dashboard (run this to use the app)
├── test_model.py                 ← Script to test the model from terminal
├── START APP.bat                 ← Double-click to launch the dashboard
│
├── ids_lstm_model.keras          ← Original trained LSTM model
├── ids_lstm_model_patched.keras  ← Fixed version of the model (use this one)
├── scaler.pkl                    ← Saved data normalizer (MinMaxScaler)
├── label_encoder.pkl             ← Converts numbers to Attack/Normal labels
│
├── archive/
│   └── output.csv               ← Full dataset (3.2 million network flows)
│
├── botnet-Maaref-lstm.ipynb     ← Jupyter notebook used to train the model
├── botnet_project_explanation.md ← Extra project notes
├── README.md                    ← This file
│
└── .streamlit/
    └── config.toml              ← Streamlit settings (allows large file uploads)
```

### File details

| File | What it is | Why it exists |
|------|-----------|---------------|
| `app.py` | The visual dashboard | So users can interact with the model without writing code |
| `test_model.py` | Terminal-based test script | Quick way to verify the model works |
| `ids_lstm_model_patched.keras` | The trained neural network | This is the brain of the system |
| `scaler.pkl` | A MinMaxScaler object | Makes all numbers fit between 0 and 1 before feeding to model |
| `label_encoder.pkl` | A LabelEncoder object | Converts model output (0 or 1) to the words "Attack" or "Normal" |
| `output.csv` | Training & test data | 3.2 million real network flow records |
| `START APP.bat` | Windows batch file | Lets non-technical users launch the app by double-clicking |

---

## 4. Libraries Used

### Core Libraries

| Library | What it does | Why we use it |
|---------|-------------|---------------|
| **Keras** | Build and run neural networks | Our LSTM model was built and saved using Keras |
| **TensorFlow** | Backend engine for Keras | Powers the math behind the neural network |
| **NumPy** | Fast number operations | Reshapes and processes arrays for the model |
| **Pandas** | Load and manage data tables | Reads CSV files and organizes the dataset |
| **Joblib** | Save and load Python objects | Loads the scaler and label encoder from `.pkl` files |
| **Scikit-learn** | Machine learning tools | Provides MinMaxScaler and LabelEncoder |

### Dashboard Libraries

| Library | What it does | Why we use it |
|---------|-------------|---------------|
| **Streamlit** | Turns Python code into a web app | Creates the interactive dashboard without HTML or JavaScript |
| **Plotly** | Interactive charts | Creates the pie charts, bar charts, and histograms in the dashboard |

### Installation

All libraries are installed inside the `.venv` virtual environment:

```bash
.venv\Scripts\python.exe -m pip install keras tensorflow numpy pandas joblib scikit-learn streamlit plotly
```

---

## 5. The Dataset

**File:** `archive/output.csv`  
**Size:** ~1.4 GB  
**Rows:** 3,243,188 network flow records  
**Columns:** 102 (90 features + metadata + labels)

### What is a "network flow"?
Every time your computer communicates with another computer (loading a webpage, sending an email, etc.), that is a **network flow**. Each flow has measurable properties — how fast, how big, how long, what signals were sent.

### Dataset breakdown

| Traffic Type | Records | Percentage |
|-------------|---------|------------|
| Camera Overflow (camoverflow) | 1,640,039 | 50.57% |
| Normal Traffic | 766,915 | 23.65% |
| Network Scan (netscan) | 467,093 | 14.40% |
| RUDeadYet DoS (rudeadyet) | 131,081 | 4.04% |
| Apache Killer (apachekiller) | 84,579 | 2.61% |
| MQTT Malaria (mqttmalaria) | 69,623 | 2.15% |
| Slowloris DoS (slowloris) | 63,608 | 1.96% |
| ARP Spoofing (arpspoofing) | 11,236 | 0.35% |
| Slow Read DoS (slowread) | 9,014 | 0.28% |

### Label columns

| Column | Values | Meaning |
|--------|--------|---------|
| `traffic` | `normal`, `camoverflow`, `netscan`, etc. | The specific traffic type |
| `is_attack` | `0` or `1` | `0` = Normal, `1` = Attack |

---

## 6. The Machine Learning Model

**File:** `ids_lstm_model_patched.keras`

### Architecture

```
Input Layer  →  shape: (1, 90)
                1 time step, 90 features
       ↓
LSTM Layer   →  8 units
                activation: tanh
                L2 regularization (prevents overfitting)
       ↓
Dropout      →  30% of neurons randomly turned off during training
                (prevents the model from memorizing instead of learning)
       ↓
Dense Layer  →  1 unit
                activation: sigmoid
                outputs a value between 0.0 and 1.0
       ↓
Output       →  0.0 → Attack  |  1.0 → Normal
```

### Model parameters

| Setting | Value |
|---------|-------|
| Total parameters | 9,533 |
| Optimizer | Adam (lr=0.001) |
| Loss function | Binary Crossentropy |
| Metric | Accuracy |
| Input shape | (None, 1, 90) |
| Output | Single sigmoid score |

### What is the sigmoid output?

The sigmoid function always outputs a number between 0 and 1:

- **Output = 0.95** → 95% chance of being Normal → classified as **Normal**
- **Output = 0.50** → model is unsure (50/50)
- **Output = 0.02** → 2% chance of Normal → very likely **Attack**

We use **0.5 as the threshold**:
- `output > 0.5` → **Normal**
- `output ≤ 0.5` → **Attack**

### Why LSTM?

LSTM networks are designed to work with **sequences** — data that comes in order over time. Network traffic is sequential: packets arrive one after another. The LSTM can learn that normal traffic has a regular pattern, and attack traffic breaks that pattern.

### What is L2 Regularization?

L2 regularization is a technique that adds a penalty for large weight values in the LSTM layer. This prevents the model from **overfitting** — meaning it stops the model from just memorizing the training data and forces it to learn generalizable patterns.

### What is Dropout?

Dropout (30%) randomly disables 30% of neurons during each training step. This forces the network to not rely on any single neuron, making it more robust. Think of it like studying for an exam without knowing which specific questions will appear.

---

## 7. The 90 Input Features Explained

The model uses exactly **90 features** extracted from each network connection.

### Group 1 — Connection Identity

| Feature | Simple Meaning |
|---------|---------------|
| `proto` | Protocol used: TCP, UDP, ICMP (encoded as a number) |
| `service` | Application service: HTTP, DNS, SSH (encoded as a number) |
| `conn_state` | Did the connection succeed, fail, or get rejected? (encoded) |
| `duration` | How many seconds the connection lasted |
| `orig_bytes` | Bytes sent by the source (sender) |
| `resp_bytes` | Bytes sent back by the destination (receiver) |
| `missed_bytes` | Data that was lost or missed during transfer |
| `orig_pkts` | Number of packets sent by source |
| `resp_pkts` | Number of packets sent by destination |
| `orig_ip_bytes` | Total bytes including network headers (source) |
| `resp_ip_bytes` | Total bytes including network headers (destination) |

### Group 2 — Flow Statistics

| Feature | Simple Meaning |
|---------|---------------|
| `flow_duration` | Total duration of the network flow |
| `fwd_pkts_tot` | Total packets going forward (source → destination) |
| `bwd_pkts_tot` | Total packets going backward (destination → source) |
| `fwd_data_pkts_tot` | Forward packets that carried actual data |
| `bwd_data_pkts_tot` | Backward packets that carried actual data |
| `fwd_pkts_per_sec` | Forward packets per second (speed) |
| `bwd_pkts_per_sec` | Backward packets per second |
| `flow_pkts_per_sec` | Total packets per second (both directions) |
| `down_up_ratio` | Ratio of download vs upload — attacks often have abnormal ratios |
| `payload_bytes_per_second` | How fast actual data (not headers) was transferred |

### Group 3 — Packet Sizes (Payload)

> Each group has 5 measurements: min, max, total, average, standard deviation

| Feature Group | Simple Meaning |
|--------------|---------------|
| `fwd_pkts_payload.*` | Size of data in forward packets |
| `bwd_pkts_payload.*` | Size of data in backward packets |
| `flow_pkts_payload.*` | Size of data across the entire flow |

### Group 4 — TCP Flags

> TCP flags are control signals that computers send to each other. Attackers misuse them.

| Feature | Flag Meaning |
|---------|-------------|
| `flow_FIN_flag_count` | FIN = "I'm closing the connection" |
| `flow_SYN_flag_count` | SYN = "I want to connect" — many SYNs = SYN flood attack |
| `flow_RST_flag_count` | RST = "Connection reset by force" |
| `fwd_PSH_flag_count` | PSH = "Send this data immediately, don't buffer" |
| `bwd_PSH_flag_count` | Same but for backward direction |
| `flow_ACK_flag_count` | ACK = "I received your data" |
| `fwd_URG_flag_count` | URG = "This data is urgent" |
| `bwd_URG_flag_count` | Same for backward |
| `flow_CWR_flag_count` | CWR = "Reducing speed due to congestion" |
| `flow_ECE_flag_count` | ECE = "Network is getting congested" |

### Group 5 — Inter-Arrival Time (IAT)

> IAT = the time gap between consecutive packets. Attacks have abnormal gaps.

| Feature Group | Simple Meaning |
|--------------|---------------|
| `fwd_iat.*` | Time gaps between forward packets (min/max/total/avg/std) |
| `bwd_iat.*` | Time gaps between backward packets |
| `flow_iat.*` | Time gaps across the whole flow |

### Group 6 — Header Sizes

| Feature | Simple Meaning |
|---------|---------------|
| `fwd_header_size_tot/min/max` | Size of the technical envelope on forward packets |
| `bwd_header_size_tot/min/max` | Same for backward packets |

### Group 7 — Subflows & Bulk Transfers

| Feature | Simple Meaning |
|---------|---------------|
| `fwd_subflow_pkts/bytes` | Packets/bytes in sub-sections of the forward flow |
| `bwd_subflow_pkts/bytes` | Same for backward |
| `fwd_bulk_bytes/packets/rate` | Large continuous data bursts sent forward |
| `bwd_bulk_bytes/packets/rate` | Large continuous data bursts backward |

### Group 8 — Active & Idle Times

| Feature | Simple Meaning |
|---------|---------------|
| `active.min/max/tot/avg/std` | How long the connection was actively sending data |
| `idle.min/max/tot/avg/std` | How long the connection sat idle (doing nothing) |

### Group 9 — Window Sizes

| Feature | Simple Meaning |
|---------|---------------|
| `fwd_init_window_size` | Buffer size declared at connection start (sender) |
| `bwd_init_window_size` | Buffer size declared at connection start (receiver) |
| `fwd_last_window_size` | Buffer size at the end of the connection (sender) |
| `bwd_last_window_size` | Buffer size at the end (receiver) |

### Group 10 — Traffic Label (encoded)

| Feature | Simple Meaning |
|---------|---------------|
| `traffic` | The attack category name, encoded as a number |

---

## 8. The Attack Types

| Attack Name | Code | What It Does |
|------------|------|-------------|
| **Camera Overflow** | `camoverflow` | Floods IP cameras with traffic to crash them |
| **Network Scan** | `netscan` | Scans a network to find open ports and vulnerable devices |
| **RUDeadYet DoS** | `rudeadyet` | Sends specially crafted packets to crash a server |
| **Apache Killer** | `apachekiller` | Exploits Apache web server using malformed HTTP requests |
| **MQTT Malaria** | `mqttmalaria` | Attacks IoT devices using the MQTT messaging protocol |
| **Slowloris** | `slowloris` | Opens many connections to a web server and keeps them open very slowly, exhausting resources |
| **ARP Spoofing** | `arpspoofing` | Tricks devices on a network into sending traffic to the attacker |
| **Slow Read DoS** | `slowread` | Downloads data from a server extremely slowly to hold connections open |

---

## 9. Code Walkthrough

### `test_model.py` — Terminal Test Script

```python
import keras
import joblib
import pandas as pd

# Step 1: Load the model and its tools
model  = keras.saving.load_model('ids_lstm_model_patched.keras')
scaler = joblib.load('scaler.pkl')       # normalizes numbers
le     = joblib.load('label_encoder.pkl') # converts 0/1 to Attack/Normal
```

> **Why load 3 files?** The model only understands numbers between 0 and 1.
> The scaler converts raw numbers to that range.
> The label encoder converts the model's 0/1 output to readable words.

```python
# Step 2: Clean the input data
sample_df = sample_df.replace('-', 0)  # replace missing values with 0
for col in sample_df.columns:
    sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(0)
```

> **Why clean data?** Real network logs contain `-` for missing values.
> The model only accepts numbers, so we convert everything.

```python
# Step 3: Scale the features
X_scaled = scaler.transform(X.values)
```

> **Why scale?** Different features have very different ranges.
> `flow_duration` might be 0.002 seconds, while `orig_bytes` might be 13,000.
> The scaler squishes everything to 0–1 so the model treats all features fairly.

```python
# Step 4: Reshape for LSTM
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
# Shape becomes: (number of rows, 1 time step, 90 features)
```

> **Why reshape?** LSTMs expect 3D input: `(samples, time_steps, features)`.
> We use 1 time step because we classify one flow at a time.

```python
# Step 5: Predict
prediction_prob = model.predict(X_reshaped, verbose=0)
result_idx = (prediction_prob > 0.5).astype(int).flatten()[0]
predicted_label = le.inverse_transform([result_idx])[0]
```

> **How the threshold works:**
> - `prediction_prob = 0.03` → `0.03 > 0.5` is False → index = 0 → **"Attack"**
> - `prediction_prob = 0.91` → `0.91 > 0.5` is True → index = 1 → **"Normal"**

---

### `app.py` — Dashboard (Key Sections)

#### Loading artifacts with caching

```python
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model  = keras.saving.load_model('ids_lstm_model_patched.keras')
    scaler = joblib.load('scaler.pkl')
    le     = joblib.load('label_encoder.pkl')
    return model, scaler, le
```

> **Why `@st.cache_resource`?** Without this, the model would reload from disk
> every time the user clicks a button. Caching loads it once and keeps it in memory.

#### Batch prediction function

```python
def predict_batch(df_raw, model, scaler, le):
    df = df_raw[FEATURE_COLS].copy()        # keep only the 90 feature columns
    df = df.replace('-', 0)                  # clean missing values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    X = df.astype('float32').values          # convert to float array
    X_scaled = scaler.transform(X)           # normalize
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # reshape for LSTM
    probs = model.predict(X_reshaped, verbose=0).flatten()  # get scores
    indices = (probs > 0.5).astype(int)      # apply threshold
    labels = le.inverse_transform(indices)   # convert to words
    return labels, probs
```

> This function handles **many rows at once** (batch processing) instead of one at a time,
> which makes it much faster.

#### Color-coded results table

```python
def color_row(row):
    if not row['Correct']:
        return ['background-color: #fef2f2'] * len(row)  # red = wrong prediction
    if row['Predicted'] == 'Attack':
        return ['background-color: #fff7ed'] * len(row)  # orange = correct attack
    return ['background-color: #f0fdf4'] * len(row)       # green = correct normal
```

> Each row in the results table is colored based on whether the prediction was
> correct and what type it was.

---

## 10. How to Run the Project

### Requirements
- Windows 10 or later
- Python 3.10+ (already set up in `.venv`)
- The project folder with all files intact

### Option A — Double-click (Easiest)

1. Open the `BotNet` folder on the Desktop
2. Double-click **`START APP.bat`**
3. A black window opens — wait for the message:
   ```
   Local URL: http://localhost:8501
   ```
4. Open Chrome and go to: **http://localhost:8501**
5. Keep the black window open while using the app

### Option B — Manual terminal

1. Open File Explorer and navigate to the `BotNet` folder
2. Click the address bar at the top, type `powershell`, press Enter
3. In the terminal, run:
   ```
   .venv\Scripts\streamlit.exe run app.py
   ```
4. Open Chrome and go to: **http://localhost:8501**

### Option C — Test via terminal only (no dashboard)

```
.venv\Scripts\python.exe test_model.py
```

Expected output:
```
Loading model and artifacts...
Model expects 90 features.

Running inference on a test sample...
=====================================
Prediction Result : Attack
Attack Probability: 0.0000
=====================================
```

---

## 11. How to Use the Dashboard

Once the app is open at `http://localhost:8501`:

### Tab 1 — 📊 Dataset Demo
1. Use the **slider** to choose how many rows to test (10–500)
2. Click **▶ Run Predictions**
3. View results:
   - **Accuracy %** — how many predictions were correct
   - **Pie chart** — Attack vs Normal split
   - **Bar chart** — which attack types appeared
   - **Confidence histogram** — how sure the model was across all samples
   - **Color-coded table** — every row with its prediction

**Table colors:**
- 🟢 Green background = correctly predicted Normal
- 🟠 Orange background = correctly predicted Attack
- 🔴 Red background = wrong prediction (misclassified)

### Tab 2 — 📁 Upload CSV
1. Use the **slider** to choose how many rows to read (50–1000)
2. Click **Browse files** and select a CSV file
3. The file must have the same 90 feature columns as `output.csv`
4. Click **▶ Run Predictions**
5. View the pie chart and results table
6. Click **⬇ Download Results CSV** to save predictions

### Tab 3 — 📈 Dataset Overview
- No interaction needed
- Shows the full dataset breakdown with a bar chart
- Useful for explaining the data during a presentation

---

## 12. Understanding the Results

### The two output columns

| Column | What it means |
|--------|--------------|
| `Predicted` | The model's decision: **"Attack"** or **"Normal"** |
| `Confidence` | How certain the model is (0% = very sure it's an Attack, 100% = very sure it's Normal) |

### Why does 0% Confidence mean Attack?

The model's sigmoid output works like this:

```
Raw score →  0.00   0.25   0.50   0.75   1.00
              ↓       ↓      ↓      ↓      ↓
           Attack  Attack  Unsure Normal Normal
```

- **Confidence near 0%** = score near 0.0 = very confident it is an **Attack**
- **Confidence near 100%** = score near 1.0 = very confident it is **Normal**
- **Confidence near 50%** = the model is unsure

### Accuracy

Accuracy tells you what percentage of predictions matched the actual labels in the dataset.

```
Accuracy = (Correct Predictions / Total Predictions) × 100
```

For example: 95 correct out of 100 = **95% accuracy**

---

## 13. Common Questions & Answers

**Q: Why does the model only have 8 LSTM units? Isn't that small?**  
A: Yes, it is intentionally small. For a binary classification task (attack vs. normal) with well-engineered features, a small model is often enough and trains much faster. More units would risk overfitting.

**Q: Why do we need a scaler?**  
A: Different features have wildly different ranges. `flow_duration` might be 0.002, while `orig_bytes` could be 500,000. Without scaling, large numbers would dominate the model's learning. The MinMaxScaler brings everything to the 0–1 range.

**Q: What does `.pkl` mean?**  
A: `.pkl` is a Python **pickle** file. It is a way to save any Python object (like the scaler or label encoder) to disk and reload it later exactly as it was. Joblib is used instead of standard pickle because it is faster for large numerical objects.

**Q: What is the difference between `ids_lstm_model.keras` and `ids_lstm_model_patched.keras`?**  
A: The original model was saved with a newer version of Keras that includes a `quantization_config` field in its Dense layer configuration. The version of Keras installed on this machine does not recognize that field and throws an error. The patched version has that field removed from the config so it loads correctly — the model weights and behavior are identical.

**Q: Why does the app only read 200 rows from a 1.4 GB file?**  
A: Reading 1.4 GB into browser memory would be very slow or cause the app to crash. 200 rows are more than enough to demonstrate the model working. The results are representative of what the full file would produce.

**Q: What does `@st.cache_resource` do?**  
A: It tells Streamlit to load the model only once and keep it in memory. Without it, the 400 MB model would reload from disk every time the user clicks a button, making the app very slow.

**Q: Can this model be used in a real network?**  
A: Not directly — it would need to be integrated with a network monitoring tool (like Zeek or Suricata) that captures live traffic and extracts the same 90 features. This project demonstrates the detection logic; deployment would require additional engineering.

**Q: What is the difference between DoS and DDoS?**  
A: **DoS** (Denial of Service) = one attacker floods one target. **DDoS** (Distributed DoS) = many infected computers (a botnet) all attack one target at the same time. This dataset contains DoS attacks; detecting them early prevents them from becoming DDoS attacks.

---

## Summary

This project demonstrates how **deep learning** can be applied to **cybersecurity**. By training an LSTM model on 3.2 million network flows across 8 attack types, we built a system that can automatically classify network traffic as safe or dangerous — with a visual dashboard anyone can use.

| Component | Technology |
|-----------|-----------|
| Model | LSTM (Keras + TensorFlow) |
| Data processing | Pandas + NumPy + Scikit-learn |
| Dashboard | Streamlit + Plotly |
| Model storage | Keras SavedModel format |
| Artifact storage | Joblib pickle files |
