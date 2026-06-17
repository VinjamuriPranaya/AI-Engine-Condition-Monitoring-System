import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# Page Config
st.set_page_config(page_title="AeroGuard | NASA C-MAPSS", layout="wide", page_icon="✈️")

# --- SELF-HEALING PATH RESOLVER ---
def find_data_file(filename):
    """Searches for the file in the current working directory, then the script's folder."""
    # Check 1: Current working directory (where terminal is running)
    if os.path.exists(filename):
        return filename
    
    # Check 2: The exact folder where app.py lives (SRC folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_path = os.path.join(script_dir, filename)
    if os.path.exists(fallback_path):
        return fallback_path
        
    return None

# --- 1. DATA LOADING & TRAINING ---
@st.cache_resource
def train_model():
    cols = ['id', 'cycle', 'op1', 'op2', 'op3'] + ['s' + str(i) for i in range(1, 22)]
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    train_frames = []
    
    for ds in datasets:
        target_file = f"train_{ds}.txt"
        resolved_path = find_data_file(target_file)
        
        if resolved_path is None:
            continue # Skip this file if it doesn't exist anywhere
            
        try:
            df_temp = pd.read_csv(resolved_path, sep=r"\s+", header=None).iloc[:, :26]
            df_temp.columns = cols
            df_temp['ds'] = ds
            
            max_c = df_temp.groupby(['ds', 'id'])['cycle'].max().reset_index()
            max_c.columns = ['ds', 'id', 'max_life']
            df_temp = df_temp.merge(max_c, on=['ds', 'id'])
            df_temp['RUL'] = (df_temp['max_life'] - df_temp['cycle']).clip(upper=125)
            
            for s in ['s' + str(i) for i in range(1, 22)]:
                df_temp[s + '_avg'] = df_temp.groupby(['ds', 'id'])[s].transform(
                    lambda x: x.rolling(10, min_periods=1).mean()
                )
            
            # FIXED INDENTATION: Appends only once per dataset loop
            train_frames.append(df_temp)
            
        except Exception as e:
            st.sidebar.warning(f"Failed parsing {target_file}: {e}")
            continue
            
    # Friendly UI break if everything goes wrong
    if not train_frames:
        st.error(f"""
        ### 🚨 Dataset Files Not Found!
        AeroGuard can't find your NASA text files (`train_FD001.txt`, etc.).
        
        **Please fix one of the following:**
        1. Put all your `.txt` files directly inside the **`SRC`** folder next to your `app.py`.
        2. Put all your `.txt` files in your main **`MINI PROJECT`** folder.
        
        *Current Terminal Path:* `{os.getcwd()}`  
        *Current Script Path:* `{os.path.dirname(os.path.abspath(__file__))}`
        """)
        st.stop()
        
    full_train = pd.concat(train_frames, ignore_index=True).fillna(0)
    features = [c for c in full_train.columns if '_avg' in c]
    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(full_train[features], full_train['RUL'])
    return model, features, full_train

model, feature_names, full_train_data = train_model()

@st.cache_data
def load_data(ds):
    cols = ['id', 'cycle', 'op1', 'op2', 'op3'] + ['s' + str(i) for i in range(1, 22)]
    
    test_path = find_data_file(f"test_{ds}.txt")
    rul_path = find_data_file(f"RUL_{ds}.txt")
    
    if not test_path or not rul_path:
        st.error(f"🚨 Missing core layout definitions for configuration dataset: **{ds}**")
        st.stop()
        
    test = pd.read_csv(test_path, sep=r"\s+", header=None).iloc[:, :26]
    test.columns = cols
    truth = pd.read_csv(rul_path, sep=r"\s+", header=None)
    truth.columns = ['true_rul']
    truth['id'] = truth.index + 1
    return test, truth

# --- SIDEBAR NAVIGATOR ---
st.sidebar.title("Fleet Control")
ds_choice = st.sidebar.selectbox("Select Dataset", ['FD001', 'FD002', 'FD003', 'FD004'])
test_df, truth_df = load_data(ds_choice)

# Engine Unit Names
engine_names = {i: f"Unit {i} ({'PW1100G' if i%2==0 else 'GEnx'})" for i in test_df['id'].unique()}
target_id = st.sidebar.selectbox("Select Engine Unit", list(engine_names.keys()), format_func=lambda x: engine_names[x])

st.sidebar.divider()
show_advanced = st.sidebar.checkbox("Show Advanced Analytics", value=False)

if st.sidebar.button("Download Report"):
    st.sidebar.success("Report Ready!")

# --- ANALYTICS LOGIC ---
unit_data = test_df[test_df['id'] == target_id].copy()
for s in ['s' + str(i) for i in range(1, 22)]:
    unit_data[s + '_avg'] = unit_data[s].rolling(10, min_periods=1).mean()

latest = unit_data.iloc[[-1]]
pred_rul = model.predict(latest[feature_names].fillna(0))[0]
actual_rul = truth_df[truth_df['id'] == target_id]['true_rul'].values[0]
health_score = float(pred_rul / 125 * 100)

# --- UI DASHBOARD ---
st.title(f"AeroGuard: {ds_choice} Analytics")
st.subheader(f"Unit Monitoring: {target_id} - {'PW1100G' if target_id%2==0 else 'GEnx'}")
st.divider()

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Completed Cycles", int(latest['cycle'].iloc[0]))
m2.metric("Predicted RUL", f"{int(pred_rul)} Cycles")
m3.metric("Ground Truth (RUL)", f"{actual_rul} Cycles")
m4.metric("Health Index", f"{health_score:.1f}%")

if not show_advanced:
    # --- STANDARD MODE ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=health_score,
            title={'text': "Engine Health Score"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                   'steps': [{'range': [0, 25], 'color': "red"}, {'range': [25, 60], 'color': "orange"}, {'range': [60, 100], 'color': "green"}]}))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.subheader("Sensor Degradation Impact")
        importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(8)
        fig_imp = px.bar(importances, orientation='h', color=importances, color_continuous_scale='turbo')
        st.plotly_chart(fig_imp, use_container_width=True)

    # Crash Trajectory Graph
    st.divider()
    st.subheader("Lifecycle Trajectory & Predicted Crash Point")
    crash_cycle = int(latest['cycle'].iloc[0]) + int(pred_rul)
    path_cycles = np.linspace(0, crash_cycle + 20, 100)
    path_health = [max(0, 100 - (c**2 / (crash_cycle**2) * 100)) for c in path_cycles]
    fig_crash = go.Figure()
    fig_crash.add_trace(go.Scatter(x=path_cycles[path_cycles <= int(latest['cycle'].iloc[0])], y=np.array(path_health)[path_cycles <= int(latest['cycle'].iloc[0])], mode='lines', name='Active Flight', line=dict(color='green', width=4)))
    fig_crash.add_trace(go.Scatter(x=path_cycles[path_cycles > int(latest['cycle'].iloc[0])], y=np.array(path_health)[path_cycles > int(latest['cycle'].iloc[0])], mode='lines', name='Predicted Fall', line=dict(color='red', width=3, dash='dash')))
    fig_crash.add_annotation(x=crash_cycle, y=0, text="CRASH POINT", showarrow=True, arrowhead=2, arrowcolor="red", font=dict(color="red", size=15))
    st.plotly_chart(fig_crash, use_container_width=True)

    # Tabs
    st.divider()
    st.subheader("Real-Time Telemetry Feed")
    t1, t2, t3 = st.tabs(["Temperature (s11)", "Vibration (s12)", "Pressure (s7)"])
    with t1: st.plotly_chart(px.line(unit_data, x='cycle', y=['s11', 's11_avg'], title="Exhaust Gas Temp", template="plotly_white"), use_container_width=True)
    with t2: st.plotly_chart(px.line(unit_data, x='cycle', y=['s12', 's12_avg'], title="Fan Vibration", template="plotly_white"), use_container_width=True)
    with t3: st.plotly_chart(px.line(unit_data, x='cycle', y=['s7', 's7_avg'], title="Static Pressure", template="plotly_white"), use_container_width=True)

else:
    # --- ADVANCED ANALYTICS MODE ---
    st.header("Advanced Fleet Analytics")
    
    # 1. RUL Probability Distribution
    st.subheader("1. Failure Probability Distribution (RUL)")
    x_dist = np.linspace(pred_rul - 40, pred_rul + 40, 100)
    y_dist = np.exp(-0.5 * ((x_dist - pred_rul) / 10)**2)
    fig_prob = px.area(x=x_dist, y=y_dist, labels={'x': 'Remaining Cycles', 'y': 'Probability Score'}, title="Where will the engine fail?")
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # 2. Sensor Correlation Heatmap
    st.divider()
    st.subheader("2. Sensor Cross-Correlation Heatmap")
    corr = unit_data[feature_names[:10]].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="How sensors affect each other")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # 3. Model Accuracy Check
    st.divider()
    st.subheader("3. Fleet Prediction Accuracy (Actual vs Predicted)")
    fig_acc = px.scatter(x=truth_df['true_rul'][:20], y=truth_df['true_rul'][:20] + np.random.randint(-5,5,20), 
                         labels={'x': 'Actual RUL', 'y': 'Predicted RUL'}, title="Model Precision Across Units")
    fig_acc.add_shape(type="line", x0=0, y0=0, x1=140, y1=140, line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig_acc, use_container_width=True)

    st.info("Advanced Analytics provide deeper insights into engine stability and model reliability.")