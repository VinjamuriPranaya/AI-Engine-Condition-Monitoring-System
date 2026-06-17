# AI-Engine-Condition-Monitoring-System
Developed a predictive maintenance system utilizing dedicated training and testing datasets to evaluate real-time aircraft engine health, generating interactive visual graphs to track degradation trends and metrics.
# AeroGuard: AI-Based Aircraft Engine Health Monitoring System

AeroGuard is an interactive Streamlit analytics dashboard that monitors aircraft engine degradation and predicts the **Remaining Useful Life (RUL)** of turbofan engines. Built using machine learning models trained on the comprehensive **NASA C-MAPSS dataset**, this tool provides real-time fleet analytics, sensor breakdown impacts, and failure probability tracking for aviation predictive maintenance.

🚀 **[Live Dashboard Link](PASTE_YOUR_STREAMLIT_CLOUD_OR_DEPLOYMENT_URL_HERE)**

---

## 📸 Dashboard Preview

### 1. Core Fleet Dashboard (Standard Mode)
*Real-time health index indicators, sensor impact metrics, and engine life cycle degradation trajectories.*
![Standard Dashboard View](https://github.com/user-attachments/assets/37ebbe77-b0b6-45d4-8171-829c54981f8a)
### 2. Advanced Fleet Analytics Mode
*Probability curves for exact failure distribution points and sensor cross-correlation matrices.*
![Advanced Analytics View](https://github.com/user-attachments/assets/1c5f4179-5d74-4356-886d-68a0a7898bf1).

---

## 🛠️ System Architecture & Features

### 🧑‍💻 Machine Learning Engine
- **Algorithm:** Random Forest Regressor (`n_estimators=100`, `max_depth=12`).
- **Feature Engineering:** Features use a 10-cycle rolling average computation over raw telemetry signals (`s1` to `s21`) to capture temporal engine wear patterns while flattening background noise.
- **RUL Clipping:** Targeted piecewise Remaining Useful Life capped at an upper bound of 125 cycles to accurately model non-linear engine degradation closer to the wear point.

### 📊 Interactive Views
1. **Fleet Control Sidebar:** Dynamically switch across flight condition datasets (`FD001` - `FD004`) and isolate individual engine units (e.g., *Pratt & Whitney PW1100G* or *GE GEnx* simulations).
2. **Standard Mode Analytics:**
   - **Engine Health Indicator:** Live gauge mapping current structural threshold health.
   - **Sensor Degradation Graph:** Displays the top 8 sensors accelerating lifecycle failures.
   - **Lifecycle Trajectory Mapping:** Displays an interactive dynamic curve charting active flights against predicted structural fall-off paths.
3. **Advanced Mode Analytics:** Includes failure probability distribution charts and a dynamic cross-correlation matrix mapping sensor dependencies.

---

## 📁 Repository Structure

```text
├── app.py                  # Main Streamlit web application script
├── requirements.txt        # System application dependencies
├── README.md               # Professional project portfolio documentation
├── train_FD001.txt         # Sub-dataset telemetry logs (FD001 to FD004)
├── test_FD001.txt          # Engine test operational cycle data 
└── RUL_FD001.txt           # Verified ground-truth metrics
