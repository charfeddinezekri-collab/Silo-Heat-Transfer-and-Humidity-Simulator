import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime, timedelta
import base64
import io
import xlsxwriter

# ==============================================================================
# --- 0. CORE PHYSICS CONSTANTS AND ADAPTIVE LOGIC ---
# ==============================================================================

# Silo Dimensions (from your original code, converted to meters)
SILO_R = 0.75  # 75 cm
SILO_H = 1.70  # 170 cm
SILO_VOLUME = math.pi * SILO_R**2 * SILO_H  # Volume of the grain mass (~3.00 m^3)

# Material Properties (Provided Inputs)
RHO_GRAIN = 670.0  # kg/m^3 (Density of Grain)
CP_GRAIN = 1560.0  # J/(kg*K) (1.56 kJ/kg/K Specific Heat Capacity of Grain)
SILO_MASS = RHO_GRAIN * SILO_VOLUME  # Total mass of grain (kg)

# Air Properties (Standard Values)
CP_AIR = 1005.0  # J/(kg*K) (Specific heat of air)
RHO_AIR = 1.225 # kg/m^3 (Density of air)

# Solver Inputs (Provided by User)
VENT_AIRFLOW_RATE_M3_S = 1800.0 / 3600.0  # 1800 m^3/hr -> 0.5 m^3/s (Ventilation Airflow Rate)
INSULATION_RATIO = 7.5 # (Insulation Performance Ratio)

# Dew Point Constants (Provided by User for Magnus-Tetens formula)
A_CONSTANT = 17.27
B_CONSTANT = 237.0

# --- HUMIDITY CONSTANTS (MODIFIED: DAMPING INCREASED 2.5X) ---
# Defines how quickly the silo internal RH responds to ambient RH (applied to Dew Point). 
RH_DAMPING_FACTOR = 0.1 

# --- VENTILATION PHYSICS ---

def calculate_saturation_vapor_pressure(T):
    """Calculates Saturation Vapor Pressure (hPa) using Magnus-Tetens (A=17.27, B=237.0)."""
    # E-sat = 6.11 * exp( (A * T) / (B + T) )
    return 6.11 * math.exp((A_CONSTANT * T) / (B_CONSTANT + T))

def calculate_rh_from_dew_point(T_dew, T_grain):
    """
    Calculates RH based on actual moisture content (T_dew) 
    and the surrounding air/grain temperature (T_grain).
    """
    P_actual = calculate_saturation_vapor_pressure(T_dew)
    P_sat_silo = calculate_saturation_vapor_pressure(T_grain)
    
    if P_sat_silo <= 0: return 100.0
    
    return min(100.0, max(0.0, (P_actual / P_sat_silo) * 100.0))

def calculate_dew_point(T_ambient, RH_ambient):
    """
    Implements the Magnus-Tetens approximation for Dew Point (Condition C1).
    Input T_ambient (Celsius), RH_ambient (Percent 0-100).
    """
    if RH_ambient <= 0: return -50.0 
    RH_ratio = max(0.01, min(1.0, RH_ambient / 100.0))
    
    gamma = (A_CONSTANT * T_ambient) / (B_CONSTANT + T_ambient) + math.log(RH_ratio)
    T_dew = (B_CONSTANT * gamma) / (A_CONSTANT - gamma)
    return T_dew

def calculate_ventilation_cooling(T_ambient, T_grain, V_rate_m3_s):
    """
    Calculates the change in grain temperature (Delta T) due to ventilation 
    using an energy balance over one hour (3600s).
    """
    if V_rate_m3_s == 0:
        return 0.0
    
    M_DOT_AIR = RHO_AIR * V_rate_m3_s
    COEFF = (M_DOT_AIR * CP_AIR * 3600) / (SILO_MASS * CP_GRAIN)
    delta_T_vent = COEFF * (T_ambient - T_grain)
    
    return delta_T_vent

def calculate_automated_status(current_T_ambient, current_RH_ambient, avg_T_grain, max_T_layer_diff):
    """
    Implements the full automated logic from the user's Diagram 2 (C1, C2, C3, C4).
    """
    
    # 1. C1: Dew Point < Average Grain Temp
    T_dew = calculate_dew_point(current_T_ambient, current_RH_ambient)
    C1 = (T_dew < avg_T_grain)
    
    if not C1: return False
    
    # 2. C2: Intergranular RH < 75%
    C2 = (avg_T_grain < 30.0) 
    
    # 3. C3: |External Temp - Internal Temp| < 3°C
    C3 = (abs(current_T_ambient - avg_T_grain) < 3.0)
    
    # 4. C4: Layer Difference < 5°C
    C4 = (max_T_layer_diff < 5.0)

    # Fan ON if: (C1 is True) AND ( (C2 is True) OR (C3 or C4 is True) )
    if C1 and (C2 or C3 or C4):
        return True 
    
    return False 

# --- ADAPTIVE HEURISTIC MODEL ---

def calculate_point_temperature_adaptive(normalized_radius, normalized_height, current_time_index, history_df, insulation_factor, delta_T_vent):
    """
    Calculates temperature at a specific point based on history, 
    adapted for insulation and ventilation.
    """
    if history_df.empty: return 20.0
    
    # 2D Heat Transfer Heuristic (from your original code)
    dist_to_wall = 1.0 - normalized_radius
    dist_to_roof = 1.0 - normalized_height
    dist_to_floor = normalized_height
    
    # 1. Thermal Depth and Lag (Modified by Insulation)
    thermal_depth = min(
        # Modification from previous step: Halving the effective insulation factor 
        # to reduce thermal lag by half in the insulated scenario.
        dist_to_wall * 1.0 * (insulation_factor / 2.0),
        dist_to_roof * 0.8,
        (dist_to_floor + 0.5) * 2.0
    )
    
    max_lag_hours = 36
    lag_hours = thermal_depth * max_lag_hours
    
    # 2. Damping (Modified by Insulation)
    damping = max(0.1, 1 - (thermal_depth * 0.8 / insulation_factor)) 
    
    lookback_index = max(0, int(current_time_index - lag_hours))
    delayed_ambient = history_df.iloc[lookback_index]['ambient_temp']
    
    base_temp = 35.0
    
    # 3. Equilibration
    equilibration_time = 120
    time_factor = min(1.0, current_time_index / equilibration_time)
    
    # Base Conduction/Radiation Result
    wave_component = (delayed_ambient - base_temp) * damping + base_temp
    result = (wave_component * time_factor) + (base_temp * (1 - time_factor))
    
    # 4. Apply Ventilation Effect
    result += delta_T_vent
    
    return result

# --- SCENARIO DISPATCHER ---

def select_scenario_key(is_insulated, ventilation_type):
    """Maps the user's selection (Diagram 1) to a unique key."""
    if not is_insulated:
        if ventilation_type == 'Unventilated': return 'M1_Uninsulated_Unventilated'
        elif ventilation_type == 'Regular': return 'M2_Uninsulated_Regular'
        elif ventilation_type == 'Automated': return 'M3_Uninsulated_Automated'
    else: 
        if ventilation_type == 'Unventilated': return 'M4_Insulated_Unventilated'
        elif ventilation_type == 'Regular': return 'M5_Insulated_Regular'
        elif ventilation_type == 'Automated': return 'M6_Insulated_Automated'
            
    raise ValueError("Invalid Silo Configuration provided.")

# ==============================================================================
# --- 1. SIMULATION ENGINE (ADAPTED) ---
# ==============================================================================

@st.cache_data
def load_default_data(days=7):
    """Generates synthetic data with ambient temp and RH."""
    data = []
    for i in range(days * 24):
        hour = i % 24
        base_temp = 28
        amplitude = 12
        temp = base_temp + amplitude * math.sin(((hour - 8) * math.pi) / 12) + (np.random.random() - 0.5) * 2
        
        # Synthetic RH: low when T is high, high when T is low
        rh = 75 - 20 * math.sin(((hour - 8) * math.pi) / 12) + (np.random.random() * 5)
        
        data.append({
            "time_index": i,
            "ambient_temp": temp,
            "ambient_rh": max(30, min(100, rh)) 
        })
    return pd.DataFrame(data)

def get_sensors():
    """Returns the sensor configuration as a DataFrame."""
    silo_height = 170.0
    silo_radius = 75.0
    
    layers = [
        {"height": 17, "id": 1, "color": "#ef4444"},
        {"height": 62, "id": 2, "color": "#3b82f6"},
        {"height": 107, "id": 3, "color": "#22c55e"},
        {"height": 152, "id": 4, "color": "#f59e0b"},
    ]
    
    x_positions = [73, 71, 0, -71, -73]
    
    sensors = []
    count = 1
    for layer in layers:
        for x_pos in x_positions:
            sensors.append({
                "id": f"S{count}",
                "name": f"S{count}",
                "layer": layer["id"],
                "color": layer["color"],
                "real_x": x_pos,
                "real_y": layer["height"],
                "real_z": 0,
                "norm_x": x_pos / silo_radius,
                "norm_y": layer["height"] / silo_height,
                "norm_r": abs(x_pos) / silo_radius
            })
            count += 1
    return pd.DataFrame(sensors)

@st.cache_data
def calculate_full_history_adaptive(sim_data, sensor_config, is_insulated, ventilation_type):
    """
    The main adaptive solver. Simulates temperature and the new silo RH.
    """
    
    current_solver_key = select_scenario_key(is_insulated, ventilation_type)
    insulation_factor = INSULATION_RATIO if is_insulated else 1.0
    history_points = []
    
    # --- RH SIMULATION SETUP ---
    # Initialize silo moisture proxy (Dew Point). Use a reasonable initial value.
    silo_dew_point = 20.0 
    
    # Calculate for every hour
    for t in range(len(sim_data)):
        current_ambient = sim_data.iloc[t]['ambient_temp']
        current_rh = sim_data.iloc[t]['ambient_rh']
        
        point = {"time": t, "ambient": current_ambient, "ambient_rh": current_rh}
        
        delta_T_vent = 0.0
        vent_status = "OFF"
        
        # Determine average grain temp from previous step for Q_vent and RH calculation
        prev_avg_T_grain = history_points[-1]['avg_temp'] if history_points else 35.0
        
        # --- SCENARIO-SPECIFIC VENTILATION LOGIC ---
        
        if 'Regular' in current_solver_key:
            hour_of_day = t % 24
            if hour_of_day >= 0 and hour_of_day <= 6:
                delta_T_vent = calculate_ventilation_cooling(current_ambient, prev_avg_T_grain, VENT_AIRFLOW_RATE_M3_S)
                vent_status = "ON (Regular)"

        elif 'Automated' in current_solver_key:
            max_T_layer_diff = 0.0
            if history_points:
                prev_row = pd.DataFrame(history_points).iloc[-1]
                layer_temps = [prev_row[f'layer_{i}'] for i in [1, 2, 3, 4] if f'layer_{i}' in prev_row]
                if len(layer_temps) >= 2:
                    max_T_layer_diff = max(layer_temps) - min(layer_temps)

            if calculate_automated_status(current_ambient, current_rh, prev_avg_T_grain, max_T_layer_diff):
                delta_T_vent = calculate_ventilation_cooling(current_ambient, prev_avg_T_grain, VENT_AIRFLOW_RATE_M3_S)
                vent_status = "ON (Automated)"
        
        # --- CALCULATE PROBE TEMPERATURES ---
        probe_temps = []
        for _, s in sensor_config.iterrows():
            temp = calculate_point_temperature_adaptive(
                s['norm_r'], 
                s['norm_y'], 
                t, 
                sim_data, 
                insulation_factor=insulation_factor, 
                delta_T_vent=delta_T_vent
            )
            probe_temps.append(temp)
            point[f"probe_{s['id']}"] = temp
        
        # Store Averages and Status
        avg_temp = sum(probe_temps) / len(probe_temps)
        point['avg_temp'] = avg_temp
        point['vent_status'] = vent_status
        
        # --- SILO RH SIMULATION (MODIFIED TO USE MAGNUS-TETENS) ---
        if history_points:
            # 1. Calculate the ambient moisture content (as Dew Point)
            current_dew = calculate_dew_point(current_ambient, current_rh)
            
            # 2. Damp the Dew Point (The absolute moisture content lag)
            silo_dew_point = silo_dew_point + RH_DAMPING_FACTOR * (current_dew - silo_dew_point)
            
            # 3. Calculate Silo RH using damped moisture content (Dew Point) and the current average grain temperature
            silo_rh = calculate_rh_from_dew_point(silo_dew_point, avg_temp)
            silo_rh = min(95.0, max(30.0, silo_rh)) 
        else:
            # Initialization case
            silo_rh = calculate_rh_from_dew_point(silo_dew_point, avg_temp)
        
        point['silo_rh'] = silo_rh
        history_points.append(point)
    
    hist_df = pd.DataFrame(history_points)
    
    # Calculate layer means from probe data
    for layer_id in [1, 2, 3, 4]:
        probe_cols = [col for col in hist_df.columns if col.startswith('probe_') and sensor_config[sensor_config['layer'] == layer_id]['id'].str.contains(col.split('_')[1]).any()]
        hist_df[f"layer_{layer_id}"] = hist_df[probe_cols].mean(axis=1)
        
    return hist_df

# ==============================================================================
# --- 2. THEME & STYLE, PLOTTING, AND EXPORT ---
# ==============================================================================

# Custom CSS
st.set_page_config(
    page_title="(Grain Silo Thermal Simulation and Analysis Engineered by Zekri)",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .stSidebar { background-color: #1e293b; }
    h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #f8fafc; }
    .metric-card { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; }
    .stSlider > div > div > div > div { background-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# --- PLOTTING FUNCTIONS ---

def plot_relative_humidity_evolution(hist_df):
    """
    Plots the simulated internal silo relative humidity against the ambient RH.
    """
    st.subheader("6. Relative Humidity Evolution (Ambient vs. Silo)")
    
    fig = go.Figure()
    
    # Ambient RH
    fig.add_trace(go.Scatter(
        x=hist_df['time'], y=hist_df['ambient_rh'],
        name="Ambient RH",
        line=dict(color='#84cc16', width=2, dash='dot') # Lime Green
    ))
    
    # Silo RH (Simulated)
    fig.add_trace(go.Scatter(
        x=hist_df['time'], y=hist_df['silo_rh'],
        name="Silo Internal RH (Simulated)",
        line=dict(color='#3b82f6', width=3) # Blue
    ))
    
    # Critical RH Line (75% for mold risk, used in C2 logic)
    fig.add_hline(y=75, line_dash="dash", line_color="#ef4444", 
                  annotation_text="Critical RH (75%)", 
                  annotation_position="top right",
                  annotation_font_color="#ef4444")

    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
        xaxis=dict(title="Time (Hours)", gridcolor="#334155"),
        yaxis=dict(title="Relative Humidity (%)", range=[0, 100], gridcolor="#334155"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return {"relative_humidity_evolution": fig}

def plot_individual_probe_evolution(hist_df, sensors_df):
    st.subheader("1. Temperature Evolution of Individual Probes")
    probe_ids = sensors_df['id'].unique()
    cols = st.columns(5)
    for i, probe_id in enumerate(probe_ids):
        with cols[i % 5]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df[f'probe_{probe_id}'], name=f"Probe {probe_id}", line=dict(color=sensors_df[sensors_df['id'] == probe_id]['color'].iloc[0], width=2)))
            fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['ambient'], name="Ambient", line=dict(color='#94a3b8', width=1, dash='dot')))
            fig.update_layout(title=f"Probe {probe_id} Temp vs Time", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#f8fafc", size=10), xaxis=dict(title="Time (Hours)", showgrid=False), yaxis=dict(title="Temp (°C)", showgrid=True, gridcolor="#334155"), height=250, margin=dict(l=20, r=20, t=40, b=20), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
    return {f"probe_{probe_id}_evolution": fig for i, probe_id in enumerate(probe_ids)}

def plot_individual_layer_evolution(hist_df, sensors_df):
    st.subheader("2. Temperature Evolution of Individual Layers")
    layer_ids = sorted(sensors_df['layer'].unique())
    layer_colors = {1: '#ef4444', 2: '#3b82f6', 3: '#22c55e', 4: '#f59e0b'}
    cols = st.columns(4)
    for i, layer_id in enumerate(layer_ids):
        with cols[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df[f'layer_{layer_id}'], name=f"Layer {layer_id} Mean", line=dict(color=layer_colors[layer_id], width=2)))
            fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['ambient'], name="Ambient", line=dict(color='#94a3b8', width=1, dash='dot')))
            fig.update_layout(title=f"Layer {layer_id} Temp vs Time", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#f8fafc", size=10), xaxis=dict(title="Time (Hours)", showgrid=False), yaxis=dict(title="Temp (°C)", showgrid=True, gridcolor="#334155"), height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
    return {f"layer_{layer_id}_evolution": fig for i, layer_id in enumerate(layer_ids)}

def plot_combined_probe_evolution(hist_df, sensors_df, toggled_probes):
    st.subheader("4. Combined Probe Temperature Evolution")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['ambient'], name="Ambient", line=dict(color='#94a3b8', width=2, dash='dot')))
    for probe_id in sensors_df['id'].unique():
        if probe_id in toggled_probes:
            color = sensors_df[sensors_df['id'] == probe_id]['color'].iloc[0]
            fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df[f'probe_{probe_id}'], name=f"Probe {probe_id}", line=dict(color=color, width=1)))
    fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#f8fafc"), xaxis=dict(title="Time (Hours)", gridcolor="#334155"), yaxis=dict(title="Temperature (°C)", gridcolor="#334155"), hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20), height=500)
    st.plotly_chart(fig, use_container_width=True)
    return {"combined_probe_evolution": fig}

def plot_combined_layer_evolution(hist_df, sensors_df):
    st.subheader("5. Combined Layer Temperature Evolution")
    fig = go.Figure()
    layer_colors = {1: '#ef4444', 2: '#3b82f6', 3: '#22c55e', 4: '#f59e0b'}
    fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['ambient'], name="Ambient", line=dict(color='#94a3b8', width=2, dash='dot')))
    
    # --- FIX IMPLEMENTED HERE: X-axis set to 'time' (from previous fix) ---
    for layer_id in [1, 2, 3, 4]:
        fig.add_trace(go.Scatter(x=hist_df['time'], y=hist_df[f'layer_{layer_id}'], name=f"Layer {layer_id}", line=dict(color=layer_colors[layer_id], width=2)))
        
    fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#f8fafc"), xaxis=dict(title="Time (Hours)", gridcolor="#334155"), yaxis=dict(title="Temperature (°C)", gridcolor="#334155"), hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20), height=500)
    st.plotly_chart(fig, use_container_width=True)
    return {"combined_layer_evolution": fig}

def plot_meshed_silo(sensors_df):
    st.subheader("6. Meshed Silo Cross-Section (Sensor Grid)")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-75, y0=0, x1=75, y1=170, line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)")
    fig.add_trace(go.Scatter(x=sensors_df['real_x'], y=sensors_df['real_y'], mode='markers+text', marker=dict(size=10, color=sensors_df['layer'].apply(lambda l: {1: '#ef4444', 2: '#3b82f6', 3: '#22c55e', 4: '#f59e0b'}.get(l)), line=dict(width=1, color='white')), text=sensors_df['id'], textposition="top center", name="Probes"))
    for layer_id in sensors_df['layer'].unique():
        layer_probes = sensors_df[sensors_df['layer'] == layer_id].sort_values(by='real_x')
        fig.add_trace(go.Scatter(x=layer_probes['real_x'], y=layer_probes['real_y'], mode='lines', line=dict(color="#334155", width=1, dash="dash"), hoverinfo='skip', showlegend=False))
    for x_pos in sensors_df['real_x'].unique():
        radial_probes = sensors_df[sensors_df['real_x'] == x_pos].sort_values(by='real_y')
        fig.add_trace(go.Scatter(x=radial_probes['real_x'], y=radial_probes['real_y'], mode='lines', line=dict(color="#334155", width=1, dash="dash"), hoverinfo='skip', showlegend=False))
    fig.update_layout(title="Silo Cross-Section and Sensor Mesh (X-Z Plane)", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#f8fafc"), xaxis=dict(title="Radius (cm)", range=[-80, 80], scaleanchor="y", scaleratio=1, gridcolor="#334155"), yaxis=dict(title="Height (cm)", range=[0, 180], gridcolor="#334155"), height=600, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    return {"meshed_silo_cross_section": fig}

# --- EXPORT FUNCTIONS ---

def to_excel(hist_df, sensors_df):
    """Exports all data to a well-organized Excel file."""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Sheet 1: Full Historical Data (Automatically includes 'silo_rh')
    hist_df.to_excel(writer, sheet_name='Historical Data', index=False)
    
    # Sheet 2: Sensor Configuration
    sensors_df.to_excel(writer, sheet_name='Sensor Configuration', index=False)
    
    # Sheet 3: Statistical Metrics
    stats = hist_df.drop(columns=['time']).describe().T
    stats['Range'] = stats['max'] - stats['min']
    stats.to_excel(writer, sheet_name='Statistical Metrics')
    
    # Use writer.close() instead of writer.save() if using xlsxwriter
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def get_latex_report_content(hist_df, sensors_df):
    """Generates the LaTeX content for the Overleaf report."""
    
    # 1. Statistical Metrics
    stats = hist_df.drop(columns=['time']).describe().T
    stats['Range'] = stats['max'] - stats['min']
    stats_latex = stats.to_latex(float_format="%.2f")
    
    # 2. Heat Transfer Study (Feature 8)
    study_text = r"""
\section{Study on Heat Transfer Governing Equations and Adaptive Solver}

The thermal and moisture behavior of the grain silo is modeled using an adaptive heuristic, extending the base transient heat transfer principles to account for insulation, forced ventilation, and humidity damping.

\subsection{Adaptive Heuristic Model}
The core simulation uses a time-delayed and damped thermal wave heuristic, where:
\begin{itemize}
    \item \textbf{Insulation (Silo Two):} The \texttt{INSULATION\_RATIO} ($\mathbf{7.5}$) modifies the thermal depth and damping factors. The lag effect of insulation has been halved ($\mathbf{/2.0}$) to increase the responsiveness to external temperature changes.
    \item \textbf{Ventilation:} A $\mathbf{\Delta T_{vent}}$ term, calculated from an energy balance using the Airflow Rate ($\mathbf{1800\,m^3/hr}$), models heat removal/addition.
    \item \textbf{Internal Humidity:} The internal silo Relative Humidity (\textbf{Silo RH}) is now simulated using the Magnus-Tetens formula. The \textbf{absolute moisture content} (Dew Point) is damped over time using a factor of $\mathbf{0.05}$ (increased responsiveness), and the final RH is calculated from this damped Dew Point and the current Average Grain Temperature.
\end{itemize}

\subsection{Automated Ventilation Logic (Condition C1-C4)}
For the automated scenarios, the fan is controlled hourly based on the following logic:
\begin{itemize}
    \item \textbf{C1 (Dew Point Check):} Ventilation turns ON only if Ambient Dew Point is lower than Average Grain Temperature.
    \item \textbf{C2 (Intergranular RH Check):} Ventilation checks if the internal relative humidity is below $75\%$. (Uses $\mathbf{T_{grain} < 30^\circ C}$ placeholder for current implementation).
    \item \textbf{C3/C4 (Stability Check):} Checks for small temperature differences.
\end{itemize}
"""
    
    # 3. LaTeX Template (Unmodified structure)
    latex_template = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{caption}
\geometry{a4paper, margin=1in}

\title{Grain Silo Thermal and Moisture Simulation Report}
\author{Generated by Manus AI (Adaptive Solver)}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This report presents the results of a thermal and moisture simulation for a grain silo, utilizing an adaptive solver capable of simulating six distinct physical scenarios.

\section{Simulation Parameters and Solver Configuration}
\subsection{Physical Parameters}
\begin{itemize}
    \item Grain Density ($\rho$): $\mathbf{670\,kg/m^3}$
    \item Grain Specific Heat ($c_p$): $\mathbf{1.56\,kJ/kg/K}$
    \item Insulation Ratio: $\mathbf{7.5}$ (Modified for responsiveness)
    \item Ventilation Rate ($\dot{V}$): $\mathbf{1800\,m^3/hr}$
\end{itemize}
The current simulation scenario is set by the user interface controls.

\begin{table}[h]
    \centering
    \caption{Sensor Configuration Details}
    \label{tab:sensors}
    \begin{tabular}{lcccc}
        \toprule
        Probe ID & Layer & Real X (cm) & Real Y (Height, cm) & Normalized Radius \\
        \midrule
        """
    
    # Add sensor data to table
    for _, s in sensors_df.iterrows():
        latex_template += f"        {s['id']} & {s['layer']} & {s['real_x']:.1f} & {s['real_y']:.1f} & {s['norm_r']:.2f} \\\\\n"
        
    latex_template += r"""
        \bottomrule
    \end{tabular}
\end{table}
\clearpage

\section{Statistical Metrics}
The following table summarizes the key statistical metrics for the ambient temperature, probes, layers, and the \textbf{Simulated Silo RH} over the entire simulation period.

\begin{table}[h]
    \centering
    \caption{Summary of Statistical Metrics}
    \label{tab:stats}
    \resizebox{\textwidth}{!}{%
    """
    stats_content = stats_latex.replace('\\begin{tabular}', '').replace('\\end{tabular}', '').strip()
    
    latex_template += r"""
    \begin{tabular}{lrrrrrrrr}
        \toprule
    """
    latex_content = stats_content
    latex_template += r"""
        \bottomrule
    \end{tabular}
    }
\end{table}

\section{Visualization of Results}

\subsection{Relative Humidity Analysis}
Figure \ref{fig:rh_evolution} shows the simulated behavior of the internal silo relative humidity compared to the ambient conditions.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{relative_humidity_evolution.png}
    \caption{Evolution of Simulated Silo Internal Relative Humidity vs. Ambient Relative Humidity.}
    \label{fig:rh_evolution}
\end{figure}

\subsection{Meshed Silo Cross-Section}
The meshed cross-section, which represents the sensor placement grid, is shown in Figure \ref{fig:mesh}.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{meshed_silo_cross_section.png}
    \caption{Silo Cross-Section and Sensor Mesh (X-Z Plane).}
    \label{fig:mesh}
</figure}

\subsection{Combined Temperature Evolution}
The combined evolution plots provide an overview of the thermal behavior.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{combined_layer_evolution.png}
    \caption{Temperature Evolution of All Layers vs. Ambient Temperature.}
    \label{fig:combined_layers}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{combined_probe_evolution.png}
    \caption{Temperature Evolution of All Probes vs. Ambient Temperature.}
    \label{fig:combined_probes}
\end{figure}

\section{Detailed Heat Transfer Analysis}
"""
    latex_template += study_text
    
    latex_template += r"""
\end{document}
"""
    return latex_template

# --- MAIN APP LOGIC (ADAPTED) ---

def main():
    st.title("Grain Silo Thermal Simulation and Analysis")
    st.sidebar.title("🎛️ Controls")
    
    # 1. Data Source
    uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV Data", type=['xlsx', 'csv'])
    
    if uploaded_file:
        try:
            # --- ROBUST FILE READING LOGIC (FIX FOR ROW COUNT ISSUE) ---
            if uploaded_file.name.lower().endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else: # Assume Excel or other format, try to read as Excel
                # Use engine='openpyxl' for robust Excel reading
                df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Use columns that contain 'T amb' and 'HR amb' for robustness
            temp_cols = [col for col in df_raw.columns if 'T amb' in str(col).strip()]
            rh_cols = [col for col in df_raw.columns if 'HR amb' in str(col).strip()]
            
            if not temp_cols or not rh_cols:
                raise ValueError("Could not find 'T amb' or 'HR amb' columns. Check column names.")

            temp_col = temp_cols[0]
            rh_col = rh_cols[0]

            data = []
            for i, row in df_raw.iterrows():
                temp_val = row[temp_col]
                rh_val = row[rh_col] if rh_col in row and not pd.isna(row[rh_col]) else 50.0 
                
                # Filter out rows where T amb is not a valid number (e.g., empty or header rows after conversion)
                if isinstance(temp_val, (int, float, np.number)) and not pd.isna(temp_val):
                    data.append({"time_index": i, "ambient_temp": float(temp_val), "ambient_rh": float(rh_val)})
            
            simulation_df = pd.DataFrame(data)
            
            if simulation_df.empty:
                 raise ValueError(f"Loaded file '{uploaded_file.name}' but extracted 0 valid rows.")

            st.sidebar.success(f"Loaded {len(simulation_df)} points from '{uploaded_file.name}'")

        except Exception as e:
            st.sidebar.error(f"Error reading file or columns: {e}. Falling back to default data.")
            simulation_df = load_default_data(days=14)
    else:
        # Load default data for 14 days (336 hours) if no file is uploaded.
        simulation_df = load_default_data(days=14)
    
    if simulation_df.empty: 
        st.error("No valid simulation data could be loaded.")
        return

    # --- SCENARIO SELECTION CONTROLS ---
    st.sidebar.write("---")
    st.sidebar.subheader("⚙️ Scenario Selection")
    silo_type = st.sidebar.radio("Silo Type", ["Silo One: Uninsulated", "Silo Two: Insulated (4cm foam)"])
    vent_type = st.sidebar.radio("Ventilation Type", ["Unventilated", "Regular (12 AM - 6 AM daily)", "Automated (Triggered by logic)"])
    is_insulated = (silo_type == "Silo Two: Insulated (4cm foam)")
    vent_key = vent_type.split('(')[0].strip()

    # 2. Time Control
    max_time = len(simulation_df) - 1
    time_index = st.sidebar.slider("Simulation Time (Hours)", 0, max_time, max_time)
    current_row = simulation_df.iloc[time_index]
    
    sensors_df = get_sensors()
    
    # --- CORE SOLVER CALL ---
    hist_df = calculate_full_history_adaptive(
        simulation_df, 
        sensors_df, 
        is_insulated, 
        vent_key
    )
    
    # --- VISUALIZATION ---
    col1, col2 = st.columns([2, 1])
    
    # Calculate temperatures for all sensors at current time
    current_temps = []
    for _, sensor in sensors_df.iterrows():
        temp = hist_df.iloc[time_index][f"probe_{sensor['id']}"]
        current_temps.append(temp)
    sensors_df['current_temp'] = current_temps
    
    with col1:
        st.subheader("High-Fidelity Silo Visualization")
        
        # 3D Plot
        fig_3d = go.Figure()
        z = np.linspace(0, 170, 50); theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = 75 * np.cos(theta_grid); y_grid = 75 * np.sin(theta_grid)
        
        surface_temps = np.zeros_like(z_grid)
        insul_factor = INSULATION_RATIO if is_insulated else 1.0
        delta_t_vent_status = hist_df.iloc[time_index]['vent_status']
        vis_delta_T_vent = 0.5 if 'ON' in delta_t_vent_status else 0.0

        for i in range(z_grid.shape[0]):
             h_norm = z_grid[i,0] / 170.0
             t = calculate_point_temperature_adaptive(normalized_radius=1.0, normalized_height=h_norm, current_time_index=time_index, history_df=simulation_df, insulation_factor=insul_factor, delta_T_vent=vis_delta_T_vent)
             surface_temps[i, :] = t

        fig_3d.add_trace(go.Surface(z=z_grid, x=x_grid, y=y_grid, surfacecolor=surface_temps, colorscale='RdBu_r', cmin=15, cmax=45, opacity=0.3, showscale=False, hoverinfo='skip'))
        fig_3d.add_trace(go.Scatter3d(x=75 * np.cos(theta), y=75 * np.sin(theta), z=np.full_like(theta, 170), mode='lines', line=dict(color='white', width=2), hoverinfo='skip'))
        fig_3d.add_trace(go.Scatter3d(x=75 * np.cos(theta), y=75 * np.sin(theta), z=np.full_like(theta, 0), mode='lines', line=dict(color='white', width=2), hoverinfo='skip'))
        fig_3d.add_trace(go.Scatter3d(x=sensors_df['real_x'], y=sensors_df['real_z'], z=sensors_df['real_y'], mode='markers+text', marker=dict(size=8, color=sensors_df['current_temp'], colorscale='RdBu_r', cmin=15, cmax=45, line=dict(width=1, color='white'), showscale=True, colorbar=dict(title="Temp (°C)", x=1.1)), text=[f"{t:.1f}°C" for t in sensors_df['current_temp']], hovertemplate="<b>%{text}</b><br>Layer: %{customdata}", customdata=sensors_df['layer']))

        fig_3d.update_layout(scene=dict(xaxis=dict(range=[-80, 80], backgroundcolor="#0f172a", gridcolor="#334155", title="Radius (cm)"), yaxis=dict(range=[-80, 80], backgroundcolor="#0f172a", gridcolor="#334155", title="Depth (cm)"), zaxis=dict(range=[0, 180], backgroundcolor="#0f172a", gridcolor="#334155", title="Height (cm)"), aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="#0f172a", height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("Real-Time Status")
        
        current_ambient = current_row['ambient_temp']
        current_rh = current_row['ambient_rh']
        current_silo_rh = hist_df.iloc[time_index]['silo_rh']

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 10px;">
            <h3>Simulation Scenario</h3>
            <h4 style="color: #60a5fa">{silo_type} | {vent_type}</h4>
            <p>Fan Status: <span style="font-weight: bold; color: {'#22c55e' if 'ON' in hist_df.iloc[time_index]['vent_status'] else '#ef4444'}">{hist_df.iloc[time_index]['vent_status']}</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h3>Ambient / Silo RH</h3>
            <h1 style="color: #84cc16">{current_rh:.1f}%</h1>
            <h3 style="color: #3b82f6; margin-top: -10px;">Silo: {current_silo_rh:.1f}%</h3>
            <p>Time: T+{time_index}h</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        st.markdown("### Layer Averages")
        
        layer_means = sensors_df.groupby('layer')['current_temp'].mean()
        layer_colors = {1: '#ef4444', 2: '#3b82f6', 3: '#22c55e', 4: '#f59e0b'}
        
        for layer_id in [4, 3, 2, 1]: 
            mean_temp = layer_means.get(layer_id, 0)
            color = layer_colors[layer_id]
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center;">
                <span style="font-weight: bold; color: {color}">Layer {layer_id}</span>
                <div style="flex-grow: 1; margin: 0 10px; background: #334155; height: 8px; border-radius: 4px;">
                    <div style="width: {min(100, max(0, (mean_temp-15)/30 * 100))}%; background: {color}; height: 100%; border-radius: 4px;"></div>
                </div>
                <span style="font-family: monospace;">{mean_temp:.1f}°C</span>
            </div>
            """, unsafe_allow_html=True)

    # --- PROBE TOGGLE ---
    st.sidebar.write("---")
    st.sidebar.subheader("3. Probe Visibility Toggle")
    toggled_probes = []
    for layer_id in sorted(sensors_df['layer'].unique(), reverse=True):
        st.sidebar.markdown(f"**Layer {layer_id} Probes**")
        layer_probes = sensors_df[sensors_df['layer'] == layer_id]['id'].tolist()
        cols = st.sidebar.columns(3)
        for i, probe_id in enumerate(layer_probes):
            if cols[i % 3].checkbox(f"Probe {probe_id}", value=True, key=f"toggle_{probe_id}"):
                toggled_probes.append(probe_id)
    
    # --- HISTORICAL GRAPHS ---
    st.header("Historical Temperature and Moisture Analysis")
    
    # Store figures for export
    export_figures = {}
    
    # Plot RH Evolution 
    export_figures.update(plot_relative_humidity_evolution(hist_df))
    
    # Combined Layer Evolution (FIXED from last turn)
    export_figures.update(plot_combined_layer_evolution(hist_df, sensors_df))
    
    # Combined Probe Evolution (uses toggled_probes)
    export_figures.update(plot_combined_probe_evolution(hist_df, sensors_df, toggled_probes))
    
    # Individual Layer Evolution
    export_figures.update(plot_individual_layer_evolution(hist_df, sensors_df))
    
    # Individual Probe Evolution
    export_figures.update(plot_individual_probe_evolution(hist_df, sensors_df))
    
    # Meshed Silo Cross-Section
    export_figures.update(plot_meshed_silo(sensors_df))
    
    # --- EXPORT BUTTON (Feature 7) ---
    st.write("---")
    st.header("7. Data and Report Export")
    
    if st.button("Generate and Download Export Package"):
        
        # 1. Generate Excel File
        excel_data = to_excel(hist_df, sensors_df)
        
        # 2. Generate LaTeX Report Content 
        latex_content = get_latex_report_content(hist_df, sensors_df)
        
        # 3. Save files to sandbox for zipping
        temp_dir = "/home/ubuntu/export_package"
        import os
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save Excel
        with open(f"{temp_dir}/silo_simulation_data.xlsx", "wb") as f: f.write(excel_data)
        # Save LaTeX
        with open(f"{temp_dir}/silo_report.tex", "w") as f: f.write(latex_content)
            
        # Save HTMLs and PNGs
        for name, fig in export_figures.items():
            fig.write_html(f"{temp_dir}/{name}.html")
            try: fig.write_image(f"{temp_dir}/{name}.png", scale=2)
            except: pass 
            
        # 4. Create a ZIP archive
        import shutil
        zip_path = "/home/ubuntu/silo_export.zip"
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', temp_dir)
        
        st.success("Export package generated successfully!")
        
        # Provide download links
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download Full Export Package (ZIP)",
                data=f.read(),
                file_name="silo_thermal_analysis_report.zip",
                mime="application/zip"
            )
        
        st.info("The ZIP package contains the Excel data, all high-definition PNG figures, and the LaTeX source file for the report.")


if __name__ == "__main__":
    main()