import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loading and Merging
@st.cache_data
def load_and_prepare_data():
    """
    Loads the three CSV files, renames the performance columns appropriately,
    and merges them on the "Athlete" column.
    """
    # Load the CSV files
    df_imtp = pd.read_csv("Ice Hockey Isometric Testing - IMTP.csv")
    df_squat = pd.read_csv("Ice Hockey Isometric Testing - ISO Squat.csv")
    df_lp = pd.read_csv("Ice Hockey Isometric Testing - ISO LP.csv")
    
    # Rename performance columns for IMTP test
    df_imtp = df_imtp.rename(columns={
        "Peak Force (N)": "IMTP_Peak_Force",
        "RFD 200ms (N/s)": "IMTP_RFD_200ms",
        "Time to Peak Force (s)": "IMTP_Time_to_Peak"
    })
    
    # Rename performance columns for ISO Squat test
    df_squat = df_squat.rename(columns={
        "Peak Force (N)": "ISOSquat_Peak_Force",
        "RFD 100ms (N/s)": "ISOSquat_RFD_100ms",
        "Impulse 200ms (N s)": "ISOSquat_Impulse_200ms"
    })
    
    # Rename performance columns for ISO LP test (now Larsen Press)
    df_lp = df_lp.rename(columns={
        "Peak Force (N)": "ISOLP_Peak_Force",
        "RFD 100ms (N/s)": "ISOLP_RFD_100ms",
        "Time to Peak Force (s)": "ISOLP_Time_to_Peak"
    })
    
    # Merge data on "Name" and then rename the merged column to "Athlete"
    df = pd.merge(df_imtp, df_squat, on="Name")
    df = pd.merge(df, df_lp, on="Name")
    df.rename(columns={"Name": "Athlete"}, inplace=True)
    
    return df

df = load_and_prepare_data()

st.sidebar.header("Select Athlete")
athlete_names = df["Athlete"].unique()
selected_athlete = st.sidebar.selectbox("Athlete", athlete_names)

st.title("Isometric Performance Dashboard")
st.markdown("""
This dashboard visualizes peak values extracted from our isometric testing battery.

For each test, three variables are displayed:
- **IMTP:** Peak Force (N), RFD 200ms (N/s), Time to Peak Force (s)
- **ISO Squat:** Peak Force (N), RFD 100ms (N/s), Impulse 200ms (N s)
- **Iso-Larsen Press:** Peak Force (N), RFD 100ms (N/s), Time to Peak Force (s)

Each test’s dashboard (accessible via tabs) includes:
- A **bar chart** comparing all athletes, with the selected athlete highlighted and annotated with exact values.
- A **radar plot** displaying the selected athlete’s Z‑scores (relative to the group) for each variable.
""")

# Create Tabs for Each Test
tabs = st.tabs(["IMTP", "Iso-Squat", "Iso-Larsen Press"])


bar_width = 0.3
primary_offset = bar_width 

# IMTP Tab
with tabs[0]:
    st.header("IMTP Performance")
    imtp_metrics = ["IMTP_Peak_Force", "IMTP_RFD_200ms", "IMTP_Time_to_Peak"]
    imtp_labels_primary = ["Peak Force (N)", "RFD 200ms (N/s)"]
    imtp_label_secondary = "Time to Peak (s)"
    
    # Bar Chart for IMTP
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    indices = np.arange(len(df))
    
    bars_pf = ax.bar(indices - primary_offset, df[imtp_metrics[0]], bar_width,
                     alpha=0.8,
                     color=['red' if athlete == selected_athlete else 'skyblue' for athlete in df["Athlete"]],
                     label=imtp_labels_primary[0])
    bars_rfd = ax.bar(indices + primary_offset, df[imtp_metrics[1]], bar_width,
                      alpha=0.8,
                      color=['orange' if athlete == selected_athlete else 'lightgreen' for athlete in df["Athlete"]],
                      label=imtp_labels_primary[1])
    
    bars_time = ax2.bar(indices, df[imtp_metrics[2]], bar_width,
                        alpha=0.8,
                        color=['gold' if athlete == selected_athlete else '#C8A2C8' for athlete in df["Athlete"]],
                        label=imtp_label_secondary)
    
    ax.set_xlabel("Athletes")
    ax.set_ylabel("Peak Force & RFD")
    ax2.set_ylabel("Time to Peak (s)")
    ax.set_title("IMTP Performance Across the Squad")
    ax.set_xticks(indices)
    ax.set_xticklabels(df["Athlete"], rotation=45, ha="right")
    
    # Annotate bars for primary axis
    for bar in bars_pf:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_rfd:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    # Annotate secondary axis bars
    for bar in bars_time:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    st.pyplot(fig)
    
    # Radar Plot for IMTP
    st.subheader("IMTP Radar Plot (Z-scores)")
    df_imtp_z = df.copy()
    for col in imtp_metrics:
        df_imtp_z[col] = (df[col] - df[col].mean()) / df[col].std()
    df_imtp_z["IMTP_Time_to_Peak"] *= -1
    
    athlete_data = df_imtp_z[df_imtp_z["Athlete"] == selected_athlete].iloc[0]
    values = [athlete_data[col] for col in imtp_metrics]
    values += values[:1]
    
    num_vars = len(imtp_metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig2, ax2 = plt.subplots(subplot_kw={"polar": True}, figsize=(6, 6))
    ax2.plot(angles, values, linewidth=2, linestyle="solid", label=selected_athlete)
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]),
                       ["Peak Force (N)", "RFD 200ms (N/s)", "Time to Peak (s)"])
    ax2.set_title(f"IMTP Radar Plot (Z-scores) for {selected_athlete}")
    st.pyplot(fig2)
    

# ISO Squat Tab
with tabs[1]:
    st.header("Iso-Squat Performance")
    isosquat_metrics = ["ISOSquat_Peak_Force", "ISOSquat_RFD_100ms", "ISOSquat_Impulse_200ms"]
    isosquat_labels_primary = ["Peak Force (N)", "RFD 100ms (N/s)"]
    isosquat_label_secondary = "Impulse 200ms (N s)"
    
    # Bar Chart for ISO Squat
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    indices = np.arange(len(df))
    
    bars_pf = ax.bar(indices - primary_offset, df[isosquat_metrics[0]], bar_width,
                     alpha=0.8,
                     color=['red' if athlete == selected_athlete else 'skyblue' for athlete in df["Athlete"]],
                     label=isosquat_labels_primary[0])
    bars_rfd = ax.bar(indices + primary_offset, df[isosquat_metrics[1]], bar_width,
                      alpha=0.8,
                      color=['orange' if athlete == selected_athlete else 'lightgreen' for athlete in df["Athlete"]],
                      label=isosquat_labels_primary[1])
    
    bars_imp = ax2.bar(indices, df[isosquat_metrics[2]], bar_width,
                       alpha=0.8,
                       color=['gold' if athlete == selected_athlete else '#C8A2C8' for athlete in df["Athlete"]],
                       label=isosquat_label_secondary)
    
    ax.set_xlabel("Athletes")
    ax.set_ylabel("Peak Force & RFD")
    ax2.set_ylabel("Impulse (N s)")
    ax.set_title("Iso-Squat Performance Across the Squad")
    ax.set_xticks(indices)
    ax.set_xticklabels(df["Athlete"], rotation=45, ha="right")
    
    for bar in bars_pf:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_rfd:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_imp:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    st.pyplot(fig)
    
    # Radar Plot for ISO Squat
    st.subheader("Iso-Squat Radar Plot (Z-scores)")
    df_isosquat_z = df.copy()
    for col in isosquat_metrics:
        df_isosquat_z[col] = (df[col] - df[col].mean()) / df[col].std()
        
    athlete_data = df_isosquat_z[df_isosquat_z["Athlete"] == selected_athlete].iloc[0]
    values = [athlete_data[col] for col in isosquat_metrics]
    values += values[:1]
    
    num_vars = len(isosquat_metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig2, ax2 = plt.subplots(subplot_kw={"polar": True}, figsize=(6, 6))
    ax2.plot(angles, values, linewidth=2, linestyle="solid", label=selected_athlete)
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]),
                       ["Peak Force (N)", "RFD 100ms (N/s)", "Impulse 200ms (N s)"])
    ax2.set_title(f"Iso-Squat Radar Plot (Z-scores) for {selected_athlete}")
    st.pyplot(fig2)
    
# Larsen Press Tab 
with tabs[2]:
    st.header("Iso-Larsen Press Performance")
    isolp_metrics = ["ISOLP_Peak_Force", "ISOLP_RFD_100ms", "ISOLP_Time_to_Peak"]
    isolp_labels_primary = ["Peak Force (N)", "RFD 100ms (N/s)"]
    isolp_label_secondary = "Time to Peak (s)"
    
    # Bar Chart for Larsen Press 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    indices = np.arange(len(df))
    
    bars_pf = ax.bar(indices - primary_offset, df[isolp_metrics[0]], bar_width,
                     alpha=0.8,
                     color=['red' if athlete == selected_athlete else 'skyblue' for athlete in df["Athlete"]],
                     label=isolp_labels_primary[0])
    bars_rfd = ax.bar(indices + primary_offset, df[isolp_metrics[1]], bar_width,
                      alpha=0.8,
                      color=['orange' if athlete == selected_athlete else 'lightgreen' for athlete in df["Athlete"]],
                      label=isolp_labels_primary[1])
    
    bars_time = ax2.bar(indices, df[isolp_metrics[2]], bar_width,
                        alpha=0.8,
                        color=['gold' if athlete == selected_athlete else '#C8A2C8' for athlete in df["Athlete"]],
                        label=isolp_label_secondary)
    
    ax.set_xlabel("Athletes")
    ax.set_ylabel("Peak Force & RFD")
    ax2.set_ylabel("Time to Peak (s)")
    ax.set_title("Iso-Larsen Press Performance Across the Squad")
    ax.set_xticks(indices)
    ax.set_xticklabels(df["Athlete"], rotation=45, ha="right")
    
    for bar in bars_pf:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_rfd:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_time:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    st.pyplot(fig)
    
    # Radar Plot for Larsen Press
    st.subheader("Iso-Larsen Press Radar Plot (Z-scores)")
    df_isolp_z = df.copy()
    for col in isolp_metrics:
        df_isolp_z[col] = (df[col] - df[col].mean()) / df[col].std()
      
    df_isolp_z["ISOLP_Time_to_Peak"] *= -1
        
    athlete_data = df_isolp_z[df_isolp_z["Athlete"] == selected_athlete].iloc[0]
    values = [athlete_data[col] for col in isolp_metrics]
    values += values[:1]
    
    num_vars = len(isolp_metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig2, ax2 = plt.subplots(subplot_kw={"polar": True}, figsize=(6, 6))
    ax2.plot(angles, values, linewidth=2, linestyle="solid", label=selected_athlete)
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]),
                       ["Peak Force (N)", "RFD 100ms (N/s)", "Time to Peak (s)"])
    ax2.set_title(f"Iso-Larsen Press Radar Plot (Z-scores) for {selected_athlete}")
    st.pyplot(fig2)
