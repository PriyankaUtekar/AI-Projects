import os
import io
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st

st.title("Project Budget Forecast Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Load data
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Sheet2", engine="openpyxl")
    st.success("Using uploaded file.")
elif os.path.exists("sample_data.xlsx"):
    df = pd.read_excel("sample_data.xlsx", sheet_name="Sheet2", engine="openpyxl")
    st.info("No file uploaded. Using default sample_data.xlsx from the repo.")
else:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# Validate required columns
if 'Project' not in df.columns or 'Sprint' not in df.columns:
    st.warning("Required columns 'Project' and 'Sprint' not found in the uploaded file.")
    st.stop()

# Filters
project_options = df['Project'].dropna().unique().tolist()
selected_projects = st.multiselect("Select Project(s)", options=project_options, default=project_options)

max_sprint = int(df['Sprint'].max())
sprint_range = st.slider("Select Sprint Range", min_value=1, max_value=max_sprint, value=(1, max_sprint))

# Apply filters
filtered_df = df[df['Project'].isin(selected_projects) & df['Sprint'].between(sprint_range[0], sprint_range[1])]

# Export filtered data to Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    filtered_df.to_excel(writer, index=False, sheet_name='FilteredData')
output.seek(0)

st.download_button(
    label="📥 Download Filtered Data as Excel",
    data=output,
    file_name="filtered_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Project-specific parameters
project_leaves = {
    "Banking": 1,
    "Healthcare": 3,
    "Energy": 1.5
}
project_roles_9_10 = {
    "Banking": ['PM', 'Power BI Dev 1', 'Power BI Dev 2', 'Data Scientist'],
    "Healthcare": ['PM', 'Power BI Dev 1', 'Power BI Dev 2', 'Data Scientist'],
    "Energy": ['PM', 'Developer1', 'Developer2']
}
bandwidth_overrides = {
    "PM": 0.25,
    "IRM": 0.1,
    "Functional Consultant": 0.1
}

# Forecasting and plotting
projects = filtered_df['Project'].unique()
for project in projects:
    df_proj = filtered_df[filtered_df['Project'] == project].copy()
    le = LabelEncoder()
    df_proj['Role_encoded'] = le.fit_transform(df_proj['Role'])
    df_proj['Leaves'] = project_leaves.get(project, 0)
    features = ['Role_encoded', 'Bandwidth', 'FTE/week', 'Role Cost',
                'Cost / week', 'Cost / sprint', 'Sprint', 'Leaves']
    df_planned = df_proj[(df_proj['Planned'].notna()) & (df_proj['Planned'] != 0)]
    df_actuals = df_proj[(df_proj['Actuals'].notna()) & (df_proj['Actuals'] != 0)]
    if df_planned.empty or df_actuals.empty:
        st.warning(f"Skipping {project} due to insufficient data.")
        continue
    X_planned = df_planned[features]
    y_planned = df_planned['Planned']
    X_actuals = df_actuals[features]
    y_actuals = df_actuals['Actuals']
    model_planned = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_planned.fit(X_planned, y_planned)
    model_actuals = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_actuals.fit(X_actuals, y_actuals)
    future_sprints = []
    for sprint in range(sprint_range[0], sprint_range[1] + 1):
        if sprint in [9, 10]:
            roles_to_include = project_roles_9_10.get(project, [])
        else:
            roles_to_include = df_proj[df_proj['Sprint'] == 3]['Role'].unique()
        for _, row in df_proj[(df_proj['Sprint'] == 3) & (df_proj['Role'].isin(roles_to_include))].iterrows():
            role = row['Role']
            bandwidth = row['Bandwidth']
            if sprint in [9, 10] and ('Dev' in role or 'Developer' in role):
                bandwidth *= 0.5
            if sprint in [8, 9, 10] and role in bandwidth_overrides:
                bandwidth = bandwidth_overrides[role]
            future_sprints.append({
                'Role_encoded': row['Role_encoded'],
                'Bandwidth': bandwidth,
                'FTE/week': row['FTE/week'],
                'Role Cost': row['Role Cost'],
                'Cost / week': row['Cost / week'],
                'Cost / sprint': row['Cost / sprint'],
                'Sprint': sprint,
                'Leaves': project_leaves.get(project, 0)
            })
    future_df = pd.DataFrame(future_sprints)
    if not future_df.empty:
        future_df['Forecast_Planned'] = model_planned.predict(future_df[features])
        future_df['Forecast_Actuals'] = model_actuals.predict(future_df[features])
        forecast_summary = future_df.groupby('Sprint')[['Forecast_Planned', 'Forecast_Actuals']].sum().reset_index()
    else:
        forecast_summary = pd.DataFrame(columns=['Sprint', 'Forecast_Planned', 'Forecast_Actuals'])
    actual_summary = df_proj.groupby('Sprint')[['Planned', 'Actuals']].sum().reset_index()
    actual_summary = actual_summary[actual_summary['Sprint'].isin([1, 2, 3])]
    combined_sprints = list(actual_summary['Sprint']) + list(forecast_summary['Sprint'])
    combined_planned = list(actual_summary['Planned']) + list(forecast_summary['Forecast_Planned'])
    combined_actuals = list(actual_summary['Actuals']) + list(forecast_summary['Forecast_Actuals'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(combined_sprints, combined_planned, marker='o', label='Planned Budget (USD)')
    ax.plot(combined_sprints, combined_actuals, marker='o', label='Actual Budget (USD)')
    ax.set_title(f'Forecasted vs Actual Budget per Sprint - {project}')
    ax.set_xlabel('Sprint')
    ax.set_ylabel('Budget in USD')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
