import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st

st.title("Project Budget Forecast Dashboard")

# File uploader for flexibility
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Sheet2", engine="openpyxl")

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
    projects = df['Project'].unique()

    for project in projects:
        df_proj = df[df['Project'] == project].copy()
        le = LabelEncoder()
        df_proj['Role_encoded'] = le.fit_transform(df_proj['Role'])
        df_proj['Leaves'] = project_leaves[project]

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
        for sprint in range(4, 11):
            if sprint in [9, 10]:
                roles_to_include = project_roles_9_10[project]
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
                    'Leaves': project_leaves[project]
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

        # Plot with matplotlib and show in Streamlit
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
