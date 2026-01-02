import streamlit as st
import pandas as pd
from io import BytesIO

def apply_custom_style():
    """Injects custom CSS for better layout and metric card styling."""
    st.markdown("""
        <style>
        /* Maximize screen width for better table visibility */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }
        /* Style metric cards with a clean border but transparent background */
        div[data-testid="stMetric"] {
            border: 1px solid #303030; /* Darker border for dark mode compatibility */
            padding: 10px;
            border-radius: 5px;
            
        }
        </style>
    """, unsafe_allow_html=True)

def convert_df_to_excel(df):
    """
    Converts DataFrame to Excel bytes.
    Handles timezone removal (to prevent Excel crashes) and standardizes to UTC.
    """
    output = BytesIO()
    export_df = df.copy()
    
    time_col = "created_at"
    
    if time_col in export_df.columns:
        export_df[time_col] = pd.to_datetime(export_df[time_col], errors='coerce')
        
        if export_df[time_col].dt.tz is not None:
            # Convert to UTC and remove timezone info to make Excel compatible
            export_df[time_col] = export_df[time_col].dt.tz_convert("UTC").dt.tz_localize(None)
            
        # Clarify timezone in header
        export_df = export_df.rename(columns={time_col: "Created At (UTC)"})

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # Format Column A (System ID) as text so it doesn't look like a float
        text_format = workbook.add_format({'num_format': '@'}) 
        worksheet.set_column('A:A', 15, text_format)
        
    return output.getvalue()