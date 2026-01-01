import streamlit as st
import pandas as pd
from io import BytesIO

def apply_custom_style():
    """
    Applies minimal styling to layout and padding, 
    but lets standard Streamlit colors (Light/Dark mode) take over.
    """
    st.markdown("""
        <style>
        /* Increase width of the main container for better table visibility */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }
        
        /* Minimal adjustment to make metric cards look neat */
        div[data-testid="stMetric"] {
            border: 1px solid #e6e6e6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def convert_df_to_excel(df):
    """
    Converts DataFrame to Excel for download.
    Fixes 'timezone' error by removing timezone info from datetime columns.
    """
    output = BytesIO()
    
    # --- FIX: Create a copy to avoid modifying the displayed dataframe ---
    export_df = df.copy()
    
    # Iterate through columns to find datetimes and strip timezone
    for col in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[col]):
            # Start by coercing to datetime just in case it's mixed
            export_df[col] = pd.to_datetime(export_df[col], errors='coerce')
            # Remove timezone if it exists
            if export_df[col].dt.tz is not None:
                export_df[col] = export_df[col].dt.tz_localize(None)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # Add basic formatting
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)
        
    processed_data = output.getvalue()
    return processed_data