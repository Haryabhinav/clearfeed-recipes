import io
import pandas as pd
import streamlit as st
from datetime import datetime

def convert_df_to_excel(df):
    """
    Converts a Pandas DataFrame into a binary Excel file object 
    that Streamlit can download.
    """
    output = io.BytesIO()
    # Use 'xlsxwriter' or 'openpyxl' engine
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis_Results')
        
        # Auto-adjust column widths (Optional polish)
        worksheet = writer.sheets['Analysis_Results']
        for idx, col in enumerate(df.columns):
            # precise width calculation or default to 20
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)
            
    processed_data = output.getvalue()
    return processed_data

def format_iso_date(iso_str):
    """
    Converts raw ISO dates (e.g., '2023-11-12T14:30:00Z') 
    into a friendly readable format (e.g., 'Nov 12, 2023').
    """
    if not iso_str:
        return ""
    try:
        # Handle 'Z' or standard ISO format
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%b %d, %Y')
    except:
        return iso_str

def apply_custom_style():
    """
    Injects custom CSS to hide default Streamlit branding 
    and maximize screen real estate.
    """
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Make the dataframe header stand out */
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
        """, unsafe_allow_html=True)