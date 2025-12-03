import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Nintendo DCF Valuation Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base data constants
UNLEVERED_FCF = {
    2022: 312_785,
    2023: 135_293,
    2024: 26_931,
    2025: 129_254,
    2026: 16_991,
    2027: 226_868,
    2028: 378_794,
    2029: 671_380,
    2030: 818_350,
}

PERPETUITY_GROWTH_METHOD = {
    "Sum_PV_FCF": 1_487_996,
    "Growth_Rate": 0.02,
    "WACC": 0.0935,
    "Terminal_Value": 11_358_236,
    "PV_Terminal_Value": 7_265_020,
    "Enterprise_Value": 8_757_256,
    "Cash": 1_909_164,
    "Debt": 0,
    "Minority_Interest": 1_121,
    "Equity_Value": 10_661_059,
    "Diluted_Shares_Outstanding_mm": 1164.20,
    "Implied_Share_Price": 9157,
    "Current_Share_Price": 13005,
    "Implied_Upside": -0.2959,
    "Implied_Exit_Multiple": 11.72,
}

# Helper function to calculate share price from assumptions
def calculate_share_price(wacc, growth_rate, terminal_value_mult, fcf_mult, base_values):
    """Calculate implied share price based on adjusted assumptions"""
    # Protect against invalid input where wacc <= growth_rate
    if wacc <= growth_rate:
        raise ZeroDivisionError("WACC must be greater than growth rate for Gordon Growth formula.")

    # Adjust FCF values
    adjusted_sum_pv_fcf = base_values["Sum_PV_FCF"] * fcf_mult
    
    # Adjust terminal value (Gordon Growth)
    adjusted_terminal_value = UNLEVERED_FCF[2030] * (1 + growth_rate) / (wacc - growth_rate)
    
    # Recalculate PV of terminal value with new WACC
    n_years = 5
    adjusted_pv_terminal = adjusted_terminal_value / ((1 + wacc) ** n_years)
    
    # Calculate enterprise value
    adjusted_enterprise_value = adjusted_sum_pv_fcf + adjusted_pv_terminal
    
    # Calculate equity value
    adjusted_equity_value = adjusted_enterprise_value + base_values["Cash"] - base_values["Debt"] - base_values["Minority_Interest"]
    
    # Calculate share price
    adjusted_share_price = adjusted_equity_value / base_values["Diluted_Shares_Outstanding_mm"]
    
    return adjusted_share_price, adjusted_equity_value, adjusted_enterprise_value

# Title
st.title("🎮 Nintendo DCF Valuation Dashboard")
st.markdown("---")

# Sidebar with controls
with st.sidebar:
    st.header("📊 Assumption Controls")
    
    # WACC control
    wacc = st.slider(
        "WACC (%)",
        min_value=5.0,
        max_value=15.0,
        value=float(PERPETUITY_GROWTH_METHOD["WACC"] * 100),
        step=0.1,
        help="Weighted Average Cost of Capital"
    ) / 100
    
    # Growth Rate control
    growth_rate = st.slider
    ("Perpetuity Growth Rate (%)"),
    min_value=0.0,
    max_value=10.0,
    value=float(PERPETUITY_GROWTH_METHOD["Growth_Rate"] * 100),
    step=0.1,
    help="Perpetuity Growth Rate"
    # ) / 100
    
    # Terminal Value Multiplier control
    terminal_value_mult = st.slider
