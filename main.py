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
    # Adjust FCF values
    adjusted_sum_pv_fcf = base_values["Sum_PV_FCF"] * fcf_mult
    
    # Adjust terminal value
    # TV = final year FCF * (1 + growth rate) / (WACC - growth rate)
    adjusted_terminal_value = UNLEVERED_FCF[2030] * (1 + growth_rate) / (wacc - growth_rate)
    
    # Recalculate PV of terminal value with new WACC
    # Using simplified approach: PV = TV / (1 + WACC)^n where n is number of years
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
    growth_rate = st.slider(
        "Perpetuity Growth Rate (%)",
        min_value=0.0,
        max_value=5.0,
        value=float(PERPETUITY_GROWTH_METHOD["Growth_Rate"] * 100),
        step=0.1,
        help="Terminal growth rate assumption"
    ) / 100
    
    st.markdown("---")
    st.subheader("Individual FCF Years")
    
    # Individual FCF year controls
    adjusted_fcf = {}
    for year in sorted(UNLEVERED_FCF.keys()):
        base_value = UNLEVERED_FCF[year]
        adjusted_value = st.number_input(
            f"{year} (¥000s)",
            min_value=0,
            max_value=int(base_value * 3),
            value=int(base_value),
            step=1000,
            key=f"fcf_{year}"
        )
        adjusted_fcf[year] = adjusted_value

# Set default multipliers to 1.0 (no adjustment)
terminal_value_mult = 1.0
fcf_mult = 1.0

# Calculate adjusted metrics
adjusted_share_price, adjusted_equity_value, adjusted_enterprise_value = calculate_share_price(
    wacc, growth_rate, terminal_value_mult, fcf_mult, PERPETUITY_GROWTH_METHOD
)

# Key Metrics Display
st.header("📈 Key Valuation Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Share Price",
        f"¥{PERPETUITY_GROWTH_METHOD['Current_Share_Price']:,}",
        delta=None
    )

with col2:
    base_implied = PERPETUITY_GROWTH_METHOD["Implied_Share_Price"]
    delta_price = adjusted_share_price - base_implied
    st.metric(
        "Implied Share Price",
        f"¥{adjusted_share_price:,.0f}",
        delta=f"¥{delta_price:,.0f}" if abs(delta_price) > 1 else None
    )

with col3:
    upside_pct = ((adjusted_share_price - PERPETUITY_GROWTH_METHOD["Current_Share_Price"]) / 
                  PERPETUITY_GROWTH_METHOD["Current_Share_Price"]) * 100
    st.metric(
        "Upside/Downside",
        f"{upside_pct:.1f}%",
        delta=f"{upside_pct - (PERPETUITY_GROWTH_METHOD['Implied_Upside'] * 100):.1f}%" if abs(upside_pct - (PERPETUITY_GROWTH_METHOD['Implied_Upside'] * 100)) > 0.1 else None
    )

with col4:
    st.metric(
        "Equity Value",
        f"¥{adjusted_equity_value:,.0f}",
        delta=f"¥{adjusted_equity_value - PERPETUITY_GROWTH_METHOD['Equity_Value']:,.0f}"
    )

st.markdown("---")

# Charts Row 1: FCF Timeline and Waterfall
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Unlevered FCF Timeline")
    # Create FCF timeline chart
    fcf_df = pd.DataFrame({
        'Year': list(adjusted_fcf.keys()),
        'FCF (¥000s)': list(adjusted_fcf.values())
    })
    
    fig_fcf = go.Figure()
    fig_fcf.add_trace(go.Bar(
        x=fcf_df['Year'],
        y=fcf_df['FCF (¥000s)'],
        marker_color='rgb(55, 83, 109)',
        name='Unlevered FCF'
    ))
    fig_fcf.add_trace(go.Scatter(
        x=fcf_df['Year'],
        y=fcf_df['FCF (¥000s)'],
        mode='lines+markers',
        name='Trend',
        line=dict(color='rgb(255, 127, 14)', width=3),
        marker=dict(size=8)
    ))
    fig_fcf.update_layout(
        title="Free Cash Flow Over Time",
        xaxis_title="Year",
        yaxis_title="FCF (¥000s)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_fcf, use_container_width=True)
    st.markdown("hi")

with col2:
    st.subheader("💧 Valuation Waterfall")
    # Create waterfall chart
    # Recalculate PV Terminal Value with adjusted WACC
    n_years = 9
    adjusted_pv_terminal = (PERPETUITY_GROWTH_METHOD["Terminal_Value"] * terminal_value_mult) / ((1 + wacc) ** n_years)
    
    waterfall_values = [
        PERPETUITY_GROWTH_METHOD["Sum_PV_FCF"] * fcf_mult,
        adjusted_pv_terminal,
        PERPETUITY_GROWTH_METHOD["Cash"],
        -PERPETUITY_GROWTH_METHOD["Debt"],
        -PERPETUITY_GROWTH_METHOD["Minority_Interest"]
    ]
    
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=['Sum PV FCF', 'PV Terminal Value', 'Cash', 'Debt', 'Minority Interest', 'Equity Value'],
        textposition="outside",
        text=[f"¥{v:,.0f}" for v in waterfall_values] + [f"¥{adjusted_equity_value:,.0f}"],
        y=waterfall_values + [0],  # Last value is 0, will show as total
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "rgb(55, 83, 109)"}},
        decreasing={"marker": {"color": "rgb(219, 64, 82)"}},
        totals={"marker": {"color": "rgb(46, 125, 50)"}},
        
    ))
    
    fig_waterfall.update_layout(
        title="DCF Valuation Components",
        yaxis_title="Value (¥000s)",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

st.markdown("---")

# Sensitivity Analysis Section
st.header("🔥 Sensitivity Analysis")

# Generate sensitivity data
wacc_range = np.linspace(0.05, 0.15, 21)
growth_range = np.linspace(0.00, 0.05, 11)
terminal_mult_range = np.linspace(0.5, 2.0, 16)
fcf_mult_range = np.linspace(0.5, 2.0, 16)

# Determine min/max for normalization
z_min, z_mid, z_max = 6000, 8000, 10000

custom_colorscale = [
    [0.0, "green"],     # 6000
    [0.5, "yellow"],    # 8000
    [1.0, "red"]        # 10000
]# --- FIX: Build sensitivity_wacc_growth before using it ---
sensitivity_wacc_growth = np.zeros((len(growth_range), len(wacc_range)))

for i, gr in enumerate(growth_range):
    for j, w in enumerate(wacc_range):
        try:
            share_price, _, _ = calculate_share_price(
                w, gr, 1.0, 1.0, PERPETUITY_GROWTH_METHOD
            )
        except ZeroDivisionError:
            share_price = np.nan  # handle WACC == g edge case safely
        sensitivity_wacc_growth[i, j] = share_price
# ----------------------------------------------------------
fig_heatmap1 = go.Figure(data=go.Heatmap)


fig_heatmap1 = go.Figure(data=go.Heatmap(
    z=sensitivity_wacc_growth,
    x=[f"{w*100:.1f}%" for w in wacc_range],
    y=[f"{g*100:.1f}%" for g in growth_range],
    colorscale=custom_colorscale,
    zmin=z_min,
    zmax=z_max,
    colorbar=dict(title="Share Price (¥)"),
    hovertemplate='WACC: %{x}<br>Growth: %{y}<br>Share Price: ¥%{z:,.0f}<extra></extra>'
))

fig_heatmap1.update_layout(
    xaxis_title="WACC (%)",
    yaxis_title="Growth Rate (%)",
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig_heatmap1, use_container_width=True)



# Additional DCF Details
st.markdown("---")
st.header("📋 DCF Model Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Cash Flow Details")
    st.write(f"**Sum of PV FCF:** ¥{PERPETUITY_GROWTH_METHOD['Sum_PV_FCF'] * fcf_mult:,.0f}")
    st.write(f"**PV Terminal Value:** ¥{PERPETUITY_GROWTH_METHOD['PV_Terminal_Value'] * terminal_value_mult:,.0f}")
    st.write(f"**Enterprise Value:** ¥{adjusted_enterprise_value:,.0f}")

with col2:
    st.subheader("Balance Sheet Adjustments")
    st.write(f"**Cash:** ¥{PERPETUITY_GROWTH_METHOD['Cash']:,.0f}")
    st.write(f"**Debt:** ¥{PERPETUITY_GROWTH_METHOD['Debt']:,.0f}")
    st.write(f"**Minority Interest:** ¥{PERPETUITY_GROWTH_METHOD['Minority_Interest']:,.0f}")

with col3:
    st.subheader("Per Share Metrics")
    st.write(f"**Diluted Shares Outstanding:** {PERPETUITY_GROWTH_METHOD['Diluted_Shares_Outstanding_mm']:,.2f}M")
    st.write(f"**Implied Exit Multiple:** {PERPETUITY_GROWTH_METHOD['Implied_Exit_Multiple']:.2f}x")
    st.write(f"**Base Case Upside:** {PERPETUITY_GROWTH_METHOD['Implied_Upside']*100:.1f}%")
