"""
Pure-Python Monte Carlo DCF Simulation (Streamlit page)

You manually hardcode:
    - Free cash flows
    - Terminal value method
    - Shares outstanding
"""

# 1. Imports
import numpy as np
import pandas as pd
import streamlit as st
#import altair as alt
import plotly.express as px


# 2. Global constants (user inputs)
# Balance sheet adjustments (millions)
CASH = 1909164      # example: 1,909,164 mm
DEBT = 0
MINORITY_INTEREST = 1121


# Free cash flows for forecast years (in millions)
FCF = np.array([
    16911,   # Year 1
    226868,  # Year 2
    378794,  # Year 3
    671380,  # Year 4
    818350   # Year 5
])

# Shares outstanding (in millions)
SHARES_OUT = 1164.20

# Terminal value method: choose "perpetuity" or "exit"
TERMINAL_METHOD = "perpetuity"

# If using perpetuity growth:
PGR_MEAN = 0.02
PGR_STD = 0.0025

# If using exit multiple:
EXIT_LOW = 6.0
EXIT_MODE = 8.0
EXIT_HIGH = 10.0
EBITDA_TERM_YEAR = 6200  # example terminal EBITDA

# WACC distribution
WACC_MEAN = 0.1035
WACC_STD = 0.015

# Number of Monte Carlo trials
N = 5000

# Random seed
SEED = 42


st.set_page_config(
    page_title="Monte Carlo Simulation",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Helper functions

def terminal_value_perpetuity(last_fcf, wacc, g):
    """Gordon Growth Model."""
    return last_fcf * (1 + g) / (wacc - g)


def terminal_value_multiple(exit_multiple, terminal_ebitda):
    """Exit multiple terminal value."""
    return exit_multiple * terminal_ebitda


# 4. Core DCF functions

def dcf_valuation(fcf, wacc, g=None, exit_multiple=None):
    """
    Returns enterprise value using DCF.
    """
    years = np.arange(1, len(fcf) + 1)

    # Discounted FCF
    discount_factors = 1 / ((1 + wacc) ** years)
    pv_fcf = np.sum(fcf * discount_factors)

    # Terminal value
    if TERMINAL_METHOD == "perpetuity":
        tv = terminal_value_perpetuity(fcf[-1], wacc, g)
    else:
        tv = terminal_value_multiple(exit_multiple, EBITDA_TERM_YEAR)

    # Discount terminal value
    pv_tv = tv / ((1 + wacc) ** len(fcf))

    return pv_fcf + pv_tv


# 5. Monte Carlo simulation function

def run_simulation():
    np.random.seed(SEED)

    # Draw random variables
    wacc_samples = np.random.normal(WACC_MEAN, WACC_STD, N)

    if TERMINAL_METHOD == "perpetuity":
        g_samples = np.random.normal(PGR_MEAN, PGR_STD, N)
        exit_samples = np.zeros(N)
    else:
        g_samples = np.zeros(N)
        exit_samples = np.random.triangular(EXIT_LOW, EXIT_MODE, EXIT_HIGH, N)

    results = []

    for i in range(N):
        # Enterprise value from DCF
        enterprise_value = dcf_valuation(
            fcf=FCF,
            wacc=wacc_samples[i],
            g=g_samples[i],
            exit_multiple=exit_samples[i],
        )

        # Convert to equity value and implied share price
        equity_value = enterprise_value + CASH - DEBT - MINORITY_INTEREST
        implied_share_price = equity_value / SHARES_OUT

        results.append({
            "trial": i,
            "wacc": wacc_samples[i],
            "g": g_samples[i],
            "exit_multiple": exit_samples[i],
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "price_per_share": implied_share_price,
        })

    df = pd.DataFrame(results)
    df.to_csv("mc_results_hardcoded.csv", index=False)

    print("Monte Carlo completed.")
    print("Implied share price distribution (¥):")
    print(df["price_per_share"].describe())
    print("Saved results to mc_results_hardcoded.csv")

    return df


# 6. Run the simulation → df = run_simulation()

df = run_simulation()

# 7. Visualization code (Streamlit)

if isinstance(df, pd.DataFrame):
    st.subheader("Monte Carlo Simulation Results")

    # Price per share distribution (Plotly histogram)
    fig = px.histogram(
        df,
        x="price_per_share",
        nbins=100,
        title="Price Per Share Distribution",
        opacity=0.75,
        marginal="box",  # adds a box plot above — very clean
    )
    fig.update_layout(
        xaxis_title="Price per Share (¥)",
        yaxis_title="Frequency",
        bargap=0.02,
        height=600,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats (exclude trial, exit_multiple, and 'count' row)
    st.write("### Summary Statistics")
    summary_cols = ["wacc", "g", "enterprise_value", "equity_value", "price_per_share"]
    stats = df[summary_cols].describe().drop(index=["count", "std"])
    # Convert wacc and g from decimals to percentage values
    stats[["wacc", "g"]] = stats[["wacc", "g"]] * 100
    stats["wacc"] = stats["wacc"].map(lambda x: f"{x:.2f}%")
    stats["g"] = stats["g"].map(lambda x: f"{x:.2f}%")
    # Format enterprise value, equity value, and price per share as yen with commas and 2 decimals
    stats["enterprise_value"] = stats["enterprise_value"].map(
        lambda x: f"¥{x:,.2f}"
    )
    stats["equity_value"] = stats["equity_value"].map(
        lambda x: f"¥{x:,.2f}"
    )
    stats["price_per_share"] = stats["price_per_share"].map(
        lambda x: f"¥{x:,.2f}"
    )
    st.write(stats)
else:
    st.error("Simulation output is not a DataFrame. Cannot generate chart.")
