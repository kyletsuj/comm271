"""
Pure-Python Monte Carlo DCF Simulation
(Paste into Cursor)

You manually hardcode:
    - Free cash flows
    - Terminal value method
    - Shares outstanding

Run:
    python monte_carlo_dcf.py
"""

import numpy as np
import pandas as pd

# -------------------------------------
# USER INPUT SECTION — EDIT THIS PART
# -------------------------------------

# Free cash flows for forecast years (in millions USD/CAD etc.)
FCF = np.array([
    16911,  # Year 1
    226868,  # Year 2
    378794,  # Year 3
    671380,  # Year 4
    818350   # Year 5
])

# Shares outstanding
SHARES_OUT = 1164.20  # in millions (example)

# Terminal value method: choose "perpetuity" or "exit"
TERMINAL_METHOD = "perpetuity"

# If using perpetuity growth:
PGR_MEAN = 0.02
PGR_STD = 0.005

# If using exit multiple:
EXIT_LOW = 6.0
EXIT_MODE = 8.0
EXIT_HIGH = 10.0
EBITDA_TERM_YEAR = 6200  # example terminal EBITDA

# WACC distribution
WACC_MEAN = 0.0935
WACC_STD = 0.01

# Number of Monte Carlo trials
N = 5000

# Random seed
SEED = 42


# -------------------------------------
# CORE DCF FUNCTIONS
# -------------------------------------

def terminal_value_perpetuity(last_fcf, wacc, g):
    """ Gordon Growth Model """
    return last_fcf * (1 + g) / (wacc - g)

def terminal_value_multiple(exit_multiple, terminal_ebitda):
    """ Exit multiple terminal value """
    return exit_multiple * terminal_ebitda


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


# -------------------------------------
# MONTE CARLO RUN
# -------------------------------------

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
        val = dcf_valuation(
            fcf=FCF,
            wacc=wacc_samples[i],
            g=g_samples[i],
            exit_multiple=exit_samples[i]
        )

        price_per_share = val / SHARES_OUT

        results.append({
            "trial": i,
            "wacc": wacc_samples[i],
            "g": g_samples[i],
            "exit_multiple": exit_samples[i],
            "enterprise_value": val,
            "price_per_share": price_per_share
        })

    df = pd.DataFrame(results)
    df.to_csv("mc_results_hardcoded.csv", index=False)

    print("Monte Carlo completed.")
    print(df["price_per_share"].describe())
    print("Saved results to mc_results_hardcoded.csv")

    return df

# ---- STREAMLIT PRICE DISTRIBUTION GRAPH ----
import streamlit as st
import pandas as pd
import altair as alt

st.subheader("Monte Carlo Price Distribution")

# Convert results to a DataFrame if needed
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame({"price_per_share": df})

# Altair histogram chart
chart = (
    alt.Chart(df)
    .mark_bar(opacity=0.85)
    .encode(
        alt.X("price_per_share:Q", bin=alt.Bin(maxbins=50), title="Price per Share"),
        alt.Y("count()", title="Frequency"),
    )
    .properties(height=300)
)

st.altair_chart(chart, use_container_width=True)
