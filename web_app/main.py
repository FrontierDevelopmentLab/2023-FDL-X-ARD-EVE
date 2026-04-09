"""Virtual EVE Demo — Streamlit web application.

Recreated version of the original sdomldemo.org website.
Reads AIA data from AWS S3 and runs on-the-fly inference using the
pre-trained Virtual EVE irradiance model.
"""

import datetime
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_access import (
    AIA_WAVELENGTHS,
    build_time_index,
    get_aia_image,
    get_aia_root,
    get_available_dates,
    get_timestamps_in_range,
)
from inference import load_model, predict_eve_timeseries

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Virtual EVE — Solar Irradiance Demo",
    page_icon="assets/sdo_icon.jpeg",
    layout="wide",
)

st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)

# ── Cached resources (loaded once) ──────────────────────────────────────────


@st.cache_resource(show_spinner="Connecting to S3 data store...")
def init_aia():
    return get_aia_root()


@st.cache_resource(show_spinner="Building time index (first run may take a few minutes)...")
def init_time_index(_aia_root):
    return build_time_index(_aia_root)


@st.cache_resource(show_spinner="Loading Virtual EVE model...")
def init_model():
    return load_model()


aia_root = init_aia()
time_index = init_time_index(aia_root)
model, aia_norms, wavelengths, eve_ions = init_model()
date_min, date_max = get_available_dates(time_index)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align: center;'>Virtual EVE — Solar Irradiance Demo</h1>",
    unsafe_allow_html=True,
)
st.write("#")

tab_eve, tab_about = st.tabs(["Virtual EVE", "About"])

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("assets/fdlx.png", width=280)
    st.write("#")
    st.write(
        "Welcome to the Virtual EVE Demo. This app predicts solar EUV "
        "irradiance from SDO/AIA imagery using a deep learning model trained "
        "during FDL-X 2023."
    )
    st.caption(f"Data available from **{date_min.date()}** to **{date_max.date()}**")

    st.write("## Select Date Range")
    start_date = st.date_input(
        "Start Date",
        value=datetime.date(2017, 9, 6),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )
    start_time = st.time_input("Start Time", datetime.time(0, 0), step=datetime.timedelta(minutes=36))
    end_date = st.date_input(
        "End Date",
        value=datetime.date(2017, 9, 6),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )
    end_time = st.time_input("End Time", datetime.time(23, 59), step=datetime.timedelta(minutes=36))

    start_dt = datetime.datetime.combine(start_date, start_time)
    end_dt = datetime.datetime.combine(end_date, end_time)

    valid = start_dt <= end_dt

# ── Pages ────────────────────────────────────────────────────────────────────

with tab_about:
    st.write("## About")

    st.write(
        """
        Earth's upper atmosphere is principally governed by the Sun's extreme
        ultraviolet (EUV) radiation. Sudden increases during solar flares and
        geomagnetic storms can disrupt long-range communications, induce
        currents that damage power grids, degrade satellite hardware, and
        corrupt on-board data. Accurate, continuous measurement of solar EUV
        irradiance is therefore critical for space-weather forecasting.

        NASA's **Solar Dynamics Observatory (SDO)**, launched in 2010 as part
        of the *Living With a Star* program, carried the **EVE** instrument
        with two modules — **MEGS-A** and **MEGS-B** — that together measured
        irradiance across 39 spectral lines. **In 2014, a capacitor short
        destroyed MEGS-A**, eliminating coverage of 14 key EUV lines and
        substantially diminishing the scientific return.

        **Virtual EVE** restores this lost measurement capability using deep
        learning. By training on the four years of overlapping AIA imagery and
        EVE measurements (2010–2014), the model learns to predict what MEGS-A
        *would* have measured — effectively virtualising the broken instrument
        without any hardware repair.
        """
    )

    st.write("### Model Architecture")
    st.write(
        r"""
        The model uses a **hybrid Linear + EfficientNet CNN** architecture:

        $$O_{\text{total}} = O_{\text{linear}} + \lambda \cdot O_{\text{CNN}}$$

        - **Linear component** — a single-layer feedforward network that
          captures the bulk relationship between mean/std image statistics and
          irradiance.
        - **CNN component** — an **EfficientNet-B5** backbone (~30M parameters)
          that extracts spatial features from the full 512 x 512 px images.
        - **Two-phase training** — the linear model is trained first (20 epochs),
          then the CNN is activated while the linear weights are frozen
          (30 epochs), ensuring stable convergence.
        - **Loss function** — Huber Loss, which is robust to outliers compared
          to traditional MSE.

        **Input:** 9 SDO/AIA channels (94, 131, 171, 193, 211, 304, 335, 1600,
        1700 Angstrom) at 512 x 512 px resolution.

        **Output:** **38 EVE ion channels** — covering both MEGS-A and MEGS-B
        spectral lines.
        """
    )

    st.write("### Key Results")
    st.write(
        """
        - **AIA imagery alone is sufficient** to predict solar irradiance —
          adding HMI magnetogram data does not improve predictions, contrary
          to theoretical expectations.
        - Predicts **38 spectral lines**, nearly **3x** the 14 lines achieved
          by previous state-of-the-art (Szenicer et al. 2019).
        - Achieves similar or better accuracy than prior deep-learning
          approaches, and **outperforms the physics-based DeepEM model**
          (Wright et al. 2019).
        - Predictions for MEGS-B lines can be **cross-validated against live
          measurements**, providing ongoing quality assurance.
        - Strongest predictions occur for ions whose wavelengths are close to
          AIA channel frequencies (e.g., Fe IX at 171.1 Angstrom).
        """
    )

    st.write("### How This Demo Works")
    st.write(
        """
        1. AIA images (9 EUV/UV wavelengths, 512 x 512 px) are loaded from the
           SDOML v2 dataset.
        2. Images are normalized using training-set statistics stored in the
           model checkpoint.
        3. The pre-trained model predicts irradiance for all 38 EVE spectral
           lines — **on the fly, in real time**.
        """
    )

    st.write("### Reference")
    st.write(
        """
        Indaco, M., Gass, D., Fawcett, W. J., Galvez, R., Wright, P. J., &
        Muñoz-Jaramillo, A. (2024). *Virtual EVE: a Deep Learning Model for
        Solar Irradiance Prediction.*
        [arXiv:2408.17430](https://arxiv.org/abs/2408.17430)
        """
    )

    st.write("### Credits")
    st.write(
        """
        Built during **FDL-X 2023** (Frontier Development Lab) by
        Manuel Indaco, Daniel Gass, William Fawcett, Richard Galvez,
        Paul Wright, and Andrés Muñoz-Jaramillo.

        **Data source:** SDOML v2 — [sdoml.org](https://sdoml.org)
        """
    )
    st.image("assets/nasa_sdo.png", width=400)

with tab_eve:
    if not valid:
        st.error("Start date must be before end date.")
    elif st.sidebar.button("Analyze", type="primary"):
        # Get available timestamps in range
        ts_df = get_timestamps_in_range(time_index, start_dt, end_dt)

        if ts_df.empty:
            st.warning("No data available in the selected range.")
            st.stop()

        timestamps = ts_df.index.tolist()
        st.sidebar.info(f"Found {len(timestamps)} timestamps in range.")

        # ── Run inference ────────────────────────────────────────────────
        with st.spinner(f"Running inference on {len(timestamps)} images..."):
            eve_data = predict_eve_timeseries(
                model, aia_root, time_index, aia_norms, wavelengths, eve_ions, timestamps
            )

        if eve_data.empty:
            st.warning("Could not load AIA data for the selected range.")
            st.stop()

        # ── Load AIA image for the first timestamp ───────────────────────
        first_ts = timestamps[0]
        with st.spinner("Loading AIA images..."):
            aia_image = get_aia_image(aia_root, time_index, first_ts)

        # ── Download buttons ─────────────────────────────────────────────
        with st.sidebar:
            st.write("## Download")
            st.download_button(
                "Virtual EVE Predictions (CSV)",
                data=eve_data.to_csv(),
                file_name="virtual_eve.csv",
            )

        # ── Layout ───────────────────────────────────────────────────────
        col1, col2 = st.columns(2, gap="medium")

        # AIA images
        col1.write(f"### AIA Images — {first_ts}")
        ZMAX = {
            "131A": 20, "1600A": 120, "1700A": 1400,
            "171A": 400, "193A": 300, "211A": 150,
            "304A": 80, "335A": 10, "94A": 3,
        }
        if aia_image:
            cols_img = col1.columns(3)
            for i, wl in enumerate(AIA_WAVELENGTHS):
                fig = px.imshow(
                    aia_image[wl], template="seaborn",
                    zmin=0, zmax=ZMAX.get(wl, 100),
                )
                fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                cols_img[i % 3].write(f"**{wl}**")
                cols_img[i % 3].plotly_chart(fig, use_container_width=True)
        else:
            col1.write("No AIA data available for this timestamp.")

        # EVE predictions
        col2.write("### Virtual EVE Irradiance")
        fig_eve = px.line(eve_data, log_y=True, height=1200, template="seaborn")
        fig_eve.update_layout(
            xaxis_title="Time",
            yaxis_title="Irradiance",
            legend_title="Ion",
        )
        col2.plotly_chart(fig_eve, use_container_width=True)

        # Histograms
        if aia_image:
            st.write("### AIA Image Histograms")
            hist_cols = st.columns(3)
            for i, wl in enumerate(AIA_WAVELENGTHS):
                fig = px.histogram(
                    aia_image[wl].flatten(), nbins=500, template="seaborn",
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=wl,
                )
                hist_cols[i % 3].plotly_chart(fig, use_container_width=True)
