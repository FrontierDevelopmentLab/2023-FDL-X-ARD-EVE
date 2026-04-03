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

page = st.selectbox("Select Page", ("Virtual EVE", "About"))

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
    if not valid:
        st.error("Start date must be before end date.")

# ── Pages ────────────────────────────────────────────────────────────────────

if page == "About":
    st.write("## About")
    st.write(
        """
        The **MEGS-A** detector on SDO/EVE measured solar EUV irradiance from
        2010 to 2014, when it suffered a power anomaly.  This model produces
        **Virtual EVE** predictions by running a hybrid Linear + EfficientNet-B5
        CNN on SDO/AIA imagery, effectively extending the MEGS-A record beyond
        2014.

        ### How it works
        1. AIA images (9 EUV/UV wavelengths, 512 x 512 px) are loaded from the
           SDOML v2 dataset on AWS S3.
        2. Images are normalized using training-set statistics.
        3. The model predicts irradiance for 38 EVE spectral lines.

        ### Credits
        Built during **FDL-X 2023** (Frontier Development Lab) by
        William Fawcett, Richard Galvez, Daniel Gass, Manuel Indaco,
        Andrés Muñoz-Jaramillo, and Paul Wright.

        **Data source:** SDOML v2 — `s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/`
        """
    )
    st.image("assets/nasa_sdo.png", width=400)

elif valid:
    if st.sidebar.button("Analyze", type="primary"):
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
