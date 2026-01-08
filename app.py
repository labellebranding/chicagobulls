import re
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Bulls Fan Belief Intelligence", layout="wide")

# -----------------------------
# Bulls-style theme (red + black)
# INSERT THE CSS BLOCK HERE
# -----------------------------
BULLS_RED = "#CE1141"
BULLS_BLACK = "#0B0B0B"
BULLS_DARK = "#111214"
BULLS_BORDER = "rgba(255,255,255,0.10)"
BULLS_TEXT = "rgba(255,255,255,0.92)"
BULLS_MUTED = "rgba(255,255,255,0.70)"

st.markdown(
    f"""<style>
    ...the full CSS block...
    </style>""",
    unsafe_allow_html=True,
)

# THEN everything else:
# KPI definitions, dictionaries, helpers, st.title(), sidebar, tabs, etc.
