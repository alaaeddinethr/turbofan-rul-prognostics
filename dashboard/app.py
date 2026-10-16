"""Engine health dashboard: pick a test engine, see its sensors and predicted RUL.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rul import data as D  # noqa: E402
from src.rul.features import windows  # noqa: E402
from src.rul.models import XGBoostRUL  # noqa: E402

st.set_page_config(page_title="Turbofan RUL", layout="wide")


@st.cache_resource
def load_model(dataset: str):
    ds = D.load(dataset)
    tr, va = D.split_units(ds.train)
    x, y, _ = windows(tr, ds.features)
    xv, yv, _ = windows(va, ds.features)
    return ds, XGBoostRUL().fit(x, y, xv, yv)


def status(rul: float) -> tuple[str, str]:
    if rul < 30:
        return "Schedule maintenance now", "#d1242f"
    if rul < 60:
        return "Plan maintenance at next slot", "#bf8700"
    return "Healthy", "#1a7f37"


st.title("Turbofan engine — remaining useful life")
st.caption("NASA C-MAPSS test engines · XGBoost with 80 % prediction interval")

dataset = st.sidebar.selectbox("Sub-dataset", ["FD001", "FD003"])
try:
    ds, model = load_model(dataset)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

unit = st.sidebar.selectbox("Engine", sorted(ds.test["unit"].unique()))
g = ds.test[ds.test["unit"] == unit]
x, y_true, _ = windows(g, ds.features)
m, lo, hi = model.predict(x)
cycles = g["cycle"].to_numpy()[-len(m):]

label, color = status(m[-1])
c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", f"{m[-1]:.0f}", f"80 % interval {lo[-1]:.0f}–{hi[-1]:.0f}", delta_color="off")
c2.metric("True RUL (cycles)", f"{y_true[-1]:.0f}")
c3.markdown(f"<div style='padding-top:1.6rem;font-size:1.2rem;color:{color};font-weight:600'>{label}</div>",
            unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(cycles) + list(cycles[::-1]), y=list(hi) + list(lo[::-1]),
                         fill="toself", fillcolor="rgba(31,111,235,0.15)", line=dict(width=0),
                         name="80 % interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=cycles, y=y_true, name="True RUL", line=dict(color="black", width=2)))
fig.add_trace(go.Scatter(x=cycles, y=m, name="Predicted RUL", line=dict(color="#1f6feb")))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Cycle",
                  yaxis_title="RUL (cycles)", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")

st.subheader("Sensor readings (scaled)")
chosen = st.multiselect("Sensors", ds.features, default=ds.features[:4])
if chosen:
    st.line_chart(pd.DataFrame(g[chosen].to_numpy(), index=g["cycle"], columns=chosen))
