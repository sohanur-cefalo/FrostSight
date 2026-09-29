"""FrostSight local prototype dashboard (Plotly Dash).

Reads the PyTorch-scored segment table from prototype/risk.py and shows four
views in one app, mirroring the four dashboards in the original plan (risk map,
road detail, gritting priority list, platform health) at prototype scale.

Run: python -m prototype.app
"""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html

from prototype.risk import score_segments

RISK_COLORS = {
    "LOW": "#2E7D32",       # good
    "MEDIUM": "#F2B705",    # warning
    "HIGH": "#E8590C",      # serious
    "VERY_HIGH": "#C1272D", # critical
}
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]

df = score_segments()
obs = pd.read_csv("prototype/artifacts/observations.csv", parse_dates=["event_time"])

app = Dash(__name__)
app.title = "FrostSight prototype"

_latest_event_time = df["event_time"].max()
if _latest_event_time.tzinfo is None:
    _latest_event_time = _latest_event_time.tz_localize("UTC")
freshness_seconds = (pd.Timestamp.now(tz="UTC") - _latest_event_time).total_seconds()

header = html.Div(
    [
        html.H2("FrostSight — icing risk prototype (Troms pilot county)"),
        html.P(
            "Synthetic data, PyTorch risk model. Not an official warning service — "
            f"latest observation {df['event_time'].max():%Y-%m-%d %H:%M UTC}.",
            style={"color": "#666"},
        ),
    ]
)


def risk_map_tab():
    # Plain lon/lat scatter, not scattermapbox: no basemap tiles to fetch, renders offline.
    fig = {
        "data": [
            {
                "type": "scatter",
                "x": df["lon"],
                "y": df["lat"],
                "mode": "markers",
                "marker": {
                    "size": 16,
                    "color": [RISK_COLORS[lvl] for lvl in df["risk_level"]],
                    "line": {"width": 1, "color": "white"},
                },
                "text": df["road_segment_id"] + " — " + df["risk_level"],
                "hovertemplate": "%{text}<br>score=%{customdata:.2f}<extra></extra>",
                "customdata": df["ml_risk_score"],
            }
        ],
        "layout": {
            "xaxis": {"title": "longitude", "zeroline": False},
            "yaxis": {"title": "latitude", "zeroline": False, "scaleanchor": "x"},
            "margin": {"l": 60, "r": 20, "t": 10, "b": 40},
            "height": 520,
            "plot_bgcolor": "#eef2f6",
        },
    }
    legend = html.Div(
        [
            html.Span(
                f" {lvl} ",
                style={
                    "backgroundColor": RISK_COLORS[lvl],
                    "color": "white",
                    "borderRadius": "4px",
                    "padding": "2px 8px",
                    "marginRight": "8px",
                },
            )
            for lvl in RISK_ORDER
        ],
        style={"marginBottom": "8px"},
    )
    return html.Div([legend, dcc.Graph(figure=fig, config={"displayModeBar": False})])


def road_detail_tab():
    options = [{"label": r, "value": r} for r in df["road_segment_id"]]
    return html.Div(
        [
            dcc.Dropdown(id="segment-picker", options=options, value=df.iloc[0]["road_segment_id"]),
            html.Div(id="segment-summary", style={"margin": "12px 0"}),
            dcc.Graph(id="segment-chart"),
        ]
    )


def priority_list_tab():
    table_df = df[["road_segment_id", "road_name", "risk_level", "ml_risk_score", "drivers"]].copy()
    table_df["ml_risk_score"] = table_df["ml_risk_score"].round(2)
    return dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        style_data_conditional=[
            {
                "if": {"filter_query": f'{{risk_level}} = "{lvl}"'},
                "backgroundColor": color,
                "color": "white",
            }
            for lvl, color in RISK_COLORS.items()
        ],
        sort_action="native",
        page_size=15,
        style_cell={"fontFamily": "sans-serif", "fontSize": 13, "padding": "6px"},
    )


def platform_health_tab():
    fresh_status = "good" if freshness_seconds < 900 else "warning" if freshness_seconds < 3600 else "critical"
    fresh_color = {"good": "#2E7D32", "warning": "#F2B705", "critical": "#C1272D"}[fresh_status]
    counts = df["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig = {
        "data": [
            {
                "type": "bar",
                "x": RISK_ORDER,
                "y": counts.values,
                "marker": {"color": [RISK_COLORS[l] for l in RISK_ORDER]},
            }
        ],
        "layout": {
            "yaxis": {"title": "segments"},
            "margin": {"t": 20},
            "height": 320,
        },
    }
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Data freshness: ", style={"fontWeight": "bold"}),
                    html.Span(
                        f"{freshness_seconds / 60:.0f} min old",
                        style={"color": fresh_color, "fontWeight": "bold"},
                    ),
                ]
            ),
            html.Div(f"Segments scored: {len(df)}", style={"margin": "8px 0"}),
            html.Div("Sources (synthetic prototype): road_weather (mock DATEX II)", style={"color": "#666"}),
            html.H4("Segments by risk level"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]
    )


app.layout = html.Div(
    [
        header,
        dcc.Tabs(
            [
                dcc.Tab(label="Risk map", children=[risk_map_tab()]),
                dcc.Tab(label="Road detail", children=[road_detail_tab()]),
                dcc.Tab(label="Gritting priority list", children=[priority_list_tab()]),
                dcc.Tab(label="Platform health", children=[platform_health_tab()]),
            ]
        ),
    ],
    style={"fontFamily": "sans-serif", "padding": "16px"},
)


@app.callback(
    Output("segment-summary", "children"),
    Output("segment-chart", "figure"),
    Input("segment-picker", "value"),
)
def update_segment(segment_id: str):
    row = df[df["road_segment_id"] == segment_id].iloc[0]
    summary = html.Div(
        [
            html.Span(
                row["risk_level"],
                style={
                    "backgroundColor": RISK_COLORS[row["risk_level"]],
                    "color": "white",
                    "borderRadius": "4px",
                    "padding": "2px 10px",
                    "marginRight": "10px",
                },
            ),
            html.Span(f"score={row['ml_risk_score']:.2f}  —  drivers: {row['drivers']}"),
        ]
    )
    seg_obs = obs[obs["road_segment_id"] == segment_id].sort_values("event_time")
    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": seg_obs["event_time"],
                "y": seg_obs["surface_temp_c"],
                "name": "surface temp (°C)",
                "line": {"color": "#1565C0", "width": 2},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": seg_obs["event_time"],
                "y": [0] * len(seg_obs),
                "name": "freezing point",
                "line": {"color": "#999", "width": 1, "dash": "dot"},
            },
        ],
        "layout": {"yaxis": {"title": "°C"}, "height": 360, "legend": {"orientation": "h"}},
    }
    return summary, fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
