"""FrostSight local prototype dashboard (Plotly Dash).

Reads the PyTorch-scored segment table from prototype/risk.py and shows four
views in one app, mirroring the four dashboards in the original plan (risk map,
road detail, gritting priority list, platform health) at prototype scale.

Run: python -m prototype.app
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
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

# The synthetic segments are generated in order along the Tromsø -> Bardufoss corridor
# (generate_data.py), so sorting by id recovers the physical road path.
route_df = df.sort_values("road_segment_id").reset_index(drop=True)

MAP_ASPECT = 2.1  # width:height target so the full-screen map reads as a landscape panel


def _wide_map_bounds() -> tuple[list[float], list[float]]:
    """Lon/lat range padded so the map fills a wide panel with real surrounding
    coastline instead of a narrow strip hugging just the road corridor."""
    lat_min, lat_max = route_df["lat"].min(), route_df["lat"].max()
    lon_min, lon_max = route_df["lon"].min(), route_df["lon"].max()
    lat_span = (lat_max - lat_min) * 1.3
    lat_mid = (lat_max + lat_min) / 2
    lon_span_needed = MAP_ASPECT * lat_span / math.cos(math.radians(lat_mid))
    lon_span = max((lon_max - lon_min) * 1.3, lon_span_needed)
    lon_mid = (lon_max + lon_min) / 2
    return (
        [lon_mid - lon_span / 2, lon_mid + lon_span / 2],
        [lat_mid - lat_span / 2, lat_mid + lat_span / 2],
    )

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


def build_map_figure(vehicle_step: int | None = None) -> go.Figure:
    """Norway/Troms basemap (Plotly's bundled coastline vectors, no tile server needed),
    the pilot-county road drawn as a path coloured per segment by risk, and an optional
    simulated gritting-truck position moving along that path in real time.
    """
    fig = go.Figure()

    # Road path: one line trace per segment-to-segment hop, coloured by that hop's risk,
    # so the whole corridor reads like the priority list laid over the map.
    for i in range(len(route_df) - 1):
        a, b = route_df.iloc[i], route_df.iloc[i + 1]
        fig.add_trace(
            go.Scattergeo(
                lon=[a["lon"], b["lon"]],
                lat=[a["lat"], b["lat"]],
                mode="lines",
                line={"width": 5, "color": RISK_COLORS[a["risk_level"]]},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Traveled portion of the route so far, drawn as a thick outline on top of the path.
    if vehicle_step:
        traveled = route_df.iloc[: vehicle_step + 1]
        fig.add_trace(
            go.Scattergeo(
                lon=traveled["lon"],
                lat=traveled["lat"],
                mode="lines",
                line={"width": 2, "color": "#12233a", "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lon=route_df["lon"],
            lat=route_df["lat"],
            mode="markers",
            marker={
                "size": 11,
                "color": [RISK_COLORS[lvl] for lvl in route_df["risk_level"]],
                "line": {"width": 1, "color": "white"},
            },
            text=route_df["road_segment_id"] + " — " + route_df["risk_level"],
            customdata=route_df["ml_risk_score"],
            hovertemplate="%{text}<br>score=%{customdata:.2f}<extra></extra>",
            name="Segments",
            showlegend=False,
        )
    )

    if vehicle_step is not None:
        pos = route_df.iloc[vehicle_step % len(route_df)]
        fig.add_trace(
            go.Scattergeo(
                lon=[pos["lon"]],
                lat=[pos["lat"]],
                mode="markers",
                marker={"size": 18, "color": "#1565C0", "symbol": "triangle-up", "line": {"width": 2, "color": "white"}},
                text=[f"Gritting unit — near {pos['road_segment_id']}"],
                hovertemplate="%{text}<extra></extra>",
                name="Gritting unit (simulated GPS)",
                showlegend=False,
            )
        )

    lon_range, lat_range = _wide_map_bounds()
    fig.update_geos(
        scope="europe",
        resolution=50,
        projection_type="mercator",
        lonaxis_range=lon_range,
        lataxis_range=lat_range,
        showland=True,
        landcolor="#eef2f6",
        showocean=True,
        oceancolor="#dbe7f2",
        showlakes=True,
        lakecolor="#dbe7f2",
        showcountries=True,
        countrycolor="#9aa5b1",
        showsubunits=True,
        subunitcolor="#c3ccd6",
        showframe=False,
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, autosize=True)
    return fig


def _stat_tile(label: str, value: str):
    return html.Div(
        [
            html.P(label, style={"fontSize": "13px", "color": "#6a7179", "margin": "0 0 4px"}),
            html.P(value, style={"fontSize": "26px", "fontWeight": "600", "margin": 0, "color": "#12233a"}),
        ],
        style={"background": "#f4f6f8", "borderRadius": "10px", "padding": "12px 16px", "marginBottom": "12px"},
    )


def risk_map_tab():
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
        ]
        + [
            html.Span(
                " ▲ gritting unit (simulated GPS, moves every 2s) ",
                style={"color": "#1565C0", "marginLeft": "12px", "fontWeight": "bold"},
            )
        ],
        style={"marginBottom": "12px"},
    )
    n_high = int((df["risk_level"].isin(["HIGH", "VERY_HIGH"])).sum())
    stats = html.Div(
        [
            _stat_tile("Segments HIGH or above", f"{n_high} of {len(df)}"),
            _stat_tile("Stations reporting", f"{len(df)} of {len(df)}"),
            _stat_tile("Data age", f"{freshness_seconds / 60:.0f} min"),
        ],
        style={"display": "flex", "gap": "12px", "flex": "0 0 auto"},
    )
    top5 = df.sort_values("ml_risk_score", ascending=False).head(5)
    watchlist = html.Div(
        [html.Span("Top risk now: ", style={"fontSize": "13px", "color": "#6a7179", "marginRight": "6px"})]
        + [
            html.Span(
                [
                    html.Span(
                        row["risk_level"],
                        style={
                            "backgroundColor": RISK_COLORS[row["risk_level"]],
                            "color": "white",
                            "borderRadius": "4px",
                            "padding": "1px 6px",
                            "fontSize": "11px",
                            "marginRight": "4px",
                        },
                    ),
                    html.Span(f"{row['road_segment_id']}", style={"fontSize": "13px", "marginRight": "12px"}),
                ]
            )
            for _, row in top5.iterrows()
        ],
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "flex": "1 1 300px"},
    )
    return html.Div(
        [
            html.Div(
                [stats, watchlist],
                style={
                    "display": "flex",
                    "gap": "24px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "12px",
                },
            ),
            legend,
            html.Div(
                dcc.Graph(
                    id="risk-map-graph",
                    figure=build_map_figure(0),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "78vh", "width": "100%"},
                ),
                style={
                    "background": "#fbfcfd",
                    "border": "1px solid #d8dee6",
                    "borderRadius": "12px",
                    "padding": "8px",
                },
            ),
            dcc.Interval(id="vehicle-interval", interval=2000, n_intervals=0),
        ]
    )


@app.callback(
    Output("risk-map-graph", "figure"),
    Input("vehicle-interval", "n_intervals"),
)
def update_vehicle_position(n_intervals: int):
    return build_map_figure(n_intervals % len(route_df))


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


TAB_STYLE = {"padding": "10px 4px", "border": "none", "borderBottom": "3px solid transparent"}
TAB_SELECTED_STYLE = {
    "padding": "10px 4px",
    "border": "none",
    "borderBottom": "3px solid #1565C0",
    "color": "#1565C0",
    "fontWeight": "600",
}

app.layout = html.Div(
    html.Div(
        [
            header,
            dcc.Tabs(
                [
                    dcc.Tab(label="Risk map", children=[risk_map_tab()], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(
                        label="Road detail", children=[road_detail_tab()], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE
                    ),
                    dcc.Tab(
                        label="Gritting priority list",
                        children=[priority_list_tab()],
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                    ),
                    dcc.Tab(
                        label="Platform health",
                        children=[platform_health_tab()],
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
        ],
        style={
            "maxWidth": "1400px",
            "margin": "0 auto",
            "background": "white",
            "borderRadius": "16px",
            "padding": "28px 32px",
            "boxShadow": "0 1px 3px rgba(18,35,58,0.08)",
        },
    ),
    style={"fontFamily": "'IBM Plex Sans', Arial, sans-serif", "padding": "32px 16px", "background": "#eef1f5", "minHeight": "100vh"},
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
