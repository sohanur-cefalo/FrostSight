"""FrostSight local prototype dashboard (Plotly Dash).

Reads the PyTorch-scored segment table from prototype/risk.py and shows four
views in one app, mirroring the four dashboards in the original plan (risk map,
road detail, gritting priority list, platform health) at prototype scale.

Run: python -m prototype.app
"""

from __future__ import annotations

import math

import numpy as np
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

# Longitudinal degree span covered by one Mapbox/OSM tile row at each integer zoom
# level (0=world ... 20=building). Same table wsd-dashboard's wirescan.geometry.zoom_center
# uses to turn a lon/lat bounding box into a zoom level for Scattermapbox.
_ZOOM_LON_SPANS = [
    360, 180, 90, 45, 22.5, 11.25, 5.625, 2.813, 1.406, 0.703,
    0.352, 0.176, 0.088, 0.044, 0.022, 0.011, 0.005, 0.003, 0.001, 0.0007, 0.0003,
]


def _zoom_center(lons, lats) -> tuple[float, dict[str, float]]:
    """Fit a Mapbox zoom level + center to a set of coordinates, padded so the
    road reads as a landscape panel rather than a narrow strip. This is the
    Scattermapbox analogue of wsd-dashboard's plot_cables() -> zoom_center()."""
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    # Tighter padding (was 1.3x) so the corridor fills the panel instead of
    # leaving a wide margin of unrelated coastline around it.
    lat_span = max((lat_max - lat_min) * 1.08, 0.01)
    lat_mid = (lat_max + lat_min) / 2
    lon_span_needed = MAP_ASPECT * lat_span / math.cos(math.radians(lat_mid))
    lon_span = max((lon_max - lon_min) * 1.08, lon_span_needed, 0.01)
    lon_mid = (lon_max + lon_min) / 2

    zoom_levels = list(range(len(_ZOOM_LON_SPANS)))
    lon_zoom = np.interp(lon_span, _ZOOM_LON_SPANS[::-1], zoom_levels[::-1])
    lat_zoom = np.interp(lat_span / MAP_ASPECT, _ZOOM_LON_SPANS[::-1], zoom_levels[::-1])
    # +0.5 extra zoom-in on top of the strict best fit, since the best-fit level
    # rounds down and reads as noticeably too wide/zoomed-out otherwise.
    zoom = round(min(lon_zoom, lat_zoom) + 0.5, 2)
    return zoom, {"lon": lon_mid, "lat": lat_mid}

app = Dash(__name__)
app.title = "FrostSight prototype"
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { margin: 0; font-family: 'IBM Plex Sans', Arial, sans-serif; }
            ::-webkit-scrollbar { width: 10px; height: 10px; }
            ::-webkit-scrollbar-thumb { background: #c3ccd6; border-radius: 6px; }
        </style>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""

_latest_event_time = df["event_time"].max()
if _latest_event_time.tzinfo is None:
    _latest_event_time = _latest_event_time.tz_localize("UTC")
freshness_seconds = (pd.Timestamp.now(tz="UTC") - _latest_event_time).total_seconds()

NAVY = "#0d1b2e"

topbar = html.Div(
    html.Div(
        [
            html.Div(
                [
                    html.Span("FROSTSIGHT", style={"fontWeight": "700", "fontSize": "18px", "letterSpacing": "1px"}),
                    html.Span(
                        "TROMS PILOT · PROTOTYPE",
                        style={
                            "fontSize": "11px",
                            "color": "#93a4bd",
                            "letterSpacing": "1.5px",
                            "marginLeft": "12px",
                            "border": "1px solid #33455e",
                            "borderRadius": "4px",
                            "padding": "2px 8px",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Span("● ", style={"color": "#3ddc84"}),
                    html.Span(
                        f"Synthetic data · PyTorch risk model · observed {df['event_time'].max():%H:%M UTC}",
                        style={"color": "#c4cedd", "fontSize": "13px"},
                    ),
                ],
            ),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "maxWidth": "1400px",
            "margin": "0 auto",
            "padding": "16px 24px",
        },
    ),
    style={"background": NAVY, "color": "white"},
)


def load_route_df() -> pd.DataFrame:
    """Re-run the risk model over the current observation data and rebuild the
    ordered road path. Called once at startup and again on every map reload,
    the way wsd-dashboard's overview re-fetches cables/geometries on each
    on-load interval / refresh click instead of caching a stale figure."""
    fresh_df = score_segments()
    return fresh_df.sort_values("road_segment_id").reset_index(drop=True)


class RoadMapView:
    """Builds the Troms road map figure trace-by-trace, mirroring wsd-dashboard's
    MapView: each plot_* call appends Scattermapbox trace(s) on real map tiles
    (no Mapbox token required — the free "open-street-map" style, same idea as
    wsd's mapbox-styled Scattermapbox) and accumulates the lon/lat extent, then
    fit_bounds() zooms/centers the basemap to what was actually plotted instead
    of a hardcoded viewport."""

    def __init__(self) -> None:
        self.fig = go.Figure()
        self.all_lons: list[float] = []
        self.all_lats: list[float] = []

    def _track_extent(self, lons, lats) -> None:
        self.all_lons.extend(lons)
        self.all_lats.extend(lats)

    # A same-route hop longer than this is almost certainly two separate, far-apart
    # stretches of the same numbered road (e.g. E6 near Tromsø vs. E6 100 km south),
    # not a real adjacent connection — skip drawing a line for it.
    _MAX_HOP_KM = 15.0

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def plot_roads(self, route_df: pd.DataFrame) -> None:
        """One line trace per segment-to-segment hop, coloured by that hop's risk —
        the road-network analogue of MapView.plot_cables(). Real stations
        (docs/adr/0006-real-data-integration.md Phase 1) sit on different physical
        roads and even the same numbered route can jump between far-apart stretches,
        so a hop is only drawn when both ends share the same route_code AND are
        within _MAX_HOP_KM of each other — otherwise it would fake a road connection
        that doesn't exist. Segments without a route_code (older synthetic data)
        always connect, matching the original single-corridor behaviour."""
        has_route_code = "route_code" in route_df.columns
        for i in range(len(route_df) - 1):
            a, b = route_df.iloc[i], route_df.iloc[i + 1]
            if has_route_code:
                same_route = a["route_code"] == b["route_code"]
                close_enough = self._haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) <= self._MAX_HOP_KM
                if not (same_route and close_enough):
                    continue
            self.fig.add_trace(
                go.Scattermap(
                    lon=[a["lon"], b["lon"]],
                    lat=[a["lat"], b["lat"]],
                    mode="lines",
                    line={"width": 5, "color": RISK_COLORS[a["risk_level"]]},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        self._track_extent(route_df["lon"], route_df["lat"])

    def plot_traveled(self, route_df: pd.DataFrame, vehicle_step: int) -> None:
        """Thick dotted outline over the portion of the route already covered."""
        if not vehicle_step:
            return
        traveled = route_df.iloc[: vehicle_step + 1]
        self.fig.add_trace(
            go.Scattermap(
                lon=traveled["lon"],
                lat=traveled["lat"],
                mode="lines",
                line={"width": 3, "color": "#12233a"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def plot_segment_markers(self, route_df: pd.DataFrame) -> None:
        self.fig.add_trace(
            go.Scattermap(
                lon=route_df["lon"],
                lat=route_df["lat"],
                mode="markers",
                marker={
                    "size": 11,
                    "color": [RISK_COLORS[lvl] for lvl in route_df["risk_level"]],
                },
                text=route_df["road_segment_id"] + " — " + route_df["risk_level"],
                customdata=route_df["ml_risk_score"],
                hovertemplate="%{text}<br>score=%{customdata:.2f}<extra></extra>",
                name="Segments",
                showlegend=False,
            )
        )

    def plot_vehicle(self, route_df: pd.DataFrame, vehicle_step: int) -> None:
        pos = route_df.iloc[vehicle_step % len(route_df)]
        self.fig.add_trace(
            go.Scattermap(
                lon=[pos["lon"]],
                lat=[pos["lat"]],
                mode="markers",
                marker={"size": 16, "color": "#1565C0"},
                text=[f"Gritting unit — near {pos['road_segment_id']}"],
                hovertemplate="%{text}<extra></extra>",
                name="Gritting unit (simulated GPS)",
                showlegend=False,
            )
        )

    def plot_condition_labels(self, route_df: pd.DataFrame, top_n: int = 3) -> None:
        """Inline condition callouts on the worst segments, the way operational
        road-weather products label map regions directly instead of a side legend."""
        worst = route_df.sort_values("ml_risk_score", ascending=False).head(top_n)
        self.fig.add_trace(
            go.Scattermap(
                lon=worst["lon"],
                lat=worst["lat"],
                mode="text",
                text=[f"{r.risk_level} · {r.surface_temp_c:.1f}°C" for r in worst.itertuples()],
                textfont={"size": 12, "color": "#12233a", "family": "IBM Plex Sans, Arial"},
                textposition="top right",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def fit_bounds(self) -> None:
        zoom, center = _zoom_center(self.all_lons, self.all_lats)
        self.fig.update_layout(
            map={"style": "open-street-map", "zoom": zoom, "center": center},
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            autosize=True,
        )


def build_map_figure(route_df: pd.DataFrame, vehicle_step: int | None = None) -> go.Figure:
    """Real OpenStreetMap tiles (Scattermapbox, no Mapbox token needed), the
    pilot-county road drawn as a path coloured per segment by risk, and an
    optional simulated gritting-truck position moving along that path in
    real time — the same rendering approach as wsd-dashboard's cable map.
    """
    map_view = RoadMapView()
    map_view.plot_roads(route_df)
    map_view.plot_traveled(route_df, vehicle_step or 0)
    map_view.plot_segment_markers(route_df)
    if vehicle_step is not None:
        map_view.plot_vehicle(route_df, vehicle_step)
    map_view.plot_condition_labels(route_df)
    map_view.fit_bounds()
    return map_view.fig


def _stat_tile(label: str, value: str, accent: str = "#1565C0"):
    return html.Div(
        [
            html.P(
                label,
                style={
                    "fontSize": "11px",
                    "color": "#6a7179",
                    "margin": "0 0 6px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.6px",
                },
            ),
            html.P(value, style={"fontSize": "24px", "fontWeight": "600", "margin": 0, "color": "#12233a"}),
        ],
        style={
            "background": "#f8f9fb",
            "borderLeft": f"3px solid {accent}",
            "borderRadius": "6px",
            "padding": "10px 16px",
        },
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
            _stat_tile("Segments HIGH or above", f"{n_high} of {len(df)}", "#E8590C"),
            _stat_tile("Stations reporting", f"{len(df)} of {len(df)}", "#2E7D32"),
            _stat_tile("Data age", f"{freshness_seconds / 60:.0f} min", "#1565C0"),
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
                [
                    html.Div(
                        [
                            html.Span(
                                "MAP",
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "700",
                                    "letterSpacing": "0.6px",
                                    "color": "#6a7179",
                                    "textTransform": "uppercase",
                                },
                            ),
                            html.Button(
                                "⟳ Refresh",
                                id="map-refresh-button",
                                n_clicks=0,
                                style={
                                    "border": "1px solid #d8dee6",
                                    "borderRadius": "6px",
                                    "background": "white",
                                    "color": NAVY,
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "padding": "4px 10px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "8px",
                        },
                    ),
                    dcc.Loading(
                        id="risk-map-loading",
                        type="circle",
                        color=NAVY,
                        children=dcc.Graph(
                            id="risk-map-graph",
                            figure=build_map_figure(route_df, 0),
                            config={"displayModeBar": False, "responsive": True},
                            style={"height": "78vh", "width": "100%"},
                        ),
                    ),
                ],
                style={
                    "background": "#fbfcfd",
                    "border": "1px solid #d8dee6",
                    "borderRadius": "12px",
                    "padding": "8px",
                },
            ),
            # Reloads road data from source on a schedule, the way wsd-dashboard's
            # overview-on-load-interval re-fetches cables/geometries hourly.
            dcc.Interval(id="map-reload-interval", interval=1000 * 3600, n_intervals=0),
            dcc.Interval(id="vehicle-interval", interval=2000, n_intervals=0),
        ]
    )


@app.callback(
    Output("risk-map-graph", "figure", allow_duplicate=True),
    Input("map-reload-interval", "n_intervals"),
    Input("map-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def on_map_reload(_n_intervals, _n_clicks):
    """Re-fetch/re-score road data and rebuild the map — the road-map analogue
    of wsd-dashboard's on_page_load_interval / on_overview_map_refresh_button_click."""
    global route_df
    route_df = load_route_df()
    return build_map_figure(route_df, 0)


@app.callback(
    Output("risk-map-graph", "figure"),
    Input("vehicle-interval", "n_intervals"),
)
def update_vehicle_position(n_intervals: int):
    return build_map_figure(route_df, n_intervals % len(route_df))


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


TAB_STYLE = {
    "padding": "12px 4px",
    "border": "none",
    "borderBottom": "3px solid transparent",
    "fontSize": "13px",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "color": "#6a7179",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "borderBottom": f"3px solid {NAVY}",
    "color": NAVY,
    "fontWeight": "700",
}

app.layout = html.Div(
    [
        topbar,
        html.Div(
            html.Div(
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Risk map", children=[risk_map_tab()], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE
                        ),
                        dcc.Tab(
                            label="Road detail",
                            children=[road_detail_tab()],
                            style=TAB_STYLE,
                            selected_style=TAB_SELECTED_STYLE,
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
                    style={"marginBottom": "20px"},
                ),
                style={
                    "maxWidth": "1400px",
                    "margin": "0 auto",
                    "background": "white",
                    "borderRadius": "12px",
                    "padding": "24px 32px",
                    "boxShadow": "0 1px 3px rgba(18,35,58,0.08)",
                },
            ),
            style={"padding": "24px 16px"},
        ),
    ],
    style={"fontFamily": "'IBM Plex Sans', Arial, sans-serif", "background": "#eef1f5", "minHeight": "100vh"},
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
