from pathlib import Path
import base64
import json
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
os.environ["PROJ_LIB"] = str(Path(rasterio.__file__).resolve().parent / "proj_data")
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RioResampling
from rasterio.plot import reshape_as_image
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from io import BytesIO

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Hyderabad UGS–UHI WebGIS",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
if not (ROOT / "json_file").exists() and (ROOT.parent / "json_file").exists():
    ROOT = ROOT.parent


class FloatingLayerControl(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        var floatingLayersControl = L.control({position: 'topright'});
        floatingLayersControl.onAdd = function(map) {
            var container = L.DomUtil.create('div', 'floating-layers-control');
            var button = L.DomUtil.create('button', 'floating-layers-button', container);
            button.type = 'button';
            button.textContent = 'Layers';
            button.setAttribute('aria-expanded', 'false');

            var panel = L.DomUtil.create('div', 'floating-layers-panel', container);
            panel.setAttribute('aria-label', 'Map layers');

            button.addEventListener('click', function(event) {
                L.DomEvent.stop(event);
                var isOpen = panel.classList.toggle('is-open');
                button.setAttribute('aria-expanded', String(isOpen));
            });

            L.DomEvent.disableClickPropagation(container);
            L.DomEvent.disableScrollPropagation(container);

            {% for layer, label in this.layers %}
            (function(layer, label) {
                var row = L.DomUtil.create('label', 'floating-layers-option', panel);
                var checkbox = L.DomUtil.create('input', '', row);
                checkbox.type = 'checkbox';
                checkbox.checked = map.hasLayer(layer);
                checkbox.addEventListener('change', function() {
                    if (checkbox.checked) {
                        map.addLayer(layer);
                    } else {
                        map.removeLayer(layer);
                    }
                });
                var text = L.DomUtil.create('span', '', row);
                text.textContent = label;
            })({{ layer.get_name() }}, {{ label|tojson }});
            {% endfor %}

            return container;
        };
        floatingLayersControl.addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """)

    def __init__(self, layers):
        super().__init__()
        self._name = "FloatingLayerControl"
        self.layers = layers


# Paths match the folder structure shown in the project Explorer.
DATA_CANDIDATES = {
    "boundary": [
        ROOT / "json_file" / "hydrabad_boundary_shy.geojson",
        ROOT / "json_file" / "hyderabad_boundary_shy.geojson",
    ],
    "wards": [
        ROOT / "json_file" / "word_json.geojson",
        ROOT / "json_file" / "ward_json.geojson",
    ],
    "green_points": [
        ROOT / "json_file" / "green_point_json.geojson",
    ],
    "raster": [
        ROOT / "Raster1" / "clip_lst2.tif",
    ],
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_files(folder, suffixes):
    """Find data files recursively under a project folder."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def read_vector(path):
    """Read GeoJSON or vector data and return a WGS84 GeoDataFrame."""
    if path is None:
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return gdf
        if gdf.crs is None:
            # GeoJSON normally uses WGS84. For files without CRS metadata,
            # assume WGS84 only for GeoJSON.
            if path.suffix.lower() in (".geojson", ".json"):
                gdf = gdf.set_crs("EPSG:4326")
            else:
                raise ValueError("Coordinate reference system is missing.")
        return gdf.to_crs("EPSG:4326")
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def safe_bounds(gdf):
    if gdf is None or gdf.empty:
        return None
    minx, miny, maxx, maxy = gdf.total_bounds
    return [[miny, minx], [maxy, maxx]]


def add_gdf_layer(map_obj, gdf, layer_name, color, fill_opacity=0.35, marker=None):
    if gdf is None or gdf.empty:
        return
    folium.GeoJson(
        data=json.loads(gdf.to_json(default=str)),
        name=layer_name,
        marker=marker,
        style_function=lambda feature: {
            "color": color,
            "weight": 1.5,
            "fillColor": color,
            "fillOpacity": fill_opacity,
        },
        highlight_function=lambda feature: {
            "weight": 3,
            "color": "#ffffff",
            "fillOpacity": min(fill_opacity + 0.2, 0.8),
        },
        tooltip=folium.GeoJsonTooltip(
            fields=list(gdf.columns.drop("geometry"))[:5],
            aliases=[str(c) for c in list(gdf.columns.drop("geometry"))[:5]],
            localize=True,
            sticky=False,
        ) if len(gdf.columns.drop("geometry")) else None,
    ).add_to(map_obj)


def raster_summary_and_overlay(path):
    """
    Read the raster, calculate valid-pixel statistics in its stored units,
    and prepare a small WGS84 PNG overlay for Folium.
    Does not assume a Celsius scale factor.
    """
    if path is None:
        return None, None, None, None

    with rasterio.open(path) as src:
        band = src.read(1, masked=True).astype("float64")
        valid = band.compressed()
        if valid.size == 0:
            raise ValueError("Raster contains no valid pixels.")

        stats = {
            "mean": float(np.mean(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "valid_pixels": int(valid.size),
            "crs": str(src.crs),
            "nodata": src.nodata,
        }

        if src.crs is None:
            raise ValueError("Raster CRS is missing.")

        # Reproject a display copy to geographic coordinates.
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            dst_width=min(src.width, 1200),
            dst_height=min(src.height, 1200),
        )
        destination = np.full((height, width), np.nan, dtype="float32")
        source = src.read(1).astype("float32")
        if src.nodata is not None:
            source[source == src.nodata] = np.nan

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        finite = destination[np.isfinite(destination)]
        if finite.size == 0:
            return stats, None, None, None

        low, high = np.nanpercentile(finite, [2, 98])
        if low == high:
            high = low + 1
        normalized = np.clip((destination - low) / (high - low), 0, 1)
        rgba = matplotlib.colormaps["inferno"](normalized, bytes=True)
        rgba[~np.isfinite(destination), 3] = 0

        image = Image.fromarray(rgba, mode="RGBA")
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        west = transform.c
        north = transform.f
        east = west + transform.a * width
        south = north + transform.e * height
        image_bounds = [[south, west], [north, east]]

        return stats, buffer.getvalue(), image_bounds, (low, high)


# ------------------------------------------------------------
# LOAD PROJECT DATA
# ------------------------------------------------------------
boundary_path = first_existing(DATA_CANDIDATES["boundary"])
wards_path = first_existing(DATA_CANDIDATES["wards"])
green_points_path = first_existing(DATA_CANDIDATES["green_points"])
raster_path = first_existing(DATA_CANDIDATES["raster"])

# Additional vector files in the project folders.
green_folder_files = find_files(ROOT / "Green_space", {".geojson", ".json", ".shp", ".gpkg"})
green_point_folder_files = find_files(ROOT / "green_point", {".geojson", ".json", ".shp", ".gpkg"})
ward_folder_files = find_files(ROOT / "hydrabad_shp", {".geojson", ".json", ".shp", ".gpkg"})

# Prefer the explicitly named GeoJSONs; otherwise use a discovered vector file.
green_polygon_path = green_folder_files[0] if green_folder_files else None
if green_polygon_path is None:
    # The project may contain the green-space polygon in the json_file folder.
    possible = [
        p for p in find_files(ROOT / "json_file", {".geojson", ".json", ".shp", ".gpkg"})
        if "green" in p.stem.lower() and "point" not in p.stem.lower()
    ]
    green_polygon_path = possible[0] if possible else None

if green_points_path is None and green_point_folder_files:
    green_points_path = green_point_folder_files[0]

if wards_path is None and ward_folder_files:
    wards_path = ward_folder_files[0]

boundary = read_vector(boundary_path)
wards = read_vector(wards_path)
green_polygons = read_vector(green_polygon_path)
green_points = read_vector(green_points_path)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🌿 Hyderabad Urban Green Spaces & Urban Heat Island")
st.caption("WebGIS dashboard | Local project datasets")

with st.sidebar:
    st.subheader("Layer transparency")
    ward_opacity = st.slider("Ward fill opacity", 0.05, 1.0, 0.25, 0.05)
    green_opacity = st.slider("Green polygon opacity", 0.05, 1.0, 0.45, 0.05)
    raster_opacity = st.slider("Raster opacity", 0.1, 1.0, 0.65, 0.05)

    st.divider()
    st.subheader("Basemap")
    basemap_choice = st.selectbox(
        "Choose basemap",
        ["OpenStreetMap", "CartoDB Voyager", "Esri World Imagery"],
        index=2,
    )

# ------------------------------------------------------------
# RASTER STATISTICS
# ------------------------------------------------------------
raster_stats = None
raster_png = None
raster_bounds = None
raster_display_range = None
if raster_path:
    try:
        raster_stats, raster_png, raster_bounds, raster_display_range = raster_summary_and_overlay(raster_path)
    except Exception as exc:
        st.sidebar.error(f"Raster could not be displayed: {exc}")

# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------
ward_count = len(wards) if wards is not None else 0
green_count = len(green_polygons) if green_polygons is not None else 0
point_count = len(green_points) if green_points is not None else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ward polygons", f"{ward_count:,}" if wards is not None else "N/A")
k2.metric(
    "Mean raster value",
    f"{raster_stats['mean']:.2f}" if raster_stats else "N/A",
    help="Mean of valid raster pixels in the GeoTIFF's stored units. Convert to °C only after confirming the raster scale and offset.",
)
k3.metric("Park / green points", f"{point_count:,}" if green_points is not None else "N/A")
k4.metric("Green-space polygons", f"{green_count:,}" if green_polygons is not None else "N/A")

if raster_stats:
    st.caption(
        f"LST raster statistics are in stored raster units: "
        f"min {raster_stats['min']:.2f} · max {raster_stats['max']:.2f} · "
        f"valid pixels {raster_stats['valid_pixels']:,} · CRS {raster_stats['crs']}. "
        "The dashboard does not assume a scale factor or label the raw mean as °C."
    )

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------
tiles = {
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Voyager": "CartoDB Voyager",
    "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}
m = folium.Map(
    location=[17.385, 78.487],
    zoom_start=11,
    tiles=tiles[basemap_choice],
    attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    control_scale=True,
)

boundary_layer = folium.FeatureGroup(name="Study Boundary", show=True).add_to(m)
ward_layer = folium.FeatureGroup(name="Ward Polygons", show=False).add_to(m)
green_layer = folium.FeatureGroup(name="Green Space Polygons", show=True).add_to(m)
green_points_layer = folium.FeatureGroup(name="Green-space / park points", show=False).add_to(m)
raster_layer = folium.FeatureGroup(name="LST Raster", show=False).add_to(m)

if raster_png and raster_bounds:
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + base64.b64encode(raster_png).decode("ascii"),
        bounds=raster_bounds,
        opacity=raster_opacity,
        name="LST Raster",
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(raster_layer)

add_gdf_layer(boundary_layer, boundary, "Study Boundary", "#f4d35e", 0.03)
add_gdf_layer(ward_layer, wards, "Ward Polygons", "#39c779", ward_opacity)
add_gdf_layer(green_layer, green_polygons, "Green Space Polygons", "#2bd576", green_opacity)
add_gdf_layer(
    green_points_layer,
    green_points,
    "Green-space / park points",
    "#2bd576",
    0.9,
    marker=folium.CircleMarker(
        radius=4,
        color="#16864b",
        weight=1,
        fill=True,
        fill_color="#2bd576",
        fill_opacity=0.95,
    ),
)

m.get_root().header.add_child(folium.Element("""
<style>
.floating-layers-control {
    background: transparent;
    border: 0;
    box-shadow: none;
}
.floating-layers-button {
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 8px;
    padding: 7px 11px;
    background: #173c32;
    color: #fff;
    font: 600 12px/1.2 sans-serif;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.24);
}
.floating-layers-button:hover {
    background: #245846;
}
.floating-layers-panel {
    display: none;
    min-width: 185px;
    margin-top: 6px;
    padding: 8px 10px;
    border: 1px solid #dce5df;
    border-radius: 9px;
    background: rgba(255, 255, 255, 0.97);
    color: #20352c;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    font: 12px/1.35 sans-serif;
}
.floating-layers-panel.is-open {
    display: block;
}
.floating-layers-option {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 4px 1px;
    cursor: pointer;
    white-space: nowrap;
}
.floating-layers-option input {
    margin: 0;
    accent-color: #26845d;
}
</style>
"""))
FloatingLayerControl([
    (boundary_layer, "Study Boundary"),
    (ward_layer, "Ward Polygons"),
    (green_layer, "Green Space Polygons"),
    (green_points_layer, "Green-space / park points"),
    (raster_layer, "LST Raster"),
]).add_to(m)

# Fit map to available vector data.
bounds_sources = [g for g in [boundary, wards, green_polygons, green_points] if g is not None and not g.empty]
if bounds_sources:
    all_bounds = np.array([g.total_bounds for g in bounds_sources])
    minx, miny = np.min(all_bounds[:, :2], axis=0)
    maxx, maxy = np.max(all_bounds[:, 2:], axis=0)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

st.subheader("Spatial analysis map")
st_folium(m, use_container_width=True, height=590, returned_objects=[])

# ------------------------------------------------------------
# CHARTS
# ------------------------------------------------------------
st.subheader("Dataset analysis")
chart1, chart2 = st.columns(2)

with chart1:
    counts = pd.DataFrame({
        "Dataset": ["Ward polygons", "Green-space polygons", "Park / green points"],
        "Feature count": [
            ward_count if wards is not None else 0,
            green_count if green_polygons is not None else 0,
            point_count if green_points is not None else 0,
        ],
    }).set_index("Dataset")
    st.bar_chart(counts)

with chart2:
    status_rows = [
        ("Study boundary", boundary_path, boundary),
        ("Ward polygons", wards_path, wards),
        ("Green-space polygons", green_polygon_path, green_polygons),
        ("Park / green points", green_points_path, green_points),
        ("LST GeoTIFF", raster_path, raster_stats),
    ]
    status_df = pd.DataFrame([
        {
            "Dataset": name,
            "File": str(path.relative_to(ROOT)) if path and path.exists() else "Not found",
            "Status": "Loaded" if data is not None else "Unavailable",
        }
        for name, path, data in status_rows
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# DATA INVENTORY
# ------------------------------------------------------------
st.subheader("Dataset inventory")
inventory = []
for label, path, data in status_rows:
    if label == "LST GeoTIFF":
        count_value = f"{raster_stats['valid_pixels']:,} valid pixels" if raster_stats else "N/A"
    else:
        count_value = f"{len(data):,} features" if data is not None else "N/A"
    inventory.append({
        "Dataset": label,
        "Path": str(path.relative_to(ROOT)) if path and path.exists() else "Not found",
        "Status": "Loaded" if data is not None else "Unavailable",
        "Features / value": count_value,
    })
st.dataframe(pd.DataFrame(inventory), use_container_width=True, hide_index=True)

csv = pd.DataFrame(inventory).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download dataset inventory CSV",
    data=csv,
    file_name="hyderabad_ugs_uhi_inventory.csv",
    mime="text/csv",
)

with st.expander("Detected project files"):
    for folder in ["json_file", "Raster1", "Green_space", "green_point", "hydrabad_shp"]:
        folder_path = ROOT / folder
        st.write(f"**{folder}/**")
        if folder_path.exists():
            files = [p.relative_to(ROOT).as_posix() for p in folder_path.rglob("*") if p.is_file()]
            st.write(files if files else "No files found.")
        else:
            st.write("Folder not found.")

st.info(
    "No artificial features or temperature values are added. The LST mean shown above is the raw mean "
    "of valid GeoTIFF pixel values. Confirm the raster's scale factor and offset before converting it to °C."
)
