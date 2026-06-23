import os
import glob
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from shapely.geometry import mapping

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


RAW_GSOD_DIR = r"G:\CYX\GSOD\NCEIgsod"
CA_DIR = r"G:\Central Asia"
LULC_TIF = r"G:\CYX\MODIS\2001to2024\24tif reclass\MCD12Q1_LC_Type1_CA_2024_LULC6.tif"
OUT_DIR = r"G:\CYX\Analyse\Fig1with MODIS"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_START = 1950
YEAR_END = 2024
TARGET_N_STATIONS = 450
PREFER_CACHE = True
FORCE_REBUILD_RAW_SUMMARY = False
FORCE_REBUILD_CA_POINTS = False
FORCE_REBUILD_LULC_CLIP = False

FIG_DPI = 600
MAP_FIGSIZE = (12.4, 8.6)
FONT_FAMILY = "Times New Roman"
LANDUSE_ALPHA = 0.88
POINT_SIZE = 34
POINT_EDGEWIDTH = 0.38


OUT_RAW_STATION_YEAR_CSV = os.path.join(OUT_DIR, "fig01_station_year_metadata_1950_2024.csv")
OUT_RAW_SUMMARY_CSV = os.path.join(OUT_DIR, "fig01_station_summary_1950_2024_from_raw.csv")
OUT_CA_ALL_CSV = os.path.join(OUT_DIR, "fig01_station_summary_CA_all.csv")
OUT_CA_ALL_GPKG = os.path.join(OUT_DIR, "fig01_station_points_CA_all.gpkg")
OUT_CA_ALL_SHP = os.path.join(OUT_DIR, "fig01_station_points_CA_all.shp")
OUT_CA_TOP450_CSV = os.path.join(OUT_DIR, "fig01_station_summary_CA_top450.csv")
OUT_CA_TOP450_GPKG = os.path.join(OUT_DIR, "fig01_station_points_CA_top450.gpkg")
OUT_CA_TOP450_SHP = os.path.join(OUT_DIR, "fig01_station_points_CA_top450.shp")
OUT_LULC_CLIP = os.path.join(OUT_DIR, "fig01_lulc6_CA_clip.tif")
OUT_MAP_PNG = os.path.join(OUT_DIR, "Fig1_CA_MODIS_GSOD_top450.png")
OUT_MAP_TIF = os.path.join(OUT_DIR, "Fig1_CA_MODIS_GSOD_top450.tif")
OUT_SKIPPED_DEBUG = os.path.join(OUT_DIR, "fig01_skipped_raw_files_debug.csv")
OUT_REPORT_TXT = os.path.join(OUT_DIR, "fig01_build_report.txt")

LANDUSE_INFO = [
    {"value": 1, "label": "Cropland", "color": "#C7A740"},
    {"value": 2, "label": "Forest", "color": "#5C8B49"},
    {"value": 3, "label": "Grassland", "color": "#8EAA4E"},
    {"value": 4, "label": "Water Body", "color": "#5A97CC"},
    {"value": 5, "label": "Built-up Areas", "color": "#B96767"},
    {"value": 6, "label": "Other", "color": "#A99273"},
]


def set_plot_style():
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10.5
    plt.rcParams["ytick.labelsize"] = 10.5
    plt.rcParams["axes.titlesize"] = 15


def choose_best_shp(shp_dir):
    shp_paths = sorted(glob.glob(os.path.join(shp_dir, "*.shp")))
    if not shp_paths:
        raise FileNotFoundError(f"未在 {shp_dir} 找到 *.shp")

    preferred_keywords = ["central", "asia", "ca", "admin", "boundary", "border"]
    scored = []
    for p in shp_paths:
        stem = Path(p).stem.lower()
        score = sum(k in stem for k in preferred_keywords)
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def read_csv_try_encodings(path, usecols=None, nrows=None):
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, nrows=nrows)
        except Exception as e:
            last_err = e
    raise last_err


def parse_numeric_series(series):
    s = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    num = pd.to_numeric(s, errors="coerce")
    return num


def valid_temp_series(series):
    num = parse_numeric_series(series)
    num = num.mask(np.abs(num) >= 900)
    return num


def list_year_folders(raw_root, year_start, year_end):
    out = []
    for p in sorted(Path(raw_root).iterdir()):
        if not p.is_dir():
            continue
        try:
            y = int(p.name)
        except Exception:
            continue
        if year_start <= y <= year_end:
            out.append((y, str(p)))
    return out


def list_station_files(year_folders):
    all_files = []
    for year, folder in year_folders:
        paths = []
        for pattern in ["*.csv", "*.CSV"]:
            paths.extend(glob.glob(os.path.join(folder, pattern)))
        paths = sorted(set(paths))
        for p in paths:
            all_files.append((year, p))
    return all_files


def parse_raw_station_year_file(file_path, year):
    try:
        sample = read_csv_try_encodings(file_path, nrows=8)
    except Exception as e:
        return None, f"read_sample_failed: {e}"

    if sample is None or sample.empty:
        return None, "empty_file"

    colmap = {c.upper(): c for c in sample.columns}
    required_upper = ["STATION", "LATITUDE", "LONGITUDE", "NAME"]
    if not all(c in colmap for c in required_upper):
        return None, f"missing_required_columns: {list(sample.columns)}"

    usecols_upper = ["STATION", "LATITUDE", "LONGITUDE", "NAME", "ELEVATION", "TEMP", "MAX", "MIN"]
    usecols_actual = [colmap[c] for c in usecols_upper if c in colmap]

    try:
        df = read_csv_try_encodings(file_path, usecols=usecols_actual)
    except Exception as e:
        return None, f"read_full_failed: {e}"

    if df.empty:
        return None, "empty_after_read"

    station_col = colmap["STATION"]
    lat_col = colmap["LATITUDE"]
    lon_col = colmap["LONGITUDE"]
    name_col = colmap["NAME"]
    elev_col = colmap.get("ELEVATION", None)

    station_vals = df[station_col].dropna().astype(str)
    if station_vals.empty:
        return None, "station_missing"
    station = station_vals.iloc[0].strip()

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    lat = lat[(lat >= -90) & (lat <= 90)]
    lon = lon[(lon >= -180) & (lon <= 180)]
    if lat.empty or lon.empty:
        return None, "invalid_latlon"

    name_vals = df[name_col].dropna().astype(str).str.strip()
    name = name_vals.mode().iloc[0] if not name_vals.empty else station

    elev = np.nan
    if elev_col is not None and elev_col in df.columns:
        elev_vals = pd.to_numeric(df[elev_col], errors="coerce")
        elev_vals = elev_vals[np.isfinite(elev_vals)]
        if not elev_vals.empty:
            elev = float(np.nanmedian(elev_vals))

    temp_cols = [c for c in [colmap.get("TEMP"), colmap.get("MAX"), colmap.get("MIN")] if c is not None]
    n_valid_temp_rows = 0
    if temp_cols:
        valid_any = pd.Series(False, index=df.index)
        for c in temp_cols:
            valid_any = valid_any | valid_temp_series(df[c]).notna()
        n_valid_temp_rows = int(valid_any.sum())

    if n_valid_temp_rows <= 0:
        return None, "no_valid_temp_rows"

    rec = {
        "station": station,
        "name": name,
        "lat": float(np.nanmedian(lat)),
        "lon": float(np.nanmedian(lon)),
        "elev": elev,
        "year": int(year),
        "n_valid_temp_rows": n_valid_temp_rows,
        "source_file": os.path.basename(file_path),
        "source_relpath": os.path.relpath(file_path, RAW_GSOD_DIR),
    }
    return rec, None


def scan_raw_gsod_station_years(raw_root, year_start, year_end):
    year_folders = list_year_folders(raw_root, year_start, year_end)
    if not year_folders:
        raise FileNotFoundError(f"未在 {raw_root} 找到 {year_start}-{year_end} 年文件夹")

    all_files = list_station_files(year_folders)
    if not all_files:
        raise FileNotFoundError(f"在 {raw_root} 的 {year_start}-{year_end} 年文件夹中未找到 CSV 文件")

    records = []
    skipped = []
    pbar = tqdm(all_files, desc="Scanning GSOD raw files", unit="file")
    for year, file_path in pbar:
        rec, reason = parse_raw_station_year_file(file_path, year)
        if rec is not None:
            records.append(rec)
        else:
            skipped.append({"year": year, "file": file_path, "reason": reason})
    return records, skipped


def build_station_summary_from_station_year_df(df_year):
    if df_year.empty:
        raise ValueError("station-year metadata 为空，无法汇总站点")

    df = df_year.copy()
    df["station"] = df["station"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["station", "year", "lat", "lon"])
    df["year"] = df["year"].astype(int)

    agg_rows = []
    for station, grp in df.groupby("station", sort=True):
        years = np.sort(grp["year"].unique())
        name = grp["name"].dropna().astype(str).mode().iloc[0] if grp["name"].notna().any() else station
        lat = float(np.nanmedian(pd.to_numeric(grp["lat"], errors="coerce")))
        lon = float(np.nanmedian(pd.to_numeric(grp["lon"], errors="coerce")))
        elev_vals = pd.to_numeric(grp["elev"], errors="coerce")
        elev = float(np.nanmedian(elev_vals)) if np.isfinite(elev_vals).any() else np.nan
        agg_rows.append({
            "station": str(station),
            "name": name,
            "lat": lat,
            "lon": lon,
            "elev": elev,
            "year_min": int(years.min()),
            "year_max": int(years.max()),
            "n_years": int(len(years)),
            "continuity_ratio": float(len(years) / (years.max() - years.min() + 1)),
            "n_station_year_files": int(grp.shape[0]),
        })

    out = pd.DataFrame(agg_rows)
    out = out.sort_values(["n_years", "year_min", "station"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def load_summary_csv(path):
    df = pd.read_csv(path)
    required = {"station", "name", "lat", "lon", "year_min", "year_max", "n_years"}
    if not required.issubset(df.columns):
        raise ValueError(f"缓存汇总表缺少字段：{required - set(df.columns)}")
    return df


def read_or_build_raw_summary():
    if PREFER_CACHE and (not FORCE_REBUILD_RAW_SUMMARY) and os.path.exists(OUT_RAW_SUMMARY_CSV):
        tqdm.write(f"[CACHE] 读取站点汇总缓存：{OUT_RAW_SUMMARY_CSV}")
        summary = load_summary_csv(OUT_RAW_SUMMARY_CSV)
        skipped = pd.read_csv(OUT_SKIPPED_DEBUG) if os.path.exists(OUT_SKIPPED_DEBUG) else pd.DataFrame()
        station_year_df = pd.read_csv(OUT_RAW_STATION_YEAR_CSV) if os.path.exists(OUT_RAW_STATION_YEAR_CSV) else pd.DataFrame()
        return station_year_df, summary, skipped

    records, skipped = scan_raw_gsod_station_years(RAW_GSOD_DIR, YEAR_START, YEAR_END)
    station_year_df = pd.DataFrame(records)
    if station_year_df.empty:
        raise RuntimeError("扫描完成后未获得任何有效 station-year 记录，请检查原始 GSOD 数据格式")

    summary = build_station_summary_from_station_year_df(station_year_df)
    station_year_df.to_csv(OUT_RAW_STATION_YEAR_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_RAW_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped).to_csv(OUT_SKIPPED_DEBUG, index=False, encoding="utf-8-sig")

    tqdm.write(f"[SAVE] station-year 元数据：{OUT_RAW_STATION_YEAR_CSV}")
    tqdm.write(f"[SAVE] 站点汇总：{OUT_RAW_SUMMARY_CSV}")
    if skipped:
        tqdm.write(f"[SAVE] 跳过文件记录：{OUT_SKIPPED_DEBUG}")
    return station_year_df, summary, pd.DataFrame(skipped)


def load_points_cache(path):
    gdf = gpd.read_file(path)
    required = {"station", "name", "lat", "lon", "year_min", "year_max", "n_years", "geometry"}
    if not required.issubset(gdf.columns):
        raise ValueError(f"缓存点图层缺少字段：{required - set(gdf.columns)}")
    return gdf


def build_ca_station_points(summary_df, ca_outer):
    df = summary_df.copy()
    df = df.dropna(subset=["lon", "lat"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf = gpd.clip(gdf, ca_outer)
    gdf = gdf.drop_duplicates(subset=["station"]).copy()
    gdf = gdf.sort_values(["n_years", "year_min", "station"], ascending=[False, True, True]).reset_index(drop=True)
    return gdf


def select_top450(gdf, target_n=450):
    g = gdf.sort_values(["n_years", "year_min", "station"], ascending=[False, True, True]).reset_index(drop=True)
    if g.shape[0] > target_n:
        g = g.head(target_n).copy()
    return g.reset_index(drop=True)


def read_or_build_ca_points(summary_df, ca_outer):
    if PREFER_CACHE and (not FORCE_REBUILD_CA_POINTS) and os.path.exists(OUT_CA_ALL_GPKG) and os.path.exists(OUT_CA_TOP450_GPKG):
        tqdm.write(f"[CACHE] 读取研究区全部站点缓存：{OUT_CA_ALL_GPKG}")
        tqdm.write(f"[CACHE] 读取研究区 top450 站点缓存：{OUT_CA_TOP450_GPKG}")
        gdf_all = load_points_cache(OUT_CA_ALL_GPKG)
        gdf_top450 = load_points_cache(OUT_CA_TOP450_GPKG)
        return gdf_all, gdf_top450

    gdf_all = build_ca_station_points(summary_df, ca_outer)
    gdf_top450 = select_top450(gdf_all, TARGET_N_STATIONS)

    gdf_all.drop(columns=[c for c in gdf_all.columns if c.startswith("index_")], errors="ignore").to_file(OUT_CA_ALL_GPKG, driver="GPKG")
    gdf_top450.drop(columns=[c for c in gdf_top450.columns if c.startswith("index_")], errors="ignore").to_file(OUT_CA_TOP450_GPKG, driver="GPKG")


    try:
        gdf_all.to_file(OUT_CA_ALL_SHP, driver="ESRI Shapefile")
    except Exception:
        pass
    try:
        gdf_top450.to_file(OUT_CA_TOP450_SHP, driver="ESRI Shapefile")
    except Exception:
        pass
    pd.DataFrame(gdf_all.drop(columns="geometry")).to_csv(OUT_CA_ALL_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(gdf_top450.drop(columns="geometry")).to_csv(OUT_CA_TOP450_CSV, index=False, encoding="utf-8-sig")

    tqdm.write(f"[SAVE] 研究区全部站点：{OUT_CA_ALL_GPKG}")
    tqdm.write(f"[SAVE] 研究区 top450 站点：{OUT_CA_TOP450_GPKG}")
    return gdf_all, gdf_top450


def clip_lulc_to_ca(ca_outer):
    if PREFER_CACHE and (not FORCE_REBUILD_LULC_CLIP) and os.path.exists(OUT_LULC_CLIP):
        tqdm.write(f"[CACHE] 读取裁剪后土地利用：{OUT_LULC_CLIP}")
        return OUT_LULC_CLIP

    with rasterio.open(LULC_TIF) as src:
        ca_raster_crs = ca_outer.to_crs(src.crs)
        geoms = [mapping(geom) for geom in ca_raster_crs.geometry]
        out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })
        with rasterio.open(OUT_LULC_CLIP, "w", **out_meta) as dst:
            dst.write(out_image)

    tqdm.write(f"[SAVE] 裁剪后土地利用：{OUT_LULC_CLIP}")
    return OUT_LULC_CLIP


def make_rgba_from_lulc(arr, alpha=0.88):
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.float32)
    for info in LANDUSE_INFO:
        mask_val = np.isfinite(arr) & (arr == info["value"])
        rgba[mask_val] = mpl.colors.to_rgba(info["color"], alpha=alpha)
    return rgba


def write_report(ca_all_count, selected_count, summary_count, skipped_count):
    lines = [
        "Fig.1 build report",
        f"Raw GSOD root: {RAW_GSOD_DIR}",
        f"Year range: {YEAR_START}-{YEAR_END}",
        f"Summary stations from raw: {summary_count}",
        f"Stations inside Central Asia before trimming: {ca_all_count}",
        f"Stations kept for plotting: {selected_count}",
        f"Skipped raw files: {skipped_count}",
        f"Target station count: {TARGET_N_STATIONS}",
        f"LULC raster: {LULC_TIF}",
        f"Output dir: {OUT_DIR}",
    ]
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_main_map(ca, ca_outer, gdf_top450, lulc_clip_path):
    set_plot_style()

    with rasterio.open(lulc_clip_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        bounds = src.bounds

    rgba = make_rgba_from_lulc(arr, alpha=LANDUSE_ALPHA)

    cover = pd.to_numeric(gdf_top450["n_years"], errors="coerce").astype(float)
    vmin = float(np.nanmin(cover)) if np.isfinite(np.nanmin(cover)) else 1.0
    vmax = float(np.nanmax(cover)) if np.isfinite(np.nanmax(cover)) else 75.0
    vmin = min(vmin, 1.0)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps.get_cmap("viridis")

    fig = plt.figure(figsize=MAP_FIGSIZE, dpi=FIG_DPI)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[16, 2.45], hspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[1, 0])


    ax.imshow(
        rgba,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        zorder=1,
    )

    ca.boundary.plot(ax=ax, lw=0.55, color="#6C6C6C", zorder=2)
    ca_outer.boundary.plot(ax=ax, lw=1.65, color="#222222", zorder=3)

    sc = ax.scatter(
        gdf_top450.geometry.x,
        gdf_top450.geometry.y,
        c=cover,
        cmap=cmap,
        norm=norm,
        s=POINT_SIZE,
        linewidths=POINT_EDGEWIDTH,
        edgecolors="#111111",
        zorder=4,
    )

    minx, miny, maxx, maxy = ca_outer.total_bounds
    ax.set_xlim(minx - 1.0, maxx + 1.0)
    ax.set_ylim(miny - 1.0, maxy + 1.0)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45, color="#BFBFBF")
    ax.set_title(f"GSOD stations over MODIS land use in Central Asia (1950–2024, N={gdf_top450.shape[0]})", pad=10)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#444444")


    ax_leg.set_facecolor("white")

    pos = ax_leg.get_position()
    ax_leg.set_position([pos.x0, pos.y0 - 0.012, pos.width, pos.height])
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    for spine in ax_leg.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_edgecolor("#555555")

    ax_leg.text(0.015, 0.78, "Land use", fontsize=10.5, fontweight="bold", va="center")
    handles = [Patch(facecolor=info["color"], edgecolor="none", label=info["label"]) for info in LANDUSE_INFO]
    leg = ax_leg.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.10, 0.44),
        frameon=False,
        ncol=6,
        fontsize=9.5,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=1.00,
        borderaxespad=0.0,
    )

    ax_leg.text(0.72, 0.78, "Coverage years", fontsize=10.5, fontweight="bold", va="center")
    cax = ax_leg.inset_axes([0.70, 0.28, 0.26, 0.28])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    if vmax - vmin <= 12:
        ticks = np.linspace(vmin, vmax, 5)
    else:
        step = 10 if vmax > 30 else 5
        t0 = int(np.ceil(vmin / step) * step)
        ticks = np.arange(t0, vmax + 0.1, step)
        if len(ticks) < 3:
            ticks = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=9, length=2.5, pad=2)
    cbar.outline.set_linewidth(0.55)

    fig.savefig(OUT_MAP_PNG, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUT_MAP_TIF, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    workflow = tqdm(total=6, desc="Fig.1 workflow", unit="step")


    ca_shp = choose_best_shp(CA_DIR)
    ca = gpd.read_file(ca_shp).to_crs("EPSG:4326")
    ca_tmp = ca.copy()
    ca_tmp["__g__"] = 1
    ca_outer = ca_tmp.dissolve(by="__g__").reset_index(drop=True)
    workflow.update(1)
    tqdm.write(f"[INFO] 研究区 shp：{ca_shp}")


    _, summary_df, skipped_df = read_or_build_raw_summary()
    workflow.update(1)
    tqdm.write(f"[INFO] 原始汇总站点数：{summary_df.shape[0]}")


    gdf_all_ca, gdf_top450 = read_or_build_ca_points(summary_df, ca_outer)
    workflow.update(1)
    tqdm.write(f"[INFO] 中亚范围内站点数（裁剪前保留）：{gdf_all_ca.shape[0]}")
    tqdm.write(f"[INFO] 最终用于绘图的站点数：{gdf_top450.shape[0]}")


    lulc_clip = clip_lulc_to_ca(ca_outer)
    workflow.update(1)


    plot_main_map(ca, ca_outer, gdf_top450, lulc_clip)
    workflow.update(1)


    write_report(
        ca_all_count=gdf_all_ca.shape[0],
        selected_count=gdf_top450.shape[0],
        summary_count=summary_df.shape[0],
        skipped_count=0 if skipped_df is None else skipped_df.shape[0],
    )
    workflow.update(1)
    workflow.close()

    print("\n[DONE] 主图已输出：")
    print(f"  PNG: {OUT_MAP_PNG}")
    print(f"  TIF: {OUT_MAP_TIF}")
    print("[DONE] 中间文件已输出：")
    print(f"  {OUT_RAW_STATION_YEAR_CSV}")
    print(f"  {OUT_RAW_SUMMARY_CSV}")
    print(f"  {OUT_CA_ALL_GPKG}")
    print(f"  {OUT_CA_TOP450_GPKG}")
    print(f"  {OUT_LULC_CLIP}")
    print(f"[DONE] 构建报告：{OUT_REPORT_TXT}")


if __name__ == "__main__":
    main()
