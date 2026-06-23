import os, glob, warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from tqdm import tqdm


STN_DIR   = r"G:\CYX\Analyse\normalized_by_station"
CA_DIR    = r"G:\Central Asia"
NE_COUNTRY= r"F:\naturalearth_data\ne_10m_admin_0_countries.shp"
OUT_DIR   = r"G:\CYX\Analyse\figs"
OUT_FIG   = os.path.join(OUT_DIR, "fig_01_map_stations.png")
OUT_CSV   = os.path.join(OUT_DIR, "fig01_station_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_MIN_REQUIRED = 1950


tqdm.write("[Step 1/4] 读取研究区边界 ...")
ca_shps = glob.glob(os.path.join(CA_DIR, "*.shp"))
if not ca_shps:
    raise FileNotFoundError(f"未在 {CA_DIR} 找到 *.shp，确认中亚研究区矢量是否存在。")
ca = gpd.read_file(ca_shps[0]).to_crs("EPSG:4326")
ca_bounds = ca.total_bounds


tqdm.write("[Step 2/4] 读取国家边界（可选） ...")
countries = None
if os.path.exists(NE_COUNTRY):
    try:
        countries = gpd.read_file(NE_COUNTRY).to_crs("EPSG:4326")
    except Exception:
        warnings.warn("无法读取国家边界矢量，略过底图。")


def _pick_col(cols, candidates):
    for c in candidates:
        if c in cols: return c

    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def parse_station_csv(fp):
    try:
        df_head = pd.read_csv(fp, nrows=200)
    except Exception:
        warnings.warn(f"无法读取文件：{os.path.basename(fp)}")
        return None

    cols = list(df_head.columns)


    id_col   = _pick_col(cols, ["station_id", "USAF", "WBAN", "station", "stnid", "id"])
    name_col = _pick_col(cols, ["name", "station_name"])
    lat_col  = _pick_col(cols, ["lat", "latitude", "Lat", "LAT"])
    lon_col  = _pick_col(cols, ["lon", "longitude", "Lon", "LON"])
    elev_col = _pick_col(cols, ["elev", "elevation", "alt", "ALT"])
    date_col = _pick_col(cols, ["date", "DATE", "time", "Time", "DATE_TIME"])

    if lat_col is None or lon_col is None:
        warnings.warn(f"{os.path.basename(fp)} 缺少经纬度列，跳过该站。")
        return None

    year_min = None; year_max = None; n_days = None
    if date_col is not None:
        try:

            dt = pd.read_csv(fp, usecols=[date_col], parse_dates=[date_col], infer_datetime_format=True)
            year_min = int(dt[date_col].dt.year.min())
            year_max = int(dt[date_col].dt.year.max())
            n_days   = int(dt[date_col].notna().sum())
        except Exception:

            try:
                dts = pd.to_datetime(df_head[date_col], errors="coerce")
                year_min = int(dts.dt.year.min())
                year_max = int(dts.dt.year.max())
                n_days   = int(dts.notna().sum())
            except Exception:
                pass

    rec = {
        "file": os.path.basename(fp),
        "station_id": df_head[id_col].iloc[0] if (id_col and id_col in df_head) else os.path.splitext(os.path.basename(fp))[0],
        "name": df_head[name_col].iloc[0] if (name_col and name_col in df_head) else None,
        "lat": float(df_head[lat_col].iloc[0]),
        "lon": float(df_head[lon_col].iloc[0]),
        "elev": float(df_head[elev_col].iloc[0]) if (elev_col and elev_col in df_head) else None,
        "year_min": year_min,
        "year_max": year_max,
        "n_years": (year_max - year_min + 1) if (year_min and year_max) else None,
        "n_days": n_days,
    }
    return rec

tqdm.write("[Step 3/4] 扫描并解析站点CSV ...")
csv_files = sorted(glob.glob(os.path.join(STN_DIR, "*.csv")))
meta_rows = []
for fp in tqdm(csv_files, desc="Parsing stations", unit="file"):
    rec = parse_station_csv(fp)
    if rec is not None and rec["year_min"] is not None and rec["year_min"] >= YEAR_MIN_REQUIRED:
        meta_rows.append(rec)

if not meta_rows:
    raise RuntimeError(f"未成功解析任何站点，请检查 {STN_DIR} 下CSV是否包含 lat/lon 与 date 列，并满足 year_min >= {YEAR_MIN_REQUIRED}。")

stn_df = pd.DataFrame(meta_rows)


tqdm.write("[Step 4/4] 构建点要素与空间裁剪 ...")

geoms = []
for lon, lat in tqdm(zip(stn_df["lon"].values, stn_df["lat"].values), total=len(stn_df), desc="Building geometries", unit="pt"):
    geoms.append(Point(lon, lat))

gdf_stn = gpd.GeoDataFrame(stn_df.copy(), geometry=geoms, crs="EPSG:4326")


try:
    gdf_stn = gpd.overlay(gdf_stn, ca[["geometry"]], how="intersection")
except Exception:
    gdf_stn = gpd.clip(gdf_stn, ca)


plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=150)

if countries is not None:
    countries.boundary.plot(ax=ax, linewidth=0.5, color="#AAAAAA", zorder=1)
ca.boundary.plot(ax=ax, linewidth=1.8, color="#333333", zorder=2)


cover = gdf_stn["n_years"].fillna(0)
sc = gdf_stn.plot(
    ax=ax, marker="o", column=cover,
    cmap="viridis", markersize=18, linewidth=0.1, edgecolor="#222222", zorder=3
)

cbar = plt.colorbar(sc.collections[0], ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Coverage (years)")

minx, miny, maxx, maxy = ca_bounds
ax.set_xlim(minx - 1.0, maxx + 1.0)
ax.set_ylim(miny - 1.0, maxy + 1.0)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

n_sites = gdf_stn.shape[0]
ax.set_title(f"Central Asia & GSOD Stations (N={n_sites}, since {YEAR_MIN_REQUIRED})", fontsize=13, pad=10)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
stn_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

tqdm.write(f"[DONE] 图已保存：{OUT_FIG}")
tqdm.write(f"[DONE] 站点汇总表已导出：{OUT_CSV}")
