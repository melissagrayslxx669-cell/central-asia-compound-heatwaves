import os, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


SUM_CSV = r"G:\CYX\Analyse\figs\fig01_station_summary.csv"
CA_DIR  = r"G:\Central Asia"
NE_SHP  = r"F:\naturalearth_data\ne_10m_admin_0_countries.shp"
OUT_FIG = r"G:\CYX\Analyse\figs\fig_01_map_stations_CA_with_inset.png"
YEAR_MIN_REQUIRED = 1950


ca_paths = glob.glob(os.path.join(CA_DIR, "*.shp"))
if not ca_paths:
    raise FileNotFoundError(f"未在 {CA_DIR} 找到 *.shp")
ca = gpd.read_file(ca_paths[0]).to_crs("EPSG:4326")


_ca = ca.copy()
_ca["__g__"] = 1
ca_outer = _ca.dissolve(by="__g__")
ca_bounds = ca_outer.total_bounds


countries = None
if os.path.exists(NE_SHP):
    try:
        countries = gpd.read_file(NE_SHP).to_crs("EPSG:4326")
    except Exception:
        countries = None


df = pd.read_csv(SUM_CSV)
df = df[df["year_min"].fillna(0) >= YEAR_MIN_REQUIRED].copy()
if "n_years" not in df.columns and {"year_min", "year_max"} <= set(df.columns):
    df["n_years"] = df["year_max"] - df["year_min"] + 1
df = df.dropna(subset=["lon", "lat"])
df = df[df["n_years"].fillna(0) > 0]


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
gdf = gpd.clip(gdf, ca_outer)


cover = gdf["n_years"].astype(float)
vmin, vmax = 1.0, 80.0
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = mpl.colormaps.get_cmap("viridis")
ticks = [1, 20, 40, 60, 80]


plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=150)


if countries is not None:
    countries.boundary.plot(ax=ax, lw=0.5, color="#B5B5B5", zorder=1, linestyle="-")
ca.boundary.plot(ax=ax, lw=0.8, color="#666666", zorder=2, linestyle="-")
ca_outer.boundary.plot(ax=ax, lw=2.2, color="#222222", zorder=3, linestyle="-")


ax.scatter(
    gdf["lon"], gdf["lat"],
    c=cover, cmap=cmap, norm=norm,
    s=18, linewidths=0.15, edgecolors="#222222", zorder=4
)


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label("Coverage (years)")
cbar.set_ticks(ticks)
cbar.set_ticklabels([str(t) for t in ticks])


minx, miny, maxx, maxy = ca_bounds
ax.set_xlim(minx - 1, maxx + 1)
ax.set_ylim(miny - 1, maxy + 1)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(True, ls="--", lw=0.4, alpha=0.55, color="#CBCBCB")

title_n = gdf.shape[0]
ax.set_title(f"Central Asia & GSOD Stations (N={title_n}, since {YEAR_MIN_REQUIRED})", fontsize=13, pad=10)


axins = inset_axes(ax, width="28%", height="32%", loc="upper right", borderpad=1.2)
bins = np.arange(max(1, int(cover.min())) - 0.5, min(80, int(cover.max())) + 1.5, 2)
axins.hist(cover, bins=bins, color="#4C72B0", edgecolor="#333333", linewidth=0.4)
axins.set_title("Coverage distribution", fontsize=9, pad=4)
axins.set_xlabel("years", fontsize=8)
axins.set_ylabel("count", fontsize=8)
axins.tick_params(labelsize=8)


med = float(np.nanmedian(cover))
axins.axvline(med, color="#D62728", lw=1.2)
axins.text(med, axins.get_ylim()[1]*0.92, f"median={med:.0f}", color="#D62728",
           ha="center", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D62728", lw=0.8))

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
print(f"[DONE] 图已保存：{OUT_FIG}")
print(f"[INFO] 中亚范围内站点数：{title_n}")
