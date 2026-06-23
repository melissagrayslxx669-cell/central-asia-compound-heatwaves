import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"


CA_SHP_DIR  = r"G:\Central Asia"
OUT_DIR     = r"G:\CYX\Analyse\figs_GSOD"
INTERMEDIATE_CSV = os.path.join(OUT_DIR, "gsod_HWI_station_metrics_1950-2024_CA.csv")


LON_MIN, LON_MAX = 46.0, 96.5
LAT_MIN, LAT_MAX = 34.0, 55.7


def load_ca_polygon_and_boundary(shp_dir):
    shp_files = sorted([f for f in os.listdir(shp_dir) if f.lower().endswith(".shp")])
    if not shp_files:
        raise FileNotFoundError(f"在 {shp_dir} 未找到 *.shp")
    shp_path = os.path.join(shp_dir, shp_files[0])
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    poly = unary_union(gdf.geometry)
    ca_geo = gpd.GeoSeries([poly], crs="EPSG:4326")
    return poly, ca_geo


def plot_scatter_metric(df, value_col, p_col, ca_geo,
                        vmin, vmax,
                        title, suptitle,
                        cbar_label, out_png, out_pdf,
                        alpha_sig=0.05):
    lon = df["lon"].values
    lat = df["lat"].values
    val = df[value_col].values

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon_v = lon[valid]
    lat_v = lat[valid]
    val_v = val[valid]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    fig.suptitle(suptitle, fontsize=12, y=0.97)


    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    sc = ax.scatter(
        lon_v, lat_v, c=val_v,
        s=22, cmap="RdBu_r", norm=norm,
        edgecolors="none", marker="s",
        zorder=2,
    )


    if (p_col is not None) and (p_col in df.columns):
        p_all = df[p_col].values
        sig = valid & np.isfinite(p_all) & (p_all < alpha_sig)
        if sig.any():
            ax.scatter(
                lon[sig], lat[sig],
                s=10, c="k", marker=".", linewidths=0,
                label=f"MK p < {alpha_sig}", zorder=3,
            )
            ax.legend(loc="lower left", fontsize=7, frameon=False)


    ca_geo.boundary.plot(
        ax=ax, color="k", linewidth=1.0, zorder=4
    )

    ax.set_title(title, pad=8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.025, pos.height])
    cb = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.set_label(cbar_label)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():

    if not os.path.exists(INTERMEDIATE_CSV):
        raise FileNotFoundError(
            f"未找到中间结果文件：{INTERMEDIATE_CSV}\n"
            f"请先运行“计算 GSOD HWI 指标并生成该 CSV 的脚本”再画图。"
        )
    print("[INFO] 读取中间结果：", INTERMEDIATE_CSV)
    dfm = pd.read_csv(INTERMEDIATE_CSV)


    print("[INFO] 读取中亚 shp 并筛选站点 …")
    ca_poly, ca_geo = load_ca_polygon_and_boundary(CA_SHP_DIR)
    pts = [Point(xy) for xy in zip(dfm["lon"].values, dfm["lat"].values)]
    inside = np.array(
        [ca_poly.contains(p) or ca_poly.touches(p) for p in tqdm(pts, desc="点在中亚内?", ncols=88)],
        dtype=bool,
    )
    dfm = dfm[inside].reset_index(drop=True)
    print(f"[INFO] 中亚内部站点数：{len(dfm)}")

    if dfm.empty:
        raise RuntimeError("中亚区域内没有可用 GSOD 站点。")


    print("[INFO] 统一色标范围 …")
    vals_delta = np.concatenate([
        np.abs(dfm["delta_dl"].values[np.isfinite(dfm["delta_dl"].values)]),
        np.abs(dfm["delta_nl"].values[np.isfinite(dfm["delta_nl"].values)]),
        np.abs(dfm["delta_cl"].values[np.isfinite(dfm["delta_cl"].values)]),
    ])
    vmax_delta = float(np.percentile(vals_delta, 98)) if vals_delta.size > 0 else 0.5
    if vmax_delta <= 0:
        vmax_delta = 0.5
    vmin_delta = -vmax_delta

    vals_trend = np.concatenate([
        np.abs(dfm["slope_dl"].values[np.isfinite(dfm["slope_dl"].values)]),
        np.abs(dfm["slope_nl"].values[np.isfinite(dfm["slope_nl"].values)]),
        np.abs(dfm["slope_cl"].values[np.isfinite(dfm["slope_cl"].values)]),
    ])
    vmax_trend = float(np.percentile(vals_trend, 98)) if vals_trend.size > 0 else 0.01
    if vmax_trend <= 0:
        vmax_trend = 0.01
    vmin_trend = -vmax_trend


    suptitle = "GSOD Heatwave Intensity (HWI), UTC+05:00"


    title_dl_delta = (
        "DL (Daytime, TX90p on Tmax)\n"
        "ΔHWI = Late (1990–2024) − Early (1950–1984)"
    )
    title_dl_trend = (
        "DL (Daytime, TX90p on Tmax)\n"
        "Trend of annual HWI (Sen's slope, 1950–2024)"
    )

    plot_scatter_metric(
        dfm, "delta_dl", None, ca_geo,
        vmin_delta, vmax_delta,
        title_dl_delta, suptitle,
        cbar_label="Change in annual mean HWI (°C)",
        out_png=os.path.join(OUT_DIR, "gsod_HWI_change_DL_1950-2024_CA_scatter.png"),
        out_pdf=os.path.join(OUT_DIR, "gsod_HWI_change_DL_1950-2024_CA_scatter.pdf"),
    )

    plot_scatter_metric(
        dfm, "slope_dl", "p_dl", ca_geo,
        vmin_trend, vmax_trend,
        title_dl_trend, suptitle,
        cbar_label="Sen's slope of annual HWI (°C per year)",
        out_png=os.path.join(OUT_DIR, "gsod_HWI_trend_DL_1950-2024_CA_scatter.png"),
        out_pdf=os.path.join(OUT_DIR, "gsod_HWI_trend_DL_1950-2024_CA_scatter.pdf"),
        alpha_sig=0.05,
    )


    title_nl_delta = (
        "NL (Nighttime, TN90p on Tmin)\n"
        "ΔHWI = Late (1990–2024) − Early (1950–1984)"
    )
    title_nl_trend = (
        "NL (Nighttime, TN90p on Tmin)\n"
        "Trend of annual HWI (Sen's slope, 1950–2024)"
    )

    plot_scatter_metric(
        dfm, "delta_nl", None, ca_geo,
        vmin_delta, vmax_delta,
        title_nl_delta, suptitle,
        cbar_label="Change in annual mean HWI (°C)",
        out_png=os.path.join(OUT_DIR, "gsod_HWI_change_NL_1950-2024_CA_scatter.png"),
        out_pdf=os.path.join(OUT_DIR, "gsod_HWI_change_NL_1950-2024_CA_scatter.pdf"),
    )

    plot_scatter_metric(
        dfm, "slope_nl", "p_nl", ca_geo,
        vmin_trend, vmax_trend,
        title_nl_trend, suptitle,
        cbar_label="Sen's slope of annual HWI (°C per year)",
        out_png=os.path.join(OUT_DIR, "gsod_HWI_trend_NL_1950-2024_CA_scatter.png"),
        out_pdf=os.path.join(OUT_DIR, "gsod_HWI_trend_NL_1950-2024_CA_scatter.pdf"),
        alpha_sig=0.05,
    )


    title_cl_delta = (
        "CL (Compound: TX90p & TN90p on same day)\n"
        "ΔHWI = Late (1990–2024) − Early (1950–1984)"
    )
    title_cl_trend = (
        "CL (Compound: TX90p & TN90p on same day)\n"
        "Trend of annual HWI (Sen's slope, 1950–2024)"
    )

    plot_scatter_metric(
        dfm, "delta_cl", None, ca_geo,
        vmin_delta, vmax_delta,
        title_cl_delta, suptitle,
        cbar_label="Change in annual mean HWI (°C)",
        out_png=os.path.join(OUT_DIR, "gsod_HWI_change_CL_1950-2024_CA_scatter.png"),
        out_pdf=os.path.join(OUT_DIR, "gsod_HWI_change_CL_1950-2024_CA_scatter.pdf"),
    )

    plot_scatter_metric(
        dfm, "slope_cl", "p_cl", ca_geo,
        vmin_trend, vmax_trend,
        title_cl_trend, suptitle,
        cbar_label="Sen's slope of annual HWI (°C per year)",
        out_png=os.path.join(OUT_DIR, "gsod_HWI_trend_CL_1950-2024_CA_scatter.png"),
        out_pdf=os.path.join(OUT_DIR, "gsod_HWI_trend_CL_1950-2024_CA_scatter.pdf"),
        alpha_sig=0.05,
    )

    print("[OK] 6 张 GSOD HWI 散点图已重新绘制完毕。")


if __name__ == "__main__":
    main()
