import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


OUT_DIR = r"G:\CYX\Analyse\geminifigs"
DIFF_CSV = os.path.join(OUT_DIR, "spatial_difference_data.csv")
CLIPPED_CSV = os.path.join(OUT_DIR, "spatial_difference_data_clipped.csv")
FIG_OUT = os.path.join(OUT_DIR, "Fig_Spatial_Difference_ERA5_GSOD_clipped_solution1.png")

SHP_DIR = r"G:\Central Asia"
SHP_PATH = None

LON_RANGE = (45.0, 96.0)
LAT_RANGE = (33.0, 56.0)

CMAP = "RdBu_r"
ROBUST_Q = 0.98
FIXED_CLIM = {"HWF": None, "HWD": None, "HWI": None}


def pick_shp(shp_dir: str) -> str:
    shp = glob.glob(os.path.join(shp_dir, "*.shp"))
    if not shp:
        shp = glob.glob(os.path.join(shp_dir, "**", "*.shp"), recursive=True)
    if not shp:
        raise FileNotFoundError(f"No shp found under: {shp_dir}")

    def score(p: str) -> int:
        n = os.path.basename(p).lower()
        s = 0
        if "central" in n: s += 2
        if "asia" in n: s += 2
        if "border" in n or "boundary" in n: s += 1
        if "adm" in n: s += 1
        return s

    shp_sorted = sorted(shp, key=score, reverse=True)
    return shp_sorted[0]


def clip_points_to_boundary(df: pd.DataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")


    boundary = boundary.copy()
    boundary["geometry"] = boundary["geometry"].buffer(0)

    region = boundary.unary_union
    mask = pts.geometry.intersects(region)
    return pts.loc[mask].copy(), boundary


def set_gridliner_labels(gl, i: int, j: int):


    if hasattr(gl, "top_labels"):
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (j == 0)
        gl.bottom_labels = (i == 2)
    else:

        if hasattr(gl, "xlabels_top"):
            gl.xlabels_top = False
        if hasattr(gl, "ylabels_right"):
            gl.ylabels_right = False
        if hasattr(gl, "ylabels_left"):
            gl.ylabels_left = (j == 0)
        if hasattr(gl, "xlabels_bottom"):
            gl.xlabels_bottom = (i == 2)


    if hasattr(gl, "x_inline"):
        gl.x_inline = False
    if hasattr(gl, "y_inline"):
        gl.y_inline = False


    if hasattr(gl, "xpadding"):
        gl.xpadding = 6
    if hasattr(gl, "ypadding"):
        gl.ypadding = 6


    if hasattr(gl, "xlabel_style"):
        gl.xlabel_style = {"size": 10}
    if hasattr(gl, "ylabel_style"):
        gl.ylabel_style = {"size": 10}


def plot_3x3(points: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, out_png: str) -> None:
    metrics, types = ["HWF", "HWD", "HWI"], ["CL", "DL", "NL"]

    fig, axes = plt.subplots(
        3, 3, figsize=(18, 14),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    plt.subplots_adjust(wspace=0.08, hspace=0.12)


    clim_row = {}
    for met in metrics:
        fixed = FIXED_CLIM.get(met, None)
        if isinstance(fixed, (int, float)) and fixed and fixed > 0:
            clim_row[met] = (-float(fixed), float(fixed))
        else:
            cols = [f"diff_{met}_{t}" for t in types if f"diff_{met}_{t}" in points.columns]
            vals = points[cols].to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                clim_row[met] = (-1.0, 1.0)
            else:
                lim = float(np.nanquantile(np.abs(vals), ROBUST_Q))
                lim = max(lim, 0.1)
                clim_row[met] = (-lim, lim)

    boundary_plot = boundary.to_crs("EPSG:4326")

    for i, met in enumerate(metrics):
        vmin, vmax = clim_row[met]
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        for j, typ in enumerate(types):
            ax = axes[i, j]
            ax.set_extent([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]])


            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, alpha=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, alpha=0.6)
            boundary_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.9)

            col = f"diff_{met}_{typ}"
            if col in points.columns:
                sub = points.dropna(subset=[col])
                if len(sub) > 0:
                    ax.scatter(
                        sub["lon"], sub["lat"],
                        c=sub[col].values,
                        cmap=CMAP, vmin=vmin, vmax=vmax,
                        s=26,
                        edgecolors="black", linewidth=0.25,
                        transform=ccrs.PlateCarree()
                    )


            if i == 0:
                ax.set_title(typ, fontsize=16, fontweight="bold", pad=8)
            if j == 0:
                ax.text(
                    -0.13, 0.5, met, transform=ax.transAxes,
                    rotation="vertical", va="center", ha="center",
                    fontsize=16, fontweight="bold"
                )


            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.35, linestyle="--")
            set_gridliner_labels(gl, i=i, j=j)


        pos = axes[i, -1].get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
        cb = fig.colorbar(sm, cax=cax, extend="both")
        cb.set_label("Diff (ERA5 − GSOD)", fontsize=12)

    fig.suptitle(
        "Spatial difference of heatwave characteristics (ERA5 − GSOD)\n"
        "Outside-boundary points removed; latitude labels only on left column; longitude labels only on bottom row",
        fontsize=18, y=0.98
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DIFF_CSV):
        raise FileNotFoundError(f"Missing diff csv: {DIFF_CSV}")

    df = pd.read_csv(DIFF_CSV)
    if not {"lon", "lat"}.issubset(df.columns):
        raise KeyError("diff csv must contain columns: lon, lat")

    shp = SHP_PATH if SHP_PATH else pick_shp(SHP_DIR)
    print(f"Using boundary shp: {shp}")
    boundary = gpd.read_file(shp)

    pts_clipped, boundary_fixed = clip_points_to_boundary(df, boundary)
    print(f"Original points: {len(df)}")
    print(f"Clipped points : {len(pts_clipped)} (outside removed, boundary kept)")


    pts_clipped.drop(columns="geometry").to_csv(CLIPPED_CSV, index=False)
    print(f"Clipped CSV saved: {CLIPPED_CSV}")


    plot_3x3(pts_clipped, boundary_fixed, FIG_OUT)

if __name__ == "__main__":
    main()
