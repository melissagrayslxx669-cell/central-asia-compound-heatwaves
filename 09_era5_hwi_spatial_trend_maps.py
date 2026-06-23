import os, glob, warnings, gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndlabel
from scipy.stats import norm
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"


ERA5_DIR  = r"G:\CYX\ERA5\daily"
Q90_FILE  = r"G:\CYX\ERA5\clim\era5_q90_Tmax_Tmean_Tmin_DOYwin15_1991-2020_CA.nc"
CA_SHP    = r"G:\Central Asia"
OUT_DIR   = r"G:\CYX\Analyse\figs"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_MIN, YEAR_MAX   = 1950, 2024
MIN_HW_LEN           = 3

EARLY_START, EARLY_END = 1950, 1984
LATE_START,  LATE_END  = 1990, 2024

LON_MIN, LON_MAX = 46.0, 96.5
LAT_MIN, LAT_MAX = 34.0, 55.7


def _preprocess(ds):
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    keep = []
    if "t2m_daily_max" in ds.data_vars:
        keep.append("t2m_daily_max")
    if "t2m_daily_min" in ds.data_vars:
        keep.append("t2m_daily_min")
    return ds[keep] if keep else ds


def open_daily_tmax_tmin():
    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_daily_t2m_*_utc+5.nc")))
    if not files:
        files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_daily_t2m_*.nc")))
    if not files:
        raise FileNotFoundError(f"未在 {ERA5_DIR} 找到 era5_daily_t2m_*.nc")

    print(f"[INFO] 发现日文件 {len(files)} 个（示例前5个）：")
    for s in files[:5]:
        print("   -", s)

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=_preprocess,
        parallel=True,
        chunks={"time": 365, "lat": 120, "lon": 120},
    )

    if ("t2m_daily_max" not in ds.data_vars) or ("t2m_daily_min" not in ds.data_vars):
        raise KeyError(f"变量缺失，实际包含：{list(ds.data_vars)}")

    tmax = ds["t2m_daily_max"].astype("float32").rename("tmax")
    tmin = ds["t2m_daily_min"].astype("float32").rename("tmin")

    def _clip(da):
        da = da.sel(time=slice(f"{YEAR_MIN}-01-01", f"{YEAR_MAX}-12-31"))
        da = da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
        lat = da["lat"].values
        if lat[0] < lat[-1]:
            da = da.sel(lat=slice(LAT_MIN, LAT_MAX))
        else:
            da = da.sel(lat=slice(LAT_MAX, LAT_MIN))
        da = da.sel(lon=slice(LON_MIN, LON_MAX))
        return da

    return _clip(tmax), _clip(tmin)


def load_q90_from_file(template_da, varname):
    if not os.path.exists(Q90_FILE):
        raise FileNotFoundError(f"阈值文件不存在：{Q90_FILE}")
    ds_q = xr.open_dataset(Q90_FILE)
    if varname not in ds_q.data_vars:
        raise KeyError(f"在 {Q90_FILE} 中未找到变量 {varname}")
    clim = ds_q[varname]

    doy_all = xr.where(
        template_da.time.dt.dayofyear == 366,
        365,
        template_da.time.dt.dayofyear
    ).astype("int16")

    thr_sel = clim.sel(doy=doy_all)
    thr_time = xr.DataArray(
        thr_sel.data,
        coords={"time": template_da.time,
                "lat": template_da["lat"],
                "lon": template_da["lon"]},
        dims=("time", "lat", "lon"),
        name=varname
    )
    ds_q.close()
    return thr_time


def hw_mask_1d(arr_bool, min_len=3):
    lab, nlab = ndlabel(arr_bool.astype(np.uint8))
    if nlab == 0:
        return np.zeros_like(arr_bool, dtype=bool)
    mask = np.zeros_like(arr_bool, dtype=bool)
    for k in range(1, nlab + 1):
        idx = (lab == k)
        if idx.sum() >= min_len:
            mask[idx] = True
    return mask


def HWI_per_year_DL(tmax, tx90, hit_dl, min_len=3):
    years = np.unique(tmax.time.dt.year.values)
    ny, nlat, nlon = len(years), tmax.sizes["lat"], tmax.sizes["lon"]
    arr = np.zeros((ny, nlat, nlon), dtype=np.float32)

    for i, y in enumerate(tqdm(years, desc="HWI-DL per year", ncols=88)):
        tmax_y = tmax.sel(time=str(int(y))).chunk({"time": -1})
        tx90_y = tx90.sel(time=str(int(y))).chunk({"time": -1})
        hit_y  = hit_dl.sel(time=str(int(y))).chunk({"time": -1})

        hw_mask = xr.apply_ufunc(
            hw_mask_1d, hit_y,
            kwargs={"min_len": min_len},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[bool],
        ).compute()

        ex = (tmax_y - tx90_y).where(hw_mask)
        ex_mean = ex.mean("time", skipna=True).fillna(0.0).compute()
        arr[i, :, :] = ex_mean.values.astype(np.float32)

        del tmax_y, tx90_y, hit_y, hw_mask, ex, ex_mean
        gc.collect()

    hwi = xr.DataArray(
        arr,
        coords={"time": years.astype(int), "lat": tmax["lat"], "lon": tmax["lon"]},
        dims=("time", "lat", "lon"),
        name="HWI_DL",
    )
    return hwi


def HWI_per_year_NL(tmin, tn90, hit_nl, min_len=3):
    years = np.unique(tmin.time.dt.year.values)
    ny, nlat, nlon = len(years), tmin.sizes["lat"], tmin.sizes["lon"]
    arr = np.zeros((ny, nlat, nlon), dtype=np.float32)

    for i, y in enumerate(tqdm(years, desc="HWI-NL per year", ncols=88)):
        tmin_y = tmin.sel(time=str(int(y))).chunk({"time": -1})
        tn90_y = tn90.sel(time=str(int(y))).chunk({"time": -1})
        hit_y  = hit_nl.sel(time=str(int(y))).chunk({"time": -1})

        hw_mask = xr.apply_ufunc(
            hw_mask_1d, hit_y,
            kwargs={"min_len": min_len},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[bool],
        ).compute()

        ex = (tmin_y - tn90_y).where(hw_mask)
        ex_mean = ex.mean("time", skipna=True).fillna(0.0).compute()
        arr[i, :, :] = ex_mean.values.astype(np.float32)

        del tmin_y, tn90_y, hit_y, hw_mask, ex, ex_mean
        gc.collect()

    hwi = xr.DataArray(
        arr,
        coords={"time": years.astype(int), "lat": tmin["lat"], "lon": tmin["lon"]},
        dims=("time", "lat", "lon"),
        name="HWI_NL",
    )
    return hwi


def HWI_per_year_CL(tmax, tx90, tmin, tn90, hit_cl, min_len=3):
    years = np.unique(tmax.time.dt.year.values)
    ny, nlat, nlon = len(years), tmax.sizes["lat"], tmax.sizes["lon"]
    arr = np.zeros((ny, nlat, nlon), dtype=np.float32)

    for i, y in enumerate(tqdm(years, desc="HWI-CL per year", ncols=88)):
        tmax_y = tmax.sel(time=str(int(y))).chunk({"time": -1})
        tmin_y = tmin.sel(time=str(int(y))).chunk({"time": -1})
        tx90_y = tx90.sel(time=str(int(y))).chunk({"time": -1})
        tn90_y = tn90.sel(time=str(int(y))).chunk({"time": -1})
        hit_y  = hit_cl.sel(time=str(int(y))).chunk({"time": -1})

        hw_mask = xr.apply_ufunc(
            hw_mask_1d, hit_y,
            kwargs={"min_len": min_len},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[bool],
        ).compute()


        ex_day   = (tmax_y - tx90_y)
        ex_night = (tmin_y - tn90_y)
        ex = ((ex_day + ex_night) / 2.0).where(hw_mask)

        ex_mean = ex.mean("time", skipna=True).fillna(0.0).compute()
        arr[i, :, :] = ex_mean.values.astype(np.float32)

        del tmax_y, tmin_y, tx90_y, tn90_y, hit_y, hw_mask, ex, ex_day, ex_night, ex_mean
        gc.collect()

    hwi = xr.DataArray(
        arr,
        coords={"time": years.astype(int), "lat": tmax["lat"], "lon": tmax["lon"]},
        dims=("time", "lat", "lon"),
        name="HWI_CL",
    )
    return hwi


def calc_sen_mk_1d(y, x):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(y)
    y = y[mask]; x = x[mask]
    n = len(y)
    if n < 3:
        return np.nan, np.nan

    S = 0.0
    slopes_list = []
    for i in range(n - 1):
        dy = y[i + 1:] - y[i]
        dx = x[i + 1:] - x[i]
        slopes_i = dy / dx
        slopes_list.append(slopes_i)
        S += np.sum(np.sign(dy))

    slopes = np.concatenate(slopes_list)
    slope = np.median(slopes)

    uniq, cnts = np.unique(y, return_counts=True)
    tie_term = np.sum(cnts[cnts > 1] * (cnts[cnts > 1] - 1) * (2 * cnts[cnts > 1] + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s == 0:
        return slope, np.nan

    if S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * norm.sf(abs(z))
    return slope, p


def mk_sen_trend_3d(da):
    years = da["time"].values.astype(float)
    data = da.values
    nt, nlat, nlon = data.shape
    ngrid = nlat * nlon

    slope_flat = np.full(ngrid, np.nan, dtype=np.float32)
    p_flat     = np.full(ngrid, np.nan, dtype=np.float32)
    data_2d = data.reshape(nt, ngrid)

    print("趋势：MK + Sen 逐格点计算 …")
    for idx in tqdm(range(ngrid), desc="MK+Sen", ncols=88):
        ts = data_2d[:, idx]
        if np.all(~np.isfinite(ts)):
            continue
        s, p = calc_sen_mk_1d(ts, years)
        slope_flat[idx] = np.float32(s) if np.isfinite(s) else np.nan
        p_flat[idx]     = np.float32(p) if np.isfinite(p) else np.nan

    slope = slope_flat.reshape(nlat, nlon)
    pval  = p_flat.reshape(nlat, nlon)

    slope_da = xr.DataArray(slope, coords={"lat": da["lat"], "lon": da["lon"]},
                            dims=("lat", "lon"), name="slope")
    p_da = xr.DataArray(pval, coords={"lat": da["lat"], "lon": da["lon"]},
                        dims=("lat", "lon"), name="p_value")
    return slope_da, p_da


def load_ca_polygon(shp_dir):
    shp_files = sorted(glob.glob(os.path.join(shp_dir, "*.shp")))
    if not shp_files:
        raise FileNotFoundError(f"在 {shp_dir} 未找到 *.shp")
    gdf = gpd.read_file(shp_files[0])
    gdf = gdf.to_crs(epsg=4326) if gdf.crs else gdf.set_crs(epsg=4326)
    return unary_union(gdf.geometry), gdf.boundary


def mask_outside_polygon(lat, lon, polygon):
    lon2, lat2 = np.meshgrid(lon.values, lat.values)
    pts = [Point(xy) for xy in zip(lon2.ravel(), lat2.ravel())]
    inside = np.array([polygon.contains(p) for p in tqdm(pts, desc="生成区域掩膜", ncols=88)], dtype=bool)
    return inside.reshape(lat.shape[0], lon.shape[0])


def plot_diff_single(delta_da, ca_boundary,
                     vmin, vmax,
                     title, cbar_label,
                     out_png, out_pdf):
    lon = delta_da["lon"].values
    lat = delta_da["lat"].values

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    fig.suptitle("ERA5 Heatwave Intensity (HWI), UTC+05:00", fontsize=12, y=0.97)

    ca_boundary.plot(ax=ax, color="none", edgecolor="k", linewidth=0.6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    pcm = ax.pcolormesh(lon, lat, delta_da.values.astype(np.float32),
                        cmap="RdBu_r", norm=norm, shading="auto")

    ax.set_title(title, pad=8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim([float(lon.min()), float(lon.max())])
    ax.set_ylim([float(lat.min()), float(lat.max())])

    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.025, pos.height])
    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label(cbar_label)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_trend_single(slope_da, p_da, ca_boundary,
                      vmin, vmax,
                      title, cbar_label,
                      alpha_sig,
                      out_png, out_pdf):
    lon = slope_da["lon"].values
    lat = slope_da["lat"].values

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    fig.suptitle("ERA5 Heatwave Intensity (HWI), UTC+05:00", fontsize=12, y=0.97)

    ca_boundary.plot(ax=ax, color="none", edgecolor="k", linewidth=0.6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    pcm = ax.pcolormesh(lon, lat, slope_da.values.astype(np.float32),
                        cmap="RdBu_r", norm=norm, shading="auto")


    if p_da is not None:
        P = p_da.values
        Z = slope_da.values
        sig = (P < alpha_sig) & np.isfinite(P) & np.isfinite(Z)
        if np.any(sig):
            lon2, lat2 = np.meshgrid(lon, lat)
            ax.scatter(lon2[sig], lat2[sig],
                       s=4, c="k", marker=".", linewidths=0, alpha=0.7,
                       label=f"MK p < {alpha_sig}")
            ax.legend(loc="lower left", fontsize=7, frameon=False)

    ax.set_title(title, pad=8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim([float(lon.min()), float(lon.max())])
    ax.set_ylim([float(lat.min()), float(lat.max())])

    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.025, pos.height])
    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label(cbar_label)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Step 1/7 打开 ERA5 日 Tmax/Tmin …")
    tmax, tmin = open_daily_tmax_tmin()

    print("Step 2/7 从阈值文件读取 TX90p(Tmax) & TN90p(Tmin) …")
    tx90 = load_q90_from_file(tmax, "tx90_tmax")
    tn90 = load_q90_from_file(tmin, "tn90_tmin")

    print("Step 3/7 构造命中布尔场：DL, NL, CL …")
    hit_dl = xr.where(tmax > tx90, True, False)
    hit_nl = xr.where(tmin > tn90, True, False)
    hit_cl = xr.where((tmax > tx90) & (tmin > tn90), True, False)

    print("Step 4/7 逐年 HWI（DL/NL/CL） …")
    print("  -> DL")
    hwi_dl = HWI_per_year_DL(tmax, tx90, hit_dl, MIN_HW_LEN)
    print("  -> NL")
    hwi_nl = HWI_per_year_NL(tmin, tn90, hit_nl, MIN_HW_LEN)
    print("  -> CL")
    hwi_cl = HWI_per_year_CL(tmax, tx90, tmin, tn90, hit_cl, MIN_HW_LEN)


    del tmax, tmin, tx90, tn90, hit_dl, hit_nl, hit_cl
    gc.collect()

    print("Step 5/7 计算两阶段年均差（late−early） …")
    def early_late_mean(hwi):
        early = hwi.sel(time=slice(EARLY_START, EARLY_END)).mean("time", skipna=True)
        late  = hwi.sel(time=slice(LATE_START,  LATE_END)).mean("time", skipna=True)
        return early, late

    early_dl, late_dl = early_late_mean(hwi_dl)
    early_nl, late_nl = early_late_mean(hwi_nl)
    early_cl, late_cl = early_late_mean(hwi_cl)

    d_dl = (late_dl - early_dl).astype("float32")
    d_nl = (late_nl - early_nl).astype("float32")
    d_cl = (late_cl - early_cl).astype("float32")

    print("Step 6/7 计算 Sen 斜率 + MK 显著性（DL/NL/CL） …")
    slope_dl, p_dl = mk_sen_trend_3d(hwi_dl)
    slope_nl, p_nl = mk_sen_trend_3d(hwi_nl)
    slope_cl, p_cl = mk_sen_trend_3d(hwi_cl)

    print("Step 7/7 掩膜、统一色标并绘制 6 张图 …")
    ca_poly, ca_boundary = load_ca_polygon(CA_SHP)
    mask = mask_outside_polygon(d_dl["lat"], d_dl["lon"], ca_poly)


    d_dl_m = d_dl.where(mask); d_nl_m = d_nl.where(mask); d_cl_m = d_cl.where(mask)
    slope_dl_m = slope_dl.where(mask); slope_nl_m = slope_nl.where(mask); slope_cl_m = slope_cl.where(mask)
    p_dl_m = p_dl.where(mask); p_nl_m = p_nl.where(mask); p_cl_m = p_cl.where(mask)


    vals_diff = np.concatenate([
        np.abs(d_dl_m.values[np.isfinite(d_dl_m.values)]),
        np.abs(d_nl_m.values[np.isfinite(d_nl_m.values)]),
        np.abs(d_cl_m.values[np.isfinite(d_cl_m.values)])
    ])
    if vals_diff.size == 0:
        vmax_diff = 1.0
    else:
        vmax_diff = float(np.percentile(vals_diff, 98))
        if vmax_diff == 0:
            vmax_diff = 1.0
    vmin_diff = -vmax_diff


    vals_trend = np.concatenate([
        np.abs(slope_dl_m.values[np.isfinite(slope_dl_m.values)]),
        np.abs(slope_nl_m.values[np.isfinite(slope_nl_m.values)]),
        np.abs(slope_cl_m.values[np.isfinite(slope_cl_m.values)])
    ])
    if vals_trend.size == 0:
        vmax_trend = 0.1
    else:
        vmax_trend = float(np.percentile(vals_trend, 98))
        if vmax_trend == 0:
            vmax_trend = 0.1
    vmin_trend = -vmax_trend


    title_dl_diff = "DL (Daytime, TX90p on Tmax)\nΔHWI = Late (1990–2024) − Early (1950–1984)"
    title_dl_trend = "DL (Daytime, TX90p on Tmax)\nTrend of annual HWI (Sen's slope, 1950–2024)"

    plot_diff_single(
        d_dl_m, ca_boundary,
        vmin_diff, vmax_diff,
        title_dl_diff,
        cbar_label="Change in annual mean HWI (°C)",
        out_png=os.path.join(OUT_DIR, "era5_HWI_change_DL_1950-2024_CA_single.png"),
        out_pdf=os.path.join(OUT_DIR, "era5_HWI_change_DL_1950-2024_CA_single.pdf"),
    )

    plot_trend_single(
        slope_dl_m, p_dl_m, ca_boundary,
        vmin_trend, vmax_trend,
        title_dl_trend,
        cbar_label="Sen's slope of annual HWI (°C per year)",
        alpha_sig=0.05,
        out_png=os.path.join(OUT_DIR, "era5_HWI_trend_DL_1950-2024_CA_single.png"),
        out_pdf=os.path.join(OUT_DIR, "era5_HWI_trend_DL_1950-2024_CA_single.pdf"),
    )


    title_nl_diff = "NL (Nighttime, TN90p on Tmin)\nΔHWI = Late (1990–2024) − Early (1950–1984)"
    title_nl_trend = "NL (Nighttime, TN90p on Tmin)\nTrend of annual HWI (Sen's slope, 1950–2024)"

    plot_diff_single(
        d_nl_m, ca_boundary,
        vmin_diff, vmax_diff,
        title_nl_diff,
        cbar_label="Change in annual mean HWI (°C)",
        out_png=os.path.join(OUT_DIR, "era5_HWI_change_NL_1950-2024_CA_single.png"),
        out_pdf=os.path.join(OUT_DIR, "era5_HWI_change_NL_1950-2024_CA_single.pdf"),
    )

    plot_trend_single(
        slope_nl_m, p_nl_m, ca_boundary,
        vmin_trend, vmax_trend,
        title_nl_trend,
        cbar_label="Sen's slope of annual HWI (°C per year)",
        alpha_sig=0.05,
        out_png=os.path.join(OUT_DIR, "era5_HWI_trend_NL_1950-2024_CA_single.png"),
        out_pdf=os.path.join(OUT_DIR, "era5_HWI_trend_NL_1950-2024_CA_single.pdf"),
    )


    title_cl_diff = "CL (Compound: TX90p & TN90p on same day)\nΔHWI = Late (1990–2024) − Early (1950–1984)"
    title_cl_trend = "CL (Compound: TX90p & TN90p on same day)\nTrend of annual HWI (Sen's slope, 1950–2024)"

    plot_diff_single(
        d_cl_m, ca_boundary,
        vmin_diff, vmax_diff,
        title_cl_diff,
        cbar_label="Change in annual mean HWI (°C)",
        out_png=os.path.join(OUT_DIR, "era5_HWI_change_CL_1950-2024_CA_single.png"),
        out_pdf=os.path.join(OUT_DIR, "era5_HWI_change_CL_1950-2024_CA_single.pdf"),
    )

    plot_trend_single(
        slope_cl_m, p_cl_m, ca_boundary,
        vmin_trend, vmax_trend,
        title_cl_trend,
        cbar_label="Sen's slope of annual HWI (°C per year)",
        alpha_sig=0.05,
        out_png=os.path.join(OUT_DIR, "era5_HWI_trend_CL_1950-2024_CA_single.png"),
        out_pdf=os.path.join(OUT_DIR, "era5_HWI_trend_CL_1950-2024_CA_single.pdf"),
    )

    print("[OK] HWI 六张图已输出到：", OUT_DIR)


if __name__ == "__main__":
    main()
