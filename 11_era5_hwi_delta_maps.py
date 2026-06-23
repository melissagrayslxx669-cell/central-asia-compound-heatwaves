import os, glob, warnings, gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndlabel
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"


ERA5_DIR  = r"G:\CYX\ERA5\daily"
CA_SHP    = r"G:\Central Asia"
OUT_DIR   = r"G:\CYX\Analyse\figs"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_MIN, YEAR_MAX   = 1950, 2024
BASE_START, BASE_END = 1991, 2020
DOY_WINDOW           = 15

EARLY_START, EARLY_END = 1950, 1984
LATE_START,  LATE_END  = 1990, 2024


LON_MIN, LON_MAX = 46.0, 96.5
LAT_MIN, LAT_MAX = 34.0, 55.7


def _preprocess(ds):
    if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
    if "latitude"  in ds.coords: ds = ds.rename({"latitude":  "lat"})
    keep = []
    if "t2m_daily_max" in ds.data_vars: keep.append("t2m_daily_max")
    if "t2m_daily_min" in ds.data_vars: keep.append("t2m_daily_min")
    return ds[keep] if keep else ds

def open_daily_tmax_tmin():
    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_daily_t2m_*_utc+*.nc")))
    if not files:
        files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_daily_t2m_*.nc")))
    if not files:
        raise FileNotFoundError(f"未在 {ERA5_DIR} 找到 era5_daily_t2m_*.nc")
    print(f"[INFO] 发现日文件 {len(files)} 个（示例前5个）：")
    for s in files[:5]:
        print("   -", s)

    ds = xr.open_mfdataset(
        files, combine="by_coords", preprocess=_preprocess, parallel=True,
        chunks={"time": 365, "lat": 120, "lon": 120}
    )
    if ("t2m_daily_max" not in ds.data_vars) or ("t2m_daily_min" not in ds.data_vars):
        raise KeyError(f"变量缺失，实际包含：{list(ds.data_vars)}")

    tmax = ds["t2m_daily_max"].astype("float32").rename("tmax")
    tmin = ds["t2m_daily_min"].astype("float32").rename("tmin")

    def _clip(da):
        da = da.sel(time=slice(f"{YEAR_MIN}-01-01", f"{YEAR_MAX}-12-31"))
        da = da.sel(time=~((da.time.dt.month==2) & (da.time.dt.day==29)))
        lat = da["lat"].values
        if lat[0] < lat[-1]:
            da = da.sel(lat=slice(LAT_MIN, LAT_MAX))
        else:
            da = da.sel(lat=slice(LAT_MAX, LAT_MIN))
        da = da.sel(lon=slice(LON_MIN, LON_MAX))
        return da

    return _clip(tmax), _clip(tmin)


def compute_q90_DOYwin_streaming(da, base_start=1991, base_end=2020, win=15):
    base = da.sel(time=slice(f"{base_start}-01-01", f"{base_end}-12-31"))
    doy_all  = xr.where(da.time.dt.dayofyear==366, 365, da.time.dt.dayofyear).astype("int16")
    doy_base = xr.where(base.time.dt.dayofyear==366, 365, base.time.dt.dayofyear).astype("int16")

    thr_list = []
    for d in tqdm(range(1, 366), desc="阈值：逐DOY计算（逐步）", ncols=88):
        delta = ((doy_base - d + 182) % 365) - 182
        sel = abs(delta) <= win
        q = base.sel(time=sel).quantile(0.9, dim="time", skipna=True).astype("float32").compute()
        thr_list.append(q)

    thr_doy = xr.concat(thr_list, dim="doy").assign_coords(doy=np.arange(1,366,dtype="int16")).astype("float32")
    thr_sel = thr_doy.sel(doy=doy_all)
    thr_time = xr.DataArray(
        thr_sel.data, coords={"time": da.time, "lat": da.lat, "lon": da.lon},
        dims=("time","lat","lon"), name="q90"
    )
    del thr_list, thr_doy, thr_sel; gc.collect()
    return thr_time


def HWI_per_year_streaming(da, thr, hit_mask):
    years = np.unique(da.time.dt.year.values)
    ny, nlat, nlon = len(years), da.sizes["lat"], da.sizes["lon"]
    arr = np.zeros((ny, nlat, nlon), dtype=np.float32)

    for i, y in enumerate(tqdm(years, desc="HWI：逐年计算（逐步）", ncols=88)):
        sub_da   = da.sel(time=str(int(y)))
        sub_thr  = thr.sel(time=str(int(y)))
        sub_hit  = hit_mask.sel(time=str(int(y))).chunk({"time": -1})
        ex = xr.where(sub_hit, (sub_da - sub_thr), np.nan).astype("float32")
        mean_ex = ex.mean("time", skipna=True).fillna(0.0).compute()
        arr[i,:,:] = mean_ex.values
        del sub_da, sub_thr, sub_hit, ex, mean_ex; gc.collect()

    hwi = xr.DataArray(arr, coords={"time": years.astype(int), "lat": da.lat, "lon": da.lon},
                       dims=("time","lat","lon"), name="HWI")
    return hwi

def HWI_per_year_streaming_CL(tmax, tx90, tmin, tn90, hit_cl):
    years = np.unique(tmax.time.dt.year.values)
    ny, nlat, nlon = len(years), tmax.sizes["lat"], tmax.sizes["lon"]
    arr = np.zeros((ny, nlat, nlon), dtype=np.float32)

    for i, y in enumerate(tqdm(years, desc="HWI-CL：逐年计算（逐步）", ncols=88)):
        tmax_y = tmax.sel(time=str(int(y)))
        tmin_y = tmin.sel(time=str(int(y)))
        tx90_y = tx90.sel(time=str(int(y)))
        tn90_y = tn90.sel(time=str(int(y)))
        hit_y  = hit_cl.sel(time=str(int(y))).chunk({"time": -1})

        ex_cl = xr.where(hit_y, ((tmax_y - tx90_y) + (tmin_y - tn90_y))/2.0, np.nan).astype("float32")
        mean_ex = ex_cl.mean("time", skipna=True).fillna(0.0).compute()
        arr[i,:,:] = mean_ex.values
        del tmax_y, tmin_y, tx90_y, tn90_y, hit_y, ex_cl, mean_ex; gc.collect()

    hwi = xr.DataArray(arr, coords={"time": years.astype(int), "lat": tmax.lat, "lon": tmax.lon},
                       dims=("time","lat","lon"), name="HWI_CL")
    return hwi


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


def plot_map(delta_da, ca_boundary, vmin, vmax, title, out_png, out_pdf, unit_label="°C"):
    Z = np.array(delta_da.values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9.2, 6.9), dpi=220)
    ca_boundary.plot(ax=ax, color="none", edgecolor="k", linewidth=0.6)

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    pcm = ax.pcolormesh(delta_da["lon"].values, delta_da["lat"].values, Z,
                        cmap="RdBu_r", norm=norm, shading="auto")

    ax.set_xlim([float(delta_da["lon"].min()), float(delta_da["lon"].max())])
    ax.set_ylim([float(delta_da["lat"].min()), float(delta_da["lat"].max())])
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(title, pad=10)


    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.022, pos.height])
    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label(f"{unit_label} (late − early)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


print("Step 1/12 打开 ERA5 日 Tmax/Tmin …")
tmax, tmin = open_daily_tmax_tmin()

print("Step 2/12 计算 TX90p（Tmax） …")
tx90 = compute_q90_DOYwin_streaming(tmax, BASE_START, BASE_END, DOY_WINDOW)

print("Step 3/12 计算 TN90p（Tmin） …")
tn90 = compute_q90_DOYwin_streaming(tmin, BASE_START, BASE_END, DOY_WINDOW)

print("Step 4/12 构造命中布尔场：DL, NL, CL …")
hit_dl = xr.where(tmax > tx90, True, False)
hit_nl = xr.where(tmin > tn90, True, False)
hit_cl = xr.where((tmax > tx90) & (tmin > tn90), True, False)

print("Step 5/12 DL 年均强度（逐年） …")
hwi_dl = HWI_per_year_streaming(tmax, tx90, hit_dl); del hit_dl; gc.collect()
print("Step 6/12 NL 年均强度（逐年） …")
hwi_nl = HWI_per_year_streaming(tmin, tn90, hit_nl); del hit_nl; gc.collect()
print("Step 7/12 CL 年均强度（逐年） …")
hwi_cl = HWI_per_year_streaming_CL(tmax, tx90, tmin, tn90, hit_cl); del hit_cl; gc.collect()


del tmax, tmin, tx90, tn90; gc.collect()

print("Step 8/12 计算两阶段年均（°C） …")
mean_dl_early = hwi_dl.sel(time=slice(EARLY_START, EARLY_END)).mean("time", skipna=True).astype("float32")
mean_dl_late  = hwi_dl.sel(time=slice(LATE_START,  LATE_END )).mean("time", skipna=True).astype("float32")
mean_nl_early = hwi_nl.sel(time=slice(EARLY_START, EARLY_END)).mean("time", skipna=True).astype("float32")
mean_nl_late  = hwi_nl.sel(time=slice(LATE_START,  LATE_END )).mean("time", skipna=True).astype("float32")
mean_cl_early = hwi_cl.sel(time=slice(EARLY_START, EARLY_END)).mean("time", skipna=True).astype("float32")
mean_cl_late  = hwi_cl.sel(time=slice(LATE_START,  LATE_END )).mean("time", skipna=True).astype("float32")

print("Step 9/12 计算年均强度变化（late − early，°C） …")
d_dl = (mean_dl_late - mean_dl_early).rename("ΔHWI_DL")
d_nl = (mean_nl_late - mean_nl_early).rename("ΔHWI_NL")
d_cl = (mean_cl_late - mean_cl_early).rename("ΔHWI_CL")


del hwi_dl, hwi_nl, hwi_cl
del mean_dl_early, mean_dl_late, mean_nl_early, mean_nl_late, mean_cl_early, mean_cl_late
gc.collect()

print("Step 10/12 掩膜 …")
ca_poly, ca_boundary = load_ca_polygon(CA_SHP)
mask = mask_outside_polygon(d_dl["lat"], d_dl["lon"], ca_poly)
d_dl_m = d_dl.where(mask); d_nl_m = d_nl.where(mask); d_cl_m = d_cl.where(mask)

print("Step 11/12 统一色标（对称） …")
vals_all = np.concatenate([
    np.abs(d_dl_m.values[np.isfinite(d_dl_m.values)]),
    np.abs(d_nl_m.values[np.isfinite(d_nl_m.values)]),
    np.abs(d_cl_m.values[np.isfinite(d_cl_m.values)])
])
if vals_all.size == 0:
    vmax = 0.5
else:
    vmax = float(np.percentile(vals_all, 98))
    if vmax == 0: vmax = 0.5
vmin = -vmax

print("Step 12/12 绘图输出 …")
title_base = "ERA5 — Change in Annual Mean Heatwave Intensity (HWI)\nLate (1990–2024) − Early (1950–1984), UTC+05:00"

plot_map(
    d_dl_m, ca_boundary, vmin, vmax,
    title=f"{title_base}\nDL (Daytime, mean(Tmax − TX90p) over HW days)",
    out_png=os.path.join(OUT_DIR, "era5_HWI_change_DL_1950-2024_CA.png"),
    out_pdf=os.path.join(OUT_DIR, "era5_HWI_change_DL_1950-2024_CA.pdf"),
    unit_label="°C",
)

plot_map(
    d_nl_m, ca_boundary, vmin, vmax,
    title=f"{title_base}\nNL (Nighttime, mean(Tmin − TN90p) over HW nights)",
    out_png=os.path.join(OUT_DIR, "era5_HWI_change_NL_1950-2024_CA.png"),
    out_pdf=os.path.join(OUT_DIR, "era5_HWI_change_NL_1950-2024_CA.pdf"),
    unit_label="°C",
)

plot_map(
    d_cl_m, ca_boundary, vmin, vmax,
    title=f"{title_base}\nCL (Compound: mean(((Tmax−TX90p)+(Tmin−TN90p))/2) over CL days)",
    out_png=os.path.join(OUT_DIR, "era5_HWI_change_CL_1950-2024_CA.png"),
    out_pdf=os.path.join(OUT_DIR, "era5_HWI_change_CL_1950-2024_CA.pdf"),
    unit_label="°C",
)

print("[OK] 三张 HWI 图已输出到：", OUT_DIR)
