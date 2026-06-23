import os, glob, warnings, gc
import numpy as np
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")


ERA5_DIR  = r"G:\CYX\ERA5\daily"
OUT_DIR   = r"G:\CYX\ERA5\clim"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(
    OUT_DIR,
    "era5_q90_Tmax_Tmean_Tmin_DOYwin15_1991-2020_CA.nc"
)

YEAR_MIN, YEAR_MAX   = 1950, 2024
BASE_START, BASE_END = 1991, 2020
DOY_WINDOW           = 15


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
    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_daily_t2m_*_utc+*.nc")))
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


def compute_q90_DOY_clim(da, base_start=1991, base_end=2020, win=15):
    base = da.sel(time=slice(f"{base_start}-01-01", f"{base_end}-12-31"))


    doy_base = xr.where(
        base.time.dt.dayofyear == 366,
        365,
        base.time.dt.dayofyear
    ).astype("int16")

    thr_list = []
    for d in tqdm(range(1, 366), desc="阈值：逐 DOY 计算 q90", ncols=88):

        delta = ((doy_base - d + 182) % 365) - 182
        sel = abs(delta) <= win
        q = base.sel(time=sel).quantile(
            0.9,
            dim="time",
            skipna=True,
        ).astype("float32").compute()
        thr_list.append(q)

    thr_doy = xr.concat(thr_list, dim="doy").assign_coords(
        doy=np.arange(1, 366, dtype="int16")
    ).astype("float32")


    thr_doy.name = "q90"
    thr_doy.attrs["description"] = "90th percentile (DOY±%d, %d–%d)" % (win, base_start, base_end)
    thr_doy.attrs["units"] = "degC"

    del thr_list, base, doy_base
    gc.collect()
    return thr_doy


def main():
    print("Step 1/4 打开 ERA5 日 Tmax/Tmin …")
    tmax, tmin = open_daily_tmax_tmin()

    print("Step 2/4 计算 Tmax 的 TX90p（DOY×lat×lon） …")
    q90_tmax = compute_q90_DOY_clim(tmax, BASE_START, BASE_END, DOY_WINDOW)
    q90_tmax = q90_tmax.rename("tx90_tmax")

    print("Step 3/4 计算 Tmin 的 TN90p（DOY×lat×lon） …")
    q90_tmin = compute_q90_DOY_clim(tmin, BASE_START, BASE_END, DOY_WINDOW)
    q90_tmin = q90_tmin.rename("tn90_tmin")

    print("Step 4/4 计算 Tmean 的 TX90p（Tmean = (Tmax+Tmin)/2） …")
    tmean = ((tmax + tmin) * 0.5).astype("float32").rename("tmean")
    q90_tmean = compute_q90_DOY_clim(tmean, BASE_START, BASE_END, DOY_WINDOW)
    q90_tmean = q90_tmean.rename("tx90_tmean")


    ds_out = xr.Dataset(
        {
            "tx90_tmax": q90_tmax,
            "tn90_tmin": q90_tmin,
            "tx90_tmean": q90_tmean,
        }
    )

    ds_out.attrs["note"] = (
        "Q90 climatology for Tmax/Tmin/Tmean, "
        "computed with DOY±%d, %d–%d, ERA5 UTC+5, clipped to Central Asia."
        % (DOY_WINDOW, BASE_START, BASE_END)
    )


    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_out.data_vars
    }

    print(f"[SAVE] 写出阈值文件：{OUT_FILE}")
    ds_out.to_netcdf(OUT_FILE, encoding=encoding)
    print("[OK] 预计算完成，以后直接调用这个文件即可。")


if __name__ == "__main__":
    main()
