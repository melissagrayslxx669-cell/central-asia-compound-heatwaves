import os
import re
import glob
import warnings
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from netCDF4 import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Transformer


ERA5_DAILY_DIR = r"G:\CYX\ERA5\daily"
GSOD_DIR = r"G:\CYX\Analyse\normalized_by_station"


LULC6_DIR = r"G:\CYX\MODIS\2001to2024\24tif reclass"
LULC6_GLOB = "*_LULC6.tif"


OUT_DIR = r"G:/CYX/MODIS/2001to2024/python figs"
os.makedirs(OUT_DIR, exist_ok=True)


CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ERA5_LULC_CACHE_DIR = os.path.join(CACHE_DIR, "era5_lulc_grid_by_year")
os.makedirs(ERA5_LULC_CACHE_DIR, exist_ok=True)


BASELINE_START, BASELINE_END = 1991, 2020
YEAR_START, YEAR_END = 2001, 2024
MIN_EVENT_LEN = 3
MIN_BASELINE_YEARS_STATION = 5


PURGE_OLD = False


FORCE_REBUILD_ERA5_LULC_CACHE = False


FORCE_REBUILD_THRESHOLD_CACHE = False


THR_TARGET_MB = 220


CL_EXCEED_MODE = "mean"
CL_WEIGHT_DAY = 0.5
CL_WEIGHT_NIGHT = 0.5


BAR_WIDTH = 0.34
COLOR_ERA5 = "#DD8452"
COLOR_GSOD = "#4C72B0"


LULC6_ORDER = [1, 2, 3, 4, 5, 6]
LULC6_LABELS = {
    1: "Cropland",
    2: "Forest",
    3: "Grassland",
    4: "Water",
    5: "Built-up",
    6: "Other",
}
HW_TYPES = ["DL", "NL", "CL"]
INDICATORS = ["HWF", "HWD", "HWI"]


Y_AXIS_LIMITS = {
    "HWF": 20,
    "HWD": 6,
    "HWI": 4,
}


THR_FILE = os.path.join(
    CACHE_DIR,
    f"era5_tx90_tn90_baseline_{BASELINE_START}_{BASELINE_END}.nc"
)

ERA5_FILE_INDEX_CSV = os.path.join(CACHE_DIR, f"era5_file_index_{YEAR_START}_{YEAR_END}.csv")
GSOD_FILE_INDEX_CSV = os.path.join(CACHE_DIR, "gsod_file_index.csv")
LULC_YEAR_INDEX_CSV = os.path.join(CACHE_DIR, f"lulc6_year_index_{YEAR_START}_{YEAR_END}.csv")
GSOD_STATION_META_CSV = os.path.join(CACHE_DIR, "gsod_station_meta.csv")
GSOD_STATION_LULC_YEAR_CSV = os.path.join(CACHE_DIR, f"gsod_station_lulc_year_{YEAR_START}_{YEAR_END}.csv")
ERA5_VALID_CELLS_CSV = os.path.join(CACHE_DIR, f"ERA5_dynamicLULC6_valid_cells_by_year_{YEAR_START}_{YEAR_END}.csv")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_remove(path: str):
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except Exception:
        bak = f"{path}.bak_{_timestamp()}"
        os.rename(path, bak)
        print(f"[WARN] Could not delete {path}, renamed to {bak}")


def purge_outputs():
    targets = [
        THR_FILE,
        ERA5_FILE_INDEX_CSV,
        GSOD_FILE_INDEX_CSV,
        LULC_YEAR_INDEX_CSV,
        GSOD_STATION_META_CSV,
        GSOD_STATION_LULC_YEAR_CSV,
        ERA5_VALID_CELLS_CSV,
    ]

    for ind in INDICATORS:
        targets += [
            os.path.join(OUT_DIR, f"ERA5_dynamicLULC6_{ind}_by_year_{YEAR_START}_{YEAR_END}.csv"),
            os.path.join(OUT_DIR, f"GSOD_dynamicLULC6_{ind}_by_year_{YEAR_START}_{YEAR_END}.csv"),
            os.path.join(OUT_DIR, f"ERA5_dynamicLULC6_{ind}_mean_std_{YEAR_START}_{YEAR_END}.csv"),
            os.path.join(OUT_DIR, f"GSOD_dynamicLULC6_{ind}_mean_std_{YEAR_START}_{YEAR_END}.csv"),
            os.path.join(OUT_DIR, f"Fig_{ind}_dynamicLULC6_6x3_ERA5_vs_GSOD_{YEAR_START}_{YEAR_END}.png"),
        ]

    for fp in glob.glob(os.path.join(ERA5_LULC_CACHE_DIR, "*.npz")):
        targets.append(fp)

    print("[INFO] Purging old outputs/caches ...")
    for p in targets:
        safe_remove(p)
    print("[DONE] Purge completed.")


def purge_era5_lulc_cache():
    files = glob.glob(os.path.join(ERA5_LULC_CACHE_DIR, "*.npz"))
    if files:
        print(f"[INFO] Purging old ERA5-LULC grid cache: {len(files)} files")
        for fp in files:
            safe_remove(fp)
        print("[DONE] ERA5-LULC grid cache cleared.")


def count_lulc_codes(arr: np.ndarray) -> Dict[int, int]:
    arr = np.asarray(arr)
    return {code: int(np.sum(arr == code)) for code in LULC6_ORDER}


def ensure_datetime(ds: xr.Dataset) -> xr.Dataset:
    try:
        if not np.issubdtype(ds["time"].dtype, np.datetime64):
            ds = xr.decode_cf(ds)
    except Exception:
        ds = xr.decode_cf(ds)
    return ds


def detect_lat_lon_names_from_ds(ds: xr.Dataset) -> Tuple[str, str]:
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]

    lat_name = None
    lon_name = None

    for n in lat_candidates:
        if n in ds.dims or n in ds.coords:
            lat_name = n
            break

    for n in lon_candidates:
        if n in ds.dims or n in ds.coords:
            lon_name = n
            break

    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot detect lat/lon names. dims={ds.dims}, coords={list(ds.coords)}")

    return lat_name, lon_name


def _find_var(ds: xr.Dataset, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"Cannot find variable from {candidates}. Available: {list(ds.data_vars)}")


def wrap_lon_if_needed(lons: np.ndarray) -> np.ndarray:
    lons = np.asarray(lons, dtype=float)
    if np.nanmax(lons) > 180.0 and np.nanmin(lons) >= 0.0:
        return ((lons + 180.0) % 360.0) - 180.0
    return lons


def count_run_max_1d(arr_1d: np.ndarray) -> np.int32:
    arr = np.asarray(arr_1d, dtype=bool)
    run = 0
    mx = 0
    for v in arr:
        if v:
            run += 1
            if run > mx:
                mx = run
        else:
            run = 0
    return np.int32(mx)


def apply_max_run(mask_3d: xr.DataArray) -> xr.DataArray:
    if hasattr(mask_3d.data, "chunks"):
        mask_3d = mask_3d.chunk({"time": -1})

    out = xr.apply_ufunc(
        count_run_max_1d,
        mask_3d,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int32],
    )
    return out


def enforce_min_event_len(mask_3d: xr.DataArray, min_len: int) -> xr.DataArray:
    def _filter_short_runs_1d(a: np.ndarray, m: int) -> np.ndarray:
        a = np.asarray(a, dtype=bool)
        out = a.copy()
        n = len(a)
        i = 0
        while i < n:
            if not a[i]:
                i += 1
                continue
            j = i
            while j < n and a[j]:
                j += 1
            if (j - i) < m:
                out[i:j] = False
            i = j
        return out.astype(np.int8)

    if hasattr(mask_3d.data, "chunks"):
        mask_3d = mask_3d.chunk({"time": -1})

    filtered = xr.apply_ufunc(
        _filter_short_runs_1d,
        mask_3d,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        kwargs={"m": min_len},
        output_dtypes=[np.int8],
    )
    return filtered.astype(bool)


def cl_exceedance(ex_day: xr.DataArray, ex_night: xr.DataArray) -> xr.DataArray:
    if CL_EXCEED_MODE == "weighted":
        return CL_WEIGHT_DAY * ex_day + CL_WEIGHT_NIGHT * ex_night
    return 0.5 * (ex_day + ex_night)


def get_transformer_4326_to_raster(lulc_tif: str) -> Transformer:
    with rasterio.open(lulc_tif) as src:
        raster_crs = src.crs
        if raster_crs is None:
            raise ValueError(f"LULC raster has no CRS: {lulc_tif}")
    return Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)


def sample_lulc_values(lulc_tif: str, lons_deg: np.ndarray, lats_deg: np.ndarray,
                       transformer: Transformer) -> np.ndarray:
    lons = wrap_lon_if_needed(lons_deg)
    lats = np.asarray(lats_deg, dtype=float)
    x, y = transformer.transform(lons, lats)
    with rasterio.open(lulc_tif) as src:
        pts = list(zip(np.asarray(x).tolist(), np.asarray(y).tolist()))
        vals = np.array([v[0] for v in src.sample(pts)], dtype=np.int32)
    return vals


def deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    t = ds["time"].values
    _, idx = np.unique(t, return_index=True)
    idx = np.sort(idx)
    if len(idx) != len(t):
        ds = ds.isel(time=idx)
    return ds


def list_era5_month_files(year: int) -> List[str]:
    files = []
    for m in range(1, 13):
        fp = os.path.join(ERA5_DAILY_DIR, f"era5_daily_t2m_{year}{m:02d}_utc+5.nc")
        if os.path.exists(fp):
            files.append(fp)
    return files


def build_era5_file_index() -> pd.DataFrame:
    records = []
    for year in tqdm(range(YEAR_START, YEAR_END + 1), desc="Scanning ERA5 files", unit="year"):
        for fp in list_era5_month_files(year):
            records.append({"year": year, "file": fp})

    df = pd.DataFrame(records)
    df.to_csv(ERA5_FILE_INDEX_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] ERA5 file index saved: {ERA5_FILE_INDEX_CSV}")
    return df


def build_gsod_file_index() -> pd.DataFrame:
    station_files = sorted(glob.glob(os.path.join(GSOD_DIR, "*.csv")))
    if not station_files:
        station_files = sorted(glob.glob(os.path.join(GSOD_DIR, "**", "*.csv"), recursive=True))

    if not station_files:
        raise FileNotFoundError(f"No GSOD station CSV found under: {GSOD_DIR}")

    df = pd.DataFrame({"station_file": station_files})
    df.to_csv(GSOD_FILE_INDEX_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] GSOD file index saved: {GSOD_FILE_INDEX_CSV}")
    return df


def build_lulc_year_index() -> pd.DataFrame:
    tif_list = sorted(glob.glob(os.path.join(LULC6_DIR, LULC6_GLOB)))
    if not tif_list:
        raise FileNotFoundError(f"No annual LULC6 tif found in: {LULC6_DIR}")

    records = []
    year_pat = re.compile(r"(20\d{2})")

    for fp in tqdm(tif_list, desc="Scanning annual LULC6 files", unit="file"):
        yy = None
        for h in year_pat.findall(os.path.basename(fp)):
            y = int(h)
            if YEAR_START <= y <= YEAR_END:
                yy = y
                break
        if yy is not None:
            records.append({"year": yy, "lulc_tif": fp})

    df = pd.DataFrame(records).sort_values("year")
    if df.empty:
        raise ValueError("No valid year parsed from LULC filenames.")

    if df["year"].duplicated().any():
        raise ValueError(f"Duplicate LULC year file exists:\n{df[df['year'].duplicated(keep=False)]}")

    df.to_csv(LULC_YEAR_INDEX_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] LULC year index saved: {LULC_YEAR_INDEX_CSV}")
    return df


def build_gsod_station_meta(gsod_idx: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"station", "date", "year", "lat", "lon", "Tmax", "Tmin"}
    recs = []

    for fp in tqdm(gsod_idx["station_file"].tolist(), desc="Scanning GSOD station meta", unit="file"):
        try:
            df = pd.read_csv(fp, nrows=20)
        except Exception:
            continue

        if df.empty or (not required_cols.issubset(df.columns)):
            continue

        try:
            lat = float(pd.to_numeric(df["lat"], errors="coerce").dropna().iloc[0])
            lon = float(pd.to_numeric(df["lon"], errors="coerce").dropna().iloc[0])
        except Exception:
            continue

        station_id = str(df["station"].iloc[0]) if "station" in df.columns else os.path.basename(fp)
        recs.append({
            "station_file": fp,
            "station_id": station_id,
            "lat": lat,
            "lon": lon
        })

    out = pd.DataFrame(recs).drop_duplicates(subset=["station_file"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid GSOD station metadata found.")

    out.to_csv(GSOD_STATION_META_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] GSOD station meta saved: {GSOD_STATION_META_CSV}")
    return out


def build_gsod_station_lulc_year_cache(st_meta: pd.DataFrame, lulc_year_df: pd.DataFrame) -> pd.DataFrame:
    lons = wrap_lon_if_needed(st_meta["lon"].values)
    lats = st_meta["lat"].values
    station_files = st_meta["station_file"].tolist()
    station_ids = st_meta["station_id"].tolist()

    recs = []
    for _, row in tqdm(lulc_year_df.iterrows(), total=len(lulc_year_df), desc="Matching GSOD to annual LULC", unit="year"):
        yy = int(row["year"])
        tif = row["lulc_tif"]
        transformer = get_transformer_4326_to_raster(tif)
        vals = sample_lulc_values(tif, lons, lats, transformer)
        vals = np.where(np.isin(vals, LULC6_ORDER), vals, 0).astype(int)

        for fp, sid, lat, lon, v in zip(station_files, station_ids, lats, lons, vals):
            recs.append({
                "year": yy,
                "station_file": fp,
                "station_id": sid,
                "lat": float(lat),
                "lon": float(lon),
                "lulc_code": int(v),
                "lulc_name": LULC6_LABELS.get(int(v), "Invalid")
            })

    out = pd.DataFrame(recs)
    out.to_csv(GSOD_STATION_LULC_YEAR_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] GSOD station-year LULC cache saved: {GSOD_STATION_LULC_YEAR_CSV}")
    return out


def _estimate_lat_chunk(n_years: int, n_lon: int, target_mb: int) -> int:
    target_bytes = target_mb * 1024 * 1024
    denom = 366 * n_years * n_lon * 4 * 2
    if denom <= 0:
        return 8
    chunk_lat = max(1, int(target_bytes // denom))
    return max(1, chunk_lat)


def build_thresholds_from_era5_daily_lowmem() -> xr.Dataset:
    baseline_years = list(range(BASELINE_START, BASELINE_END + 1))
    baseline_map = {y: list_era5_month_files(y) for y in baseline_years}

    n_files = sum(len(v) for v in baseline_map.values())
    if n_files == 0:
        raise FileNotFoundError("No baseline ERA5 daily files found.")

    sample_fp = None
    for y in baseline_years:
        if baseline_map[y]:
            sample_fp = baseline_map[y][0]
            break
    if sample_fp is None:
        raise FileNotFoundError("No sample ERA5 file found.")

    ds0 = xr.open_dataset(sample_fp)
    ds0 = ensure_datetime(ds0)
    lat_name, lon_name = detect_lat_lon_names_from_ds(ds0)
    v_tx = _find_var(ds0, ["t2m_daily_max", "Tmax", "tx", "tmax"])
    v_tn = _find_var(ds0, ["t2m_daily_min", "Tmin", "tn", "tmin"])
    lat_vals = ds0[lat_name].values
    lon_vals = ds0[lon_name].values
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    ds0.close()

    n_years = len(baseline_years)
    lat_chunk = _estimate_lat_chunk(n_years, n_lon, THR_TARGET_MB)
    lat_chunk = min(lat_chunk, n_lat)

    print(f"[INFO] Computing ERA5 thresholds from {n_files} files ({BASELINE_START}-{BASELINE_END}) ...")
    print(f"[INFO] Low-memory threshold mode: n_lat={n_lat}, n_lon={n_lon}, lat_chunk={lat_chunk}, n_years={n_years}")

    safe_remove(THR_FILE)

    nc = Dataset(THR_FILE, "w", format="NETCDF4")
    nc.createDimension("dayofyear", 366)
    nc.createDimension(lat_name, n_lat)
    nc.createDimension(lon_name, n_lon)

    doy_var = nc.createVariable("dayofyear", "i4", ("dayofyear",))
    lat_var = nc.createVariable(lat_name, "f8", (lat_name,))
    lon_var = nc.createVariable(lon_name, "f8", (lon_name,))

    tx_var = nc.createVariable("TX90p", "f4", ("dayofyear", lat_name, lon_name),
                               zlib=True, complevel=4, fill_value=np.nan)
    tn_var = nc.createVariable("TN90p", "f4", ("dayofyear", lat_name, lon_name),
                               zlib=True, complevel=4, fill_value=np.nan)

    doy_var[:] = np.arange(1, 367)
    lat_var[:] = lat_vals.astype(np.float32)
    lon_var[:] = lon_vals.astype(np.float32)

    nc.setncattr("title", "ERA5 baseline thresholds (TX90p/TN90p)")
    nc.setncattr("baseline_period", f"{BASELINE_START}-{BASELINE_END}")
    nc.setncattr("method", "Low-memory chunked exact quantile by day-of-year")

    chunk_starts = list(range(0, n_lat, lat_chunk))

    for i0 in tqdm(chunk_starts, desc="Threshold spatial chunks", unit="chunk"):
        i1 = min(i0 + lat_chunk, n_lat)
        this_nlat = i1 - i0

        tx_store = np.full((366, n_years, this_nlat, n_lon), np.nan, dtype=np.float32)
        tn_store = np.full((366, n_years, this_nlat, n_lon), np.nan, dtype=np.float32)

        for iy, year in enumerate(tqdm(baseline_years, desc=f"Reading baseline years for lat[{i0}:{i1}]", leave=False, unit="year")):
            files = baseline_map[year]
            if not files:
                continue

            for fp in files:
                ds = xr.open_dataset(fp)
                ds = ensure_datetime(ds)
                ds = deduplicate_time(ds)

                tx = ds[v_tx].isel({lat_name: slice(i0, i1), lon_name: slice(None)}).load()
                tn = ds[v_tn].isel({lat_name: slice(i0, i1), lon_name: slice(None)}).load()

                tx0 = tx.values
                if np.isfinite(tx0).any() and float(np.nanmedian(tx0)) > 150:
                    tx = tx - 273.15
                    tn = tn - 273.15

                doys = tx["time"].dt.dayofyear.values.astype(int)
                tx_vals = tx.values.astype(np.float32)
                tn_vals = tn.values.astype(np.float32)

                for it, doy in enumerate(doys):
                    d = int(doy) - 1
                    if 0 <= d < 366:
                        tx_store[d, iy, :, :] = tx_vals[it]
                        tn_store[d, iy, :, :] = tn_vals[it]

                ds.close()

        tx_q = np.nanquantile(tx_store, 0.9, axis=1).astype(np.float32)
        tn_q = np.nanquantile(tn_store, 0.9, axis=1).astype(np.float32)

        tx_var[:, i0:i1, :] = tx_q
        tn_var[:, i0:i1, :] = tn_q

        del tx_store, tn_store, tx_q, tn_q

    nc.close()

    print(f"[DONE] Threshold cache saved: {THR_FILE}")
    return xr.open_dataset(THR_FILE)


def load_or_build_thresholds() -> xr.Dataset:
    if FORCE_REBUILD_THRESHOLD_CACHE and os.path.exists(THR_FILE):
        print(f"[INFO] Force rebuilding ERA5 threshold cache: {THR_FILE}")
        safe_remove(THR_FILE)

    if os.path.exists(THR_FILE):
        print(f"[INFO] Loading ERA5 threshold cache: {THR_FILE}")
        return xr.open_dataset(THR_FILE)
    return build_thresholds_from_era5_daily_lowmem()


def _coords_match_1d(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.allclose(a.astype(float), b.astype(float), atol=atol, rtol=0.0, equal_nan=True)


def ensure_threshold_cache_matches_era5(sample_fp: str, thr: xr.Dataset) -> xr.Dataset:
    ds = xr.open_dataset(sample_fp)
    ds = ensure_datetime(ds)
    lat_name, lon_name = detect_lat_lon_names_from_ds(ds)

    thr_lat_name, thr_lon_name = detect_lat_lon_names_from_ds(thr)

    ok = (
        len(ds[lat_name]) == len(thr[thr_lat_name]) and
        len(ds[lon_name]) == len(thr[thr_lon_name]) and
        _coords_match_1d(ds[lat_name].values, thr[thr_lat_name].values) and
        _coords_match_1d(ds[lon_name].values, thr[thr_lon_name].values)
    )

    ds.close()

    if ok:
        return thr

    print("[WARN] ERA5 threshold cache grid does not exactly match the daily ERA5 grid.")
    print("[WARN] Rebuilding threshold cache to avoid silent xarray alignment errors ...")

    try:
        thr.close()
    except Exception:
        pass

    safe_remove(THR_FILE)
    return build_thresholds_from_era5_daily_lowmem()


def select_threshold_on_data_grid(data_3d: xr.DataArray, thr_field: xr.DataArray) -> xr.DataArray:
    doy = data_3d["time"].dt.dayofyear
    thr_sel = thr_field.sel(dayofyear=doy)

    if thr_sel.dims != data_3d.dims:
        thr_sel = thr_sel.transpose(*data_3d.dims)


    thr_sel = xr.DataArray(
        data=thr_sel.data,
        coords={dim: data_3d.coords[dim] for dim in data_3d.dims},
        dims=data_3d.dims,
        name=thr_field.name,
        attrs=thr_field.attrs,
    )
    return thr_sel


def da_scalar(da) -> float:
    if isinstance(da, xr.DataArray):
        if hasattr(da.data, "compute"):
            da = da.compute()
        vals = np.asarray(da.values)
    else:
        if hasattr(da, "compute"):
            da = da.compute()
        vals = np.asarray(da)

    if vals.size != 1:
        raise ValueError(f"Expected scalar-like result, got shape={vals.shape}")
    return vals.reshape(-1)[0].item()


def build_valid_era5_grid(tx: xr.DataArray, tn: xr.DataArray,
                          tx90: xr.DataArray, tn90: xr.DataArray) -> xr.DataArray:
    lat_name, lon_name = tx.dims[1], tx.dims[2]
    valid_tx = np.isfinite(tx).any("time")
    valid_tn = np.isfinite(tn).any("time")
    valid_tx90 = np.isfinite(tx90).any("time")
    valid_tn90 = np.isfinite(tn90).any("time")

    valid_grid = (valid_tx & valid_tn & valid_tx90 & valid_tn90)
    valid_grid = valid_grid.astype(bool)
    valid_grid = valid_grid.rename("ERA5_valid_grid")

    if valid_grid.dims != (lat_name, lon_name):
        valid_grid = valid_grid.transpose(lat_name, lon_name)

    return valid_grid


def build_era5_grid_transform(lons: np.ndarray, lats: np.ndarray):
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if len(lons) < 2 or len(lats) < 2:
        raise ValueError("ERA5 lat/lon length too short to build grid transform.")

    dx = float(np.nanmedian(np.abs(np.diff(lons))))
    dy = float(np.nanmedian(np.abs(np.diff(lats))))

    left = float(np.nanmin(lons) - dx / 2.0)
    top = float(np.nanmax(lats) + dy / 2.0)

    transform = from_origin(left, top, dx, dy)
    lat_desc = bool(lats[0] > lats[-1])

    return transform, lat_desc


def reproject_lulc_to_era5_grid(sample_ds: xr.Dataset, lulc_tif: str) -> Tuple[xr.DataArray, Dict[int, int]]:
    lat_name, lon_name = detect_lat_lon_names_from_ds(sample_ds)
    lats = sample_ds[lat_name].values
    lons = sample_ds[lon_name].values

    n_lat = len(lats)
    n_lon = len(lons)

    dst_transform, lat_desc = build_era5_grid_transform(lons, lats)
    dst = np.zeros((n_lat, n_lon), dtype=np.int16)

    with rasterio.open(lulc_tif) as src:
        src_arr = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata if src.nodata is not None else 0

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    if not lat_desc:
        dst = np.flipud(dst)

    dst = np.where(np.isin(dst, LULC6_ORDER), dst, 0).astype(np.int16)
    counts = {code: int(np.sum(dst == code)) for code in LULC6_ORDER}

    da = xr.DataArray(dst, coords={lat_name: lats, lon_name: lons}, dims=(lat_name, lon_name), name="LULC6")
    return da, counts


def build_or_load_lulc_on_era5_grid_for_year(sample_ds: xr.Dataset, year: int, lulc_tif: str) -> Tuple[xr.DataArray, Dict[int, int]]:
    lat_name, lon_name = detect_lat_lon_names_from_ds(sample_ds)
    lats = sample_ds[lat_name].values
    lons = sample_ds[lon_name].values

    cache_npz = os.path.join(ERA5_LULC_CACHE_DIR, f"lulc_on_era5_grid_{year}.npz")


    if FORCE_REBUILD_ERA5_LULC_CACHE and os.path.exists(cache_npz):
        print(f"[INFO] Force rebuilding ERA5-LULC cache for year {year}: {cache_npz}")
        safe_remove(cache_npz)


    if os.path.exists(cache_npz):
        try:
            npz = np.load(cache_npz)
            vals = npz["lulc"]

            count_keys = [k for k in npz.files if k.startswith("c")]
            if count_keys:
                counts = {int(k.replace("c", "")): int(npz[k]) for k in count_keys}
            else:
                counts = count_lulc_codes(vals)

            valid_total = sum(counts.values())
            if vals.shape == (len(lats), len(lons)) and valid_total > 0:
                da = xr.DataArray(vals, coords={lat_name: lats, lon_name: lons}, dims=(lat_name, lon_name), name="LULC6")
                print(f"[INFO] Reused ERA5-LULC cache for year {year}: {cache_npz}")
                return da, counts

            print(f"[WARN] Bad/empty old ERA5-LULC cache detected for year {year}, rebuilding ...")
            safe_remove(cache_npz)

        except Exception as e:
            print(f"[WARN] Failed to read ERA5-LULC cache for year {year}, rebuilding ... {e}")
            safe_remove(cache_npz)


    da, counts = reproject_lulc_to_era5_grid(sample_ds, lulc_tif)
    valid_total = sum(counts.values())

    if valid_total == 0:
        raise RuntimeError(
            f"Year {year}: rebuilt annual LULC on ERA5 grid still has 0 valid cells. "
            f"Please check annual LULC raster CRS/extent/data values."
        )

    save_dict = {"lulc": da.values.astype(np.int16)}
    for code in LULC6_ORDER:
        save_dict[f"c{code}"] = np.array(counts.get(code, 0), dtype=np.int32)

    np.savez_compressed(cache_npz, **save_dict)
    print(f"[DONE] Saved ERA5-grid annual LULC cache: {cache_npz}")

    return da, counts


def compute_era5_yearly_metrics_by_dynamic_lulc(thr: xr.Dataset, lulc_year_df: pd.DataFrame):
    rec = {ind: [] for ind in INDICATORS}
    valid_cells_records = []

    sample_files = sorted(glob.glob(os.path.join(ERA5_DAILY_DIR, "era5_daily_t2m_*_utc+5.nc")))
    if not sample_files:
        raise FileNotFoundError(f"No ERA5 daily files found in: {ERA5_DAILY_DIR}")

    ds_sample = xr.open_dataset(sample_files[0])
    ds_sample = ensure_datetime(ds_sample)
    lat_name, lon_name = detect_lat_lon_names_from_ds(ds_sample)
    ds_sample.close()

    lulc_map = dict(zip(lulc_year_df["year"].astype(int), lulc_year_df["lulc_tif"]))

    for year in tqdm(range(YEAR_START, YEAR_END + 1), desc="Computing ERA5 yearly metrics", unit="year"):
        files = list_era5_month_files(year)
        if (not files) or (year not in lulc_map):
            continue

        dsy = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        dsy = ensure_datetime(dsy)
        dsy = deduplicate_time(dsy)

        v_tx = _find_var(dsy, ["t2m_daily_max", "Tmax", "tx", "tmax"])
        v_tn = _find_var(dsy, ["t2m_daily_min", "Tmin", "tn", "tmin"])
        tx = dsy[v_tx]
        tn = dsy[v_tn]

        tx0 = tx.isel(time=0).load()
        if float(np.nanmedian(tx0.values)) > 150:
            tx = tx - 273.15
            tn = tn - 273.15

        tx90 = select_threshold_on_data_grid(tx, thr["TX90p"])
        tn90 = select_threshold_on_data_grid(tn, thr["TN90p"])

        valid_grid = build_valid_era5_grid(tx, tn, tx90, tn90)
        if hasattr(valid_grid.data, "compute"):
            valid_grid = valid_grid.load()


        hot_day = xr.where(valid_grid, tx > tx90, False)
        hot_night = xr.where(valid_grid, tn > tn90, False)
        is_DL = hot_day & (~hot_night)
        is_NL = hot_night & (~hot_day)
        is_CL = hot_day & hot_night

        is_DL_f = enforce_min_event_len(is_DL, MIN_EVENT_LEN)
        is_NL_f = enforce_min_event_len(is_NL, MIN_EVENT_LEN)
        is_CL_f = enforce_min_event_len(is_CL, MIN_EVENT_LEN)

        hwf_DL = is_DL_f.sum("time")
        hwf_NL = is_NL_f.sum("time")
        hwf_CL = is_CL_f.sum("time")

        hwd_DL = apply_max_run(is_DL_f)
        hwd_NL = apply_max_run(is_NL_f)
        hwd_CL = apply_max_run(is_CL_f)

        ex_day = tx - tx90
        ex_night = tn - tn90
        ex_cl = cl_exceedance(ex_day, ex_night)

        hwi_DL = ex_day.where(is_DL_f).mean("time", skipna=True)
        hwi_NL = ex_night.where(is_NL_f).mean("time", skipna=True)
        hwi_CL = ex_cl.where(is_CL_f).mean("time", skipna=True)

        for da in [hwf_DL, hwf_NL, hwf_CL, hwd_DL, hwd_NL, hwd_CL, hwi_DL, hwi_NL, hwi_CL]:
            if hasattr(da.data, "compute"):
                da.load()

        lulc_da, counts = build_or_load_lulc_on_era5_grid_for_year(dsy, year, lulc_map[year])
        if lulc_da.dims != (lat_name, lon_name):
            lulc_da = lulc_da.rename({lulc_da.dims[0]: lat_name, lulc_da.dims[1]: lon_name})

        valid_total = sum(counts.values())
        valid_cells_records.append({
            "year": year,
            **{f"code_{code}": counts.get(code, 0) for code in LULC6_ORDER},
            "valid_total": valid_total
        })

        if valid_total == 0:
            raise RuntimeError(
                f"Year {year}: annual LULC reprojected to ERA5 grid has 0 valid cells. "
                f"Check annual LULC raster CRS/extent/data values."
            )

        for code in LULC6_ORDER:


            mask = (lulc_da == code) & valid_grid
            n_cells = int(da_scalar(mask.sum()))

            if n_cells == 0:
                vals_hwf = (np.nan, np.nan, np.nan)
                vals_hwd = (np.nan, np.nan, np.nan)
                vals_hwi = (np.nan, np.nan, np.nan)
            else:
                vals_hwf = (
                    float(da_scalar(hwf_DL.where(mask).mean(dim=(lat_name, lon_name)))),
                    float(da_scalar(hwf_NL.where(mask).mean(dim=(lat_name, lon_name)))),
                    float(da_scalar(hwf_CL.where(mask).mean(dim=(lat_name, lon_name)))),
                )
                vals_hwd = (
                    float(da_scalar(hwd_DL.where(mask).mean(dim=(lat_name, lon_name)))),
                    float(da_scalar(hwd_NL.where(mask).mean(dim=(lat_name, lon_name)))),
                    float(da_scalar(hwd_CL.where(mask).mean(dim=(lat_name, lon_name)))),
                )
                vals_hwi = (
                    float(da_scalar(hwi_DL.where(mask).mean(dim=(lat_name, lon_name)))),
                    float(da_scalar(hwi_NL.where(mask).mean(dim=(lat_name, lon_name)))),
                    float(da_scalar(hwi_CL.where(mask).mean(dim=(lat_name, lon_name)))),
                )

            rec["HWF"].append({
                "year": year, "lulc_code": code, "lulc_name": LULC6_LABELS[code],
                "DL": vals_hwf[0], "NL": vals_hwf[1], "CL": vals_hwf[2], "n_cells": n_cells
            })
            rec["HWD"].append({
                "year": year, "lulc_code": code, "lulc_name": LULC6_LABELS[code],
                "DL": vals_hwd[0], "NL": vals_hwd[1], "CL": vals_hwd[2], "n_cells": n_cells
            })
            rec["HWI"].append({
                "year": year, "lulc_code": code, "lulc_name": LULC6_LABELS[code],
                "DL": vals_hwi[0], "NL": vals_hwi[1], "CL": vals_hwi[2], "n_cells": n_cells
            })

        dsy.close()

    valid_df = pd.DataFrame(valid_cells_records)
    valid_df.to_csv(ERA5_VALID_CELLS_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] ERA5 valid-cell diagnostics saved: {ERA5_VALID_CELLS_CSV}")

    return {ind: pd.DataFrame(rec[ind]) for ind in INDICATORS}


def _clean_temp_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([9999, 999.9, 99.99, 999.99, -9999, -999.9], np.nan)
    s[(s < -80) | (s > 80)] = np.nan
    return s


def station_thresholds_by_doy(df: pd.DataFrame):
    base = df[(df["year"] >= BASELINE_START) & (df["year"] <= BASELINE_END)].copy()
    if base.empty:
        return None, None, 0

    ny = int(base["year"].nunique())
    if ny < MIN_BASELINE_YEARS_STATION:
        return None, None, ny

    if "doy" not in base.columns:
        if "date" in base.columns:
            base["date"] = pd.to_datetime(base["date"], errors="coerce")
            base["doy"] = base["date"].dt.dayofyear
        else:
            return None, None, ny

    tx90 = base.groupby("doy")["Tmax"].quantile(0.9).to_dict()
    tn90 = base.groupby("doy")["Tmin"].quantile(0.9).to_dict()
    return tx90, tn90, ny


def filter_short_runs_1d(a: np.ndarray, m: int) -> np.ndarray:
    a = np.asarray(a, dtype=bool)
    out = a.copy()
    n = len(a)
    i = 0
    while i < n:
        if not a[i]:
            i += 1
            continue
        j = i
        while j < n and a[j]:
            j += 1
        if (j - i) < m:
            out[i:j] = False
        i = j
    return out


def station_year_metrics(df_year: pd.DataFrame, tx90_map: Dict[int, float], tn90_map: Dict[int, float]):
    dy = df_year.dropna(subset=["Tmax", "Tmin", "doy"]).copy()
    if dy.empty:
        return None

    thr_tx = dy["doy"].map(tx90_map)
    thr_tn = dy["doy"].map(tn90_map)

    hot_day = (dy["Tmax"] > thr_tx).fillna(False).values
    hot_night = (dy["Tmin"] > thr_tn).fillna(False).values

    is_DL = hot_day & (~hot_night)
    is_NL = hot_night & (~hot_day)
    is_CL = hot_day & hot_night

    is_DL = filter_short_runs_1d(is_DL, MIN_EVENT_LEN)
    is_NL = filter_short_runs_1d(is_NL, MIN_EVENT_LEN)
    is_CL = filter_short_runs_1d(is_CL, MIN_EVENT_LEN)

    hwf = {
        "DL": float(is_DL.sum()),
        "NL": float(is_NL.sum()),
        "CL": float(is_CL.sum())
    }
    hwd = {
        "DL": float(count_run_max_1d(is_DL)),
        "NL": float(count_run_max_1d(is_NL)),
        "CL": float(count_run_max_1d(is_CL))
    }

    ex_day = (dy["Tmax"] - thr_tx).values.astype(float)
    ex_night = (dy["Tmin"] - thr_tn).values.astype(float)

    if CL_EXCEED_MODE == "weighted":
        ex_cl = CL_WEIGHT_DAY * ex_day + CL_WEIGHT_NIGHT * ex_night
    else:
        ex_cl = 0.5 * (ex_day + ex_night)

    def mean_exceed(ex, mask):
        ex2 = ex[mask]
        return np.nan if ex2.size == 0 else float(np.nanmean(ex2))

    hwi = {
        "DL": mean_exceed(ex_day, is_DL),
        "NL": mean_exceed(ex_night, is_NL),
        "CL": mean_exceed(ex_cl, is_CL),
    }

    return {"HWF": hwf, "HWD": hwd, "HWI": hwi}


def compute_gsod_yearly_metrics_by_dynamic_lulc(gsod_idx: pd.DataFrame, station_lulc_year: pd.DataFrame):
    required_cols = {"station", "date", "year", "lat", "lon", "Tmax", "Tmin"}

    sums = {ind: {code: {} for code in LULC6_ORDER} for ind in INDICATORS}
    cnts = {code: {} for code in LULC6_ORDER}
    cnts_hwi = {code: {} for code in LULC6_ORDER}

    lulc_lookup = {
        (row["station_file"], int(row["year"])): int(row["lulc_code"])
        for _, row in station_lulc_year.iterrows()
    }

    station_files = gsod_idx["station_file"].tolist()

    for fp in tqdm(station_files, desc="Computing GSOD yearly metrics", unit="station"):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        if df.empty or (not required_cols.issubset(df.columns)):
            continue

        df["Tmax"] = _clean_temp_series(df["Tmax"])
        df["Tmin"] = _clean_temp_series(df["Tmin"])

        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            continue

        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "doy" not in df.columns:
            df["doy"] = df["date"].dt.dayofyear

        tx90_map, tn90_map, _ = station_thresholds_by_doy(df)
        if tx90_map is None or tn90_map is None:
            continue

        for yy in range(YEAR_START, YEAR_END + 1):
            code = lulc_lookup.get((fp, yy), 0)
            if code not in LULC6_ORDER:
                continue

            dy = df[df["year"] == yy]
            if dy.empty:
                continue

            m = station_year_metrics(dy, tx90_map, tn90_map)
            if m is None:
                continue

            if yy not in cnts[code]:
                cnts[code][yy] = 0
            if yy not in cnts_hwi[code]:
                cnts_hwi[code][yy] = np.array([0, 0, 0], dtype=int)

            for ind in INDICATORS:
                if yy not in sums[ind][code]:
                    sums[ind][code][yy] = np.array([0.0, 0.0, 0.0], dtype=float)

            sums["HWF"][code][yy] += np.array([m["HWF"]["DL"], m["HWF"]["NL"], m["HWF"]["CL"]], dtype=float)
            sums["HWD"][code][yy] += np.array([m["HWD"]["DL"], m["HWD"]["NL"], m["HWD"]["CL"]], dtype=float)

            vals_hwi = [m["HWI"]["DL"], m["HWI"]["NL"], m["HWI"]["CL"]]
            for i_type, val in enumerate(vals_hwi):
                if not np.isnan(val):
                    sums["HWI"][code][yy][i_type] += val
                    cnts_hwi[code][yy][i_type] += 1

            cnts[code][yy] += 1

    out = {}
    for ind in INDICATORS:
        recs = []
        for code in LULC6_ORDER:
            for yy in range(YEAR_START, YEAR_END + 1):
                if ind == "HWI":
                    nst_arr = cnts_hwi[code].get(yy, np.array([0, 0, 0], dtype=int))
                    sum_arr = sums[ind][code].get(yy, np.array([0.0, 0.0, 0.0], dtype=float))
                    vals = [sum_arr[k] / nst_arr[k] if nst_arr[k] > 0 else np.nan for k in range(3)]
                    dl, nl, cl = vals
                    nst_log = cnts[code].get(yy, 0)
                else:
                    nst = cnts[code].get(yy, 0)
                    if nst == 0:
                        dl = nl = cl = np.nan
                    else:
                        dl, nl, cl = (sums[ind][code][yy] / nst).tolist()
                    nst_log = nst

                recs.append({
                    "year": yy,
                    "lulc_code": code,
                    "lulc_name": LULC6_LABELS[code],
                    "DL": dl,
                    "NL": nl,
                    "CL": cl,
                    "n_stations": int(nst_log)
                })
        out[ind] = pd.DataFrame(recs)

    return out


def _read_csv_safe(csv_path: str, index_col=None) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path, index_col=index_col)
    except Exception as e:
        print(f"[WARN] Failed to read CSV: {csv_path} -> {e}")
        return None


def _normalize_summary_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if "lulc_code" in out.columns:
        out["lulc_code"] = pd.to_numeric(out["lulc_code"], errors="coerce")
        out = out.dropna(subset=["lulc_code"]).set_index("lulc_code")

    try:
        out.index = pd.Index(pd.to_numeric(out.index, errors="coerce"), name="lulc_code")
        out = out[~out.index.isna()]
        out.index = out.index.astype(int)
    except Exception:
        return pd.DataFrame()

    return out.sort_index()


def _summary_has_required_content(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False

    df = _normalize_summary_index(df)
    if df.empty:
        return False

    need_cols = {"lulc_name"}
    for hw in HW_TYPES:
        need_cols.update({f"{hw}_mean", f"{hw}_std"})

    if not need_cols.issubset(df.columns):
        return False

    if not set(LULC6_ORDER).issubset(set(df.index.tolist())):
        return False

    return True


def _yearly_has_required_content(df: pd.DataFrame, count_col: str) -> bool:
    if df is None or df.empty:
        return False

    need_cols = {"year", "lulc_code", "lulc_name", "DL", "NL", "CL", count_col}
    if not need_cols.issubset(df.columns):
        return False

    codes = pd.to_numeric(df["lulc_code"], errors="coerce").dropna().astype(int).unique().tolist()
    if not set(LULC6_ORDER).issubset(set(codes)):
        return False

    return True


def try_prepare_plot_inputs_from_intermediate(indicator: str):
    era5_yearly_csv = os.path.join(OUT_DIR, f"ERA5_dynamicLULC6_{indicator}_by_year_{YEAR_START}_{YEAR_END}.csv")
    gsod_yearly_csv = os.path.join(OUT_DIR, f"GSOD_dynamicLULC6_{indicator}_by_year_{YEAR_START}_{YEAR_END}.csv")
    era5_sum_csv = os.path.join(OUT_DIR, f"ERA5_dynamicLULC6_{indicator}_mean_std_{YEAR_START}_{YEAR_END}.csv")
    gsod_sum_csv = os.path.join(OUT_DIR, f"GSOD_dynamicLULC6_{indicator}_mean_std_{YEAR_START}_{YEAR_END}.csv")

    era5_sum = _read_csv_safe(era5_sum_csv, index_col=0)
    gsod_sum = _read_csv_safe(gsod_sum_csv, index_col=0)
    era5_sum = _normalize_summary_index(era5_sum) if era5_sum is not None else None
    gsod_sum = _normalize_summary_index(gsod_sum) if gsod_sum is not None else None

    if _summary_has_required_content(era5_sum) and _summary_has_required_content(gsod_sum):
        print(f"[INFO] {indicator}: reuse mean/std summary CSV directly for plotting.")
        return era5_sum.loc[LULC6_ORDER], gsod_sum.loc[LULC6_ORDER]

    era5_y = _read_csv_safe(era5_yearly_csv)
    gsod_y = _read_csv_safe(gsod_yearly_csv)

    if _yearly_has_required_content(era5_y, "n_cells") and _yearly_has_required_content(gsod_y, "n_stations"):
        print(f"[INFO] {indicator}: summary CSV missing/incomplete, rebuild summary from yearly CSV and plot directly.")
        era5_sum = summarize_mean_std(era5_y, "n_cells")
        gsod_sum = summarize_mean_std(gsod_y, "n_stations")
        era5_sum.to_csv(era5_sum_csv, encoding="utf-8-sig")
        gsod_sum.to_csv(gsod_sum_csv, encoding="utf-8-sig")
        return era5_sum.loc[LULC6_ORDER], gsod_sum.loc[LULC6_ORDER]

    print(f"[INFO] {indicator}: required intermediate plotting tables are not complete, fallback to full recomputation.")
    return None


def try_prepare_all_plot_inputs_from_intermediate():
    ready = {}
    for ind in INDICATORS:
        pair = try_prepare_plot_inputs_from_intermediate(ind)
        if pair is None:
            return None
        ready[ind] = pair
    return ready


def summarize_mean_std(df_by_year: pd.DataFrame, count_col: str) -> pd.DataFrame:
    out = {}
    for code in LULC6_ORDER:
        sub = df_by_year[df_by_year["lulc_code"] == code]

        row = {"lulc_name": LULC6_LABELS[code]}
        if count_col in sub.columns:
            row[f"{count_col}_mean"] = float(np.nanmean(sub[count_col].values))

        for k in HW_TYPES:
            vals = sub[k].astype(float).values
            row[f"{k}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if np.sum(np.isfinite(vals)) > 1 else np.nan

        out[code] = row

    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = "lulc_code"
    return df


def plot_6x3_two_bars(era5_sum: pd.DataFrame, gsod_sum: pd.DataFrame,
                      indicator: str, out_png: str):
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharey=True)

    fixed_ymax = float(Y_AXIS_LIMITS.get(indicator, 1.0))
    if not np.isfinite(fixed_ymax) or fixed_ymax <= 0:
        fixed_ymax = 1.0

    for i, hw in enumerate(HW_TYPES):
        for j, code in enumerate(LULC6_ORDER):
            ax = axes[i, j]

            em = float(era5_sum.loc[code, f"{hw}_mean"])
            es = float(era5_sum.loc[code, f"{hw}_std"])
            gm = float(gsod_sum.loc[code, f"{hw}_mean"])
            gs = float(gsod_sum.loc[code, f"{hw}_std"])

            ax.bar(-BAR_WIDTH / 2, em, BAR_WIDTH, yerr=es, capsize=3,
                   color=COLOR_ERA5, label="ERA5" if (i == 0 and j == 0) else None)
            ax.bar( BAR_WIDTH / 2, gm, BAR_WIDTH, yerr=gs, capsize=3,
                   color=COLOR_GSOD, label="GSOD" if (i == 0 and j == 0) else None)

            ax.set_ylim(0, fixed_ymax)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_xticks([])

            if i == 0:
                ax.set_title(LULC6_LABELS[code], fontsize=11)
            if j == 0:
                ax.set_ylabel(hw, fontsize=11)

    ylab = {
        "HWF": "HWF (days/year)",
        "HWD": "HWD (days)",
        "HWI": "HWI (°C)"
    }[indicator]

    fig.suptitle(
        f"{indicator}: ERA5 vs GSOD by Annual Dynamic Land Use Type and Heatwave Category ({YEAR_START}–{YEAR_END})",
        fontsize=14
    )
    fig.text(0.01, 0.5, ylab, va="center", rotation="vertical", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.tight_layout(rect=[0.02, 0.06, 1, 0.95])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[DONE] Figure saved: {out_png}")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if PURGE_OLD:
        purge_outputs()

    print("\n[STEP 0] Checking intermediate files for fast plotting ...")
    ready_plot_inputs = try_prepare_all_plot_inputs_from_intermediate()
    if ready_plot_inputs is not None:
        print("[INFO] All required intermediate files are available. Skip full recalculation and plot directly.")
        for ind in tqdm(INDICATORS, desc="Fast plotting from intermediates", unit="figure"):
            era5_sum, gsod_sum = ready_plot_inputs[ind]
            out_png = os.path.join(OUT_DIR, f"Fig_{ind}_dynamicLULC6_6x3_ERA5_vs_GSOD_{YEAR_START}_{YEAR_END}.png")
            plot_6x3_two_bars(era5_sum, gsod_sum, ind, out_png)
        print("\nALL DONE.")
        return

    if FORCE_REBUILD_ERA5_LULC_CACHE:
        purge_era5_lulc_cache()

    print("\n[STEP 1] Building file indexes ...")
    build_era5_file_index()
    gsod_idx = build_gsod_file_index()
    lulc_year_df = build_lulc_year_index()
    st_meta = build_gsod_station_meta(gsod_idx)

    print("\n[STEP 2] Matching GSOD stations to annual LULC ...")
    station_lulc_year = build_gsod_station_lulc_year_cache(st_meta, lulc_year_df)

    print("\n[STEP 3] Loading or computing ERA5 thresholds ...")
    thr = load_or_build_thresholds()

    sample_files = sorted(glob.glob(os.path.join(ERA5_DAILY_DIR, "era5_daily_t2m_*_utc+5.nc")))
    if not sample_files:
        raise FileNotFoundError(f"No ERA5 daily files found in: {ERA5_DAILY_DIR}")
    thr = ensure_threshold_cache_matches_era5(sample_files[0], thr)

    print("\n[STEP 4] Computing ERA5 yearly metrics ...")
    era5_yearly = compute_era5_yearly_metrics_by_dynamic_lulc(thr, lulc_year_df)

    print("\n[STEP 5] Computing GSOD yearly metrics ...")
    gsod_yearly = compute_gsod_yearly_metrics_by_dynamic_lulc(gsod_idx, station_lulc_year)

    print("\n[STEP 6] Saving tables and plotting ...")
    for ind in tqdm(INDICATORS, desc="Saving & plotting", unit="figure"):
        era5_y = era5_yearly[ind]
        gsod_y = gsod_yearly[ind]

        era5_yearly_csv = os.path.join(OUT_DIR, f"ERA5_dynamicLULC6_{ind}_by_year_{YEAR_START}_{YEAR_END}.csv")
        gsod_yearly_csv = os.path.join(OUT_DIR, f"GSOD_dynamicLULC6_{ind}_by_year_{YEAR_START}_{YEAR_END}.csv")
        era5_y.to_csv(era5_yearly_csv, index=False, encoding="utf-8-sig")
        gsod_y.to_csv(gsod_yearly_csv, index=False, encoding="utf-8-sig")

        era5_sum = summarize_mean_std(era5_y, "n_cells")
        gsod_sum = summarize_mean_std(gsod_y, "n_stations")

        era5_sum_csv = os.path.join(OUT_DIR, f"ERA5_dynamicLULC6_{ind}_mean_std_{YEAR_START}_{YEAR_END}.csv")
        gsod_sum_csv = os.path.join(OUT_DIR, f"GSOD_dynamicLULC6_{ind}_mean_std_{YEAR_START}_{YEAR_END}.csv")
        era5_sum.to_csv(era5_sum_csv, encoding="utf-8-sig")
        gsod_sum.to_csv(gsod_sum_csv, encoding="utf-8-sig")

        out_png = os.path.join(OUT_DIR, f"Fig_{ind}_dynamicLULC6_6x3_ERA5_vs_GSOD_{YEAR_START}_{YEAR_END}.png")
        plot_6x3_two_bars(era5_sum, gsod_sum, ind, out_png)

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
