from __future__ import annotations

import os
import re
import gc
import json
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


GSOD_ROOT = Path(r"G:\CYX\Analyse\normalized_by_station")
ERA5_DAILY_DIR = Path(r"G:\CYX\ERA5\daily")
SHP_ROOT = Path(r"G:\Central Asia")
OUT_ROOT = Path(r"G:\CYX\Robustness test")

START_YEAR = 1950
END_YEAR = 2024
P1_END = 1999
P2_START = 2000


ERA5_LOCAL_UTC_OFFSET_HOURS = +5

SCHEMES = [
    {"id": "S0", "label": "Base",         "q": 0.90, "base_start": 1991, "base_end": 2020, "min_run": 3, "station_set": "all"},
    {"id": "S1", "label": "Q95",          "q": 0.95, "base_start": 1991, "base_end": 2020, "min_run": 3, "station_set": "all"},
    {"id": "S2", "label": "Base61_90",    "q": 0.90, "base_start": 1961, "base_end": 1990, "min_run": 3, "station_set": "all"},
    {"id": "S3", "label": "Base81_10",    "q": 0.90, "base_start": 1981, "base_end": 2010, "min_run": 3, "station_set": "all"},
    {"id": "S4", "label": "Run>=4d",      "q": 0.90, "base_start": 1991, "base_end": 2020, "min_run": 4, "station_set": "all"},
    {"id": "S6", "label": "HQ stations",  "q": 0.90, "base_start": 1991, "base_end": 2020, "min_run": 3, "station_set": "hq"},
]

TYPE_CODE_TO_NAME = {1: "DL", 2: "NL", 3: "CL"}
TYPE_ORDER = ["CL", "DL", "NL"]
METRIC_ORDER = ["HWF", "HWD", "HWI"]
SOURCE_ORDER = ["gsod", "era5"]
SELECTED_MAP_SCHEMES = ["S1", "S4", "S6"]

HQ_MIN_VALID_YEARS = 50
HQ_MIN_VALID_DAYS_PER_YEAR = 300
HQ_MIN_OVERALL_COMPLETENESS = 0.90

FIG_DPI = 300
POINT_SIZE = 18


CACHE_DIR = OUT_ROOT / "cache"
FIG_DIR = OUT_ROOT / "figures"
FIG2_DIR = FIG_DIR / "Fig2"
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"
ERA5_MONTH_CACHE_DIR = CACHE_DIR / "era5_station_monthly"
GSOD_CHUNK_DIR = CACHE_DIR / "gsod_chunks"
THR_DIR = CACHE_DIR / "thresholds"
SCHEME_DIR = CACHE_DIR / "scheme_results"

for p in [OUT_ROOT, CACHE_DIR, FIG_DIR, FIG2_DIR, TABLE_DIR, LOG_DIR, ERA5_MONTH_CACHE_DIR, GSOD_CHUNK_DIR, THR_DIR, SCHEME_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CACHE_EXT = ".pkl.gz"


def save_df(df: pd.DataFrame, path_no_ext: Path) -> Path:
    path = path_no_ext.with_suffix(CACHE_EXT)
    df.to_pickle(path, compression="gzip")
    return path


def load_df(path_no_ext: Path) -> pd.DataFrame:
    path = path_no_ext.with_suffix(CACHE_EXT)
    return pd.read_pickle(path, compression="gzip")


def exists_df(path_no_ext: Path) -> bool:
    return path_no_ext.with_suffix(CACHE_EXT).exists()


def std_lon(arr):
    arr = np.asarray(arr, dtype="float64")
    arr = np.where(arr > 180, arr - 360, arr)
    return arr


def setup_matplotlib():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def find_primary_shp(shp_root: Path) -> Path:
    shps = sorted(shp_root.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"未在 {shp_root} 找到 .shp 文件")
    preferred = [s for s in shps if any(k in s.name.lower() for k in ["central", "asia", "admin", "boundary", "border"])]
    return preferred[0] if preferred else shps[0]


def prepare_study_area() -> gpd.GeoDataFrame:
    shp = find_primary_shp(SHP_ROOT)
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")
    return gdf


def safe_union(gdf: gpd.GeoDataFrame):
    try:
        return gdf.union_all()
    except Exception:
        return gdf.unary_union


def point_in_region(lat: float, lon: float, region_union) -> bool:
    from shapely.geometry import Point
    try:
        pt = Point(float(lon), float(lat))
        return region_union.buffer(0).contains(pt) or region_union.buffer(0).touches(pt)
    except Exception:
        return False


def normalize_columns(cols: List[str]) -> Dict[str, str]:
    cmap = {}
    for c in cols:
        cl = c.strip().lower()
        if cl in ["station", "station_id", "stn", "usaf_wban", "id"]:
            cmap[c] = "station"
        elif cl in ["date", "datetime", "time", "day"]:
            cmap[c] = "date"
        elif cl in ["lat", "latitude"]:
            cmap[c] = "lat"
        elif cl in ["lon", "longitude", "long"]:
            cmap[c] = "lon"
        elif cl in ["elev", "elevation", "alt"]:
            cmap[c] = "elev"
        elif cl in ["name", "station_name"]:
            cmap[c] = "name"
        elif cl in ["tmax", "tx", "max_temp"]:
            cmap[c] = "Tmax"
        elif cl in ["tmin", "tn", "min_temp"]:
            cmap[c] = "Tmin"
        elif cl in ["tavg", "tmean", "temp", "mean_temp"]:
            cmap[c] = "Tavg"
    return cmap


RAW_REQUIRED = {"station", "date", "lat", "lon", "Tmax", "Tmin"}
IGNORE_KEYWORDS = [
    "intermediate", "summary", "annual", "monthly", "threshold", "trend",
    "merged", "cache", "robust", "result", "extract", "figure", "plot",
    "station_meta", "era5_station", "gsod_daily", "scheme_", "skill_",
    "tx90", "tn90", "baseline"
]


def sniff_station_file(fp: Path) -> bool:
    name_low = fp.name.lower()
    if any(k in name_low for k in IGNORE_KEYWORDS):
        return False
    if fp.suffix.lower() not in [".csv", ".txt"]:
        return False
    try:
        head = pd.read_csv(fp, nrows=3)
    except Exception:
        try:
            head = pd.read_table(fp, nrows=3)
        except Exception:
            return False
    cmap = normalize_columns(list(head.columns))
    return RAW_REQUIRED.issubset(set(cmap.values()))


def discover_station_files(root: Path) -> List[Path]:
    files = [Path(p) for p in glob.glob(str(root / "**" / "*.*"), recursive=True)]
    matched = []
    for fp in tqdm(files, desc="扫描 GSOD 文件", unit="file"):
        if fp.is_file() and sniff_station_file(fp):
            matched.append(fp)
    matched = sorted(list(set(matched)))
    if not matched:
        raise FileNotFoundError(f"在 {root} 中没有识别到符合原始 GSOD 字段的文件")
    return matched


def read_one_station_file(fp: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(fp)
    except Exception:
        df = pd.read_table(fp)

    cmap = normalize_columns(list(df.columns))
    df = df.rename(columns=cmap)

    miss = RAW_REQUIRED - set(df.columns)
    if miss:
        raise ValueError(f"{fp.name} 缺少字段: {miss}")

    keep = [c for c in ["station", "date", "lat", "lon", "elev", "name", "Tmax", "Tmin", "Tavg"] if c in df.columns]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    for c in ["lat", "lon", "elev", "Tmax", "Tmin", "Tavg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["lon"] = std_lon(df["lon"].values)
    df["year"] = df["date"].dt.year.astype("int16")
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    if "Tavg" not in df.columns:
        df["Tavg"] = ((df["Tmax"] + df["Tmin"]) / 2.0).astype("float32")

    df = df[df[["station", "lat", "lon", "Tmax", "Tmin"]].notna().all(axis=1)].copy()
    if df.empty:
        return df

    df["station"] = df["station"].astype(str)
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)
    else:
        df["name"] = ""
    if "elev" not in df.columns:
        df["elev"] = np.nan

    for c in ["lat", "lon", "elev", "Tmax", "Tmin", "Tavg"]:
        df[c] = df[c].astype("float32")

    return df[["station", "date", "year", "lat", "lon", "elev", "name", "Tmax", "Tmin", "Tavg"]]


def prepare_gsod_and_meta(region_gdf: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gsod_cache = CACHE_DIR / "gsod_daily"
    meta_cache = CACHE_DIR / "station_meta"

    if exists_df(gsod_cache) and exists_df(meta_cache):
        return load_df(gsod_cache), load_df(meta_cache)

    station_files = discover_station_files(GSOD_ROOT)
    region_union = safe_union(region_gdf)

    chunk_paths = []
    meta_records = []
    chunk_list = []
    chunk_counter = 0

    for fp in tqdm(station_files, desc="读取 GSOD 原始站点文件", unit="file"):
        try:
            df = read_one_station_file(fp)
        except Exception as e:
            print(f"[WARN] 跳过 GSOD 文件 {fp.name}: {e}")
            continue
        if df.empty:
            continue

        stn = str(df["station"].iloc[0])
        lat_med = float(df["lat"].median())
        lon_med = float(df["lon"].median())

        if (abs(lat_med) < 1e-6 and abs(lon_med) < 1e-6) or (not point_in_region(lat_med, lon_med, region_union)):
            continue

        meta_records.append({
            "station": stn,
            "lat": np.float32(lat_med),
            "lon": np.float32(lon_med),
            "elev": np.float32(df["elev"].median()) if df["elev"].notna().any() else np.nan,
            "name": str(df["name"].mode().iloc[0]) if ("name" in df.columns and not df["name"].mode().empty) else "",
            "src_file": str(fp)
        })

        chunk_list.append(df)

        if len(chunk_list) >= 30:
            chunk_counter += 1
            chunk_df = pd.concat(chunk_list, ignore_index=True)
            chunk_df["month"] = chunk_df["date"].dt.month.astype("int8")
            chunk_df["doy"] = chunk_df["date"].dt.dayofyear.astype("int16")
            chunk_path = GSOD_CHUNK_DIR / f"gsod_chunk_{chunk_counter:03d}"
            save_df(chunk_df, chunk_path)
            chunk_paths.append(chunk_path)
            chunk_list.clear()
            del chunk_df
            gc.collect()

    if chunk_list:
        chunk_counter += 1
        chunk_df = pd.concat(chunk_list, ignore_index=True)
        chunk_df["month"] = chunk_df["date"].dt.month.astype("int8")
        chunk_df["doy"] = chunk_df["date"].dt.dayofyear.astype("int16")
        chunk_path = GSOD_CHUNK_DIR / f"gsod_chunk_{chunk_counter:03d}"
        save_df(chunk_df, chunk_path)
        chunk_paths.append(chunk_path)
        chunk_list.clear()
        del chunk_df
        gc.collect()

    if not chunk_paths:
        raise RuntimeError("未读到有效 GSOD 站点数据，请检查文件路径和字段")

    meta = pd.DataFrame(meta_records).drop_duplicates(subset=["station"]).copy()
    meta["station"] = meta["station"].astype(str)

    frames = []
    for cp in tqdm(chunk_paths, desc="拼接 GSOD 中间块", unit="chunk"):
        frames.append(load_df(cp))
    gsod = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    gsod = gsod.merge(meta[["station", "lat", "lon", "elev", "name"]], on="station", how="left", suffixes=("", "_meta"))
    for c in ["lat", "lon", "elev", "name"]:
        alt = f"{c}_meta"
        if alt in gsod.columns:
            gsod[c] = gsod[alt]
            gsod = gsod.drop(columns=[alt])

    gsod = gsod[["station", "date", "year", "month", "doy", "lat", "lon", "elev", "name", "Tmax", "Tmin", "Tavg"]]
    gsod = gsod.sort_values(["station", "date"]).reset_index(drop=True)

    save_df(gsod, gsod_cache)
    save_df(meta, meta_cache)
    meta.to_csv(TABLE_DIR / "station_meta.csv", index=False, encoding="utf-8-sig")
    return gsod, meta


def find_coord_name(ds: xr.Dataset, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"坐标 {candidates} 在数据中不存在")


def find_var_name(ds: xr.Dataset, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    for v in ds.data_vars:
        vl = v.lower()
        for c in candidates:
            if c.lower() in vl:
                return v
    raise KeyError(f"变量 {candidates} 在数据中不存在")


def expected_era5_daily_filename(year: int, month: int, utc_offset_hours: int = ERA5_LOCAL_UTC_OFFSET_HOURS) -> str:
    return f"era5_daily_t2m_{year}{month:02d}_utc{utc_offset_hours:+d}.nc"


def expected_era5_daily_path(year: int, month: int, utc_offset_hours: int = ERA5_LOCAL_UTC_OFFSET_HOURS) -> Path:
    return ERA5_DAILY_DIR / expected_era5_daily_filename(year, month, utc_offset_hours)


def monthly_key_from_name(fp: Path) -> Optional[str]:
    m = re.match(r"^era5_daily_t2m_(\d{6})_utc[+-]?\d+\.nc$", fp.name)
    if m:
        return m.group(1)
    return None


def validate_era5_daily_file(nc_path: Path) -> Tuple[bool, str]:
    if monthly_key_from_name(nc_path) is None:
        return False, "文件名不是标准 ERA5 日文件名"

    try:
        with xr.open_dataset(nc_path) as ds:
            try:
                _ = find_coord_name(ds, ["time", "valid_time", "date", "datetime"])
            except Exception:
                return False, "未找到时间坐标"

            try:
                _ = find_var_name(ds, ["t2m_daily_max", "daily_max", "tmax", "tasmax"])
                _ = find_var_name(ds, ["t2m_daily_min", "daily_min", "tmin", "tasmin"])
                _ = find_var_name(ds, ["t2m_daily_mean", "daily_mean", "tmean", "tavg", "tas"])
            except Exception:
                return False, "缺少 ERA5 日值变量"

        return True, "ok"
    except Exception as e:
        return False, f"无法打开或读取: {e}"


def build_expected_era5_daily_file_list() -> List[Path]:
    valid_files = []
    missing = []
    invalid = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            fp = expected_era5_daily_path(year, month)
            if not fp.exists():
                missing.append({"year": year, "month": month, "file": str(fp)})
                continue

            ok, msg = validate_era5_daily_file(fp)
            if ok:
                valid_files.append(fp)
            else:
                invalid.append({"year": year, "month": month, "file": str(fp), "reason": msg})

    if missing:
        pd.DataFrame(missing).to_csv(TABLE_DIR / "era5_missing_daily_files.csv", index=False, encoding="utf-8-sig")
    if invalid:
        pd.DataFrame(invalid).to_csv(TABLE_DIR / "era5_invalid_daily_files.csv", index=False, encoding="utf-8-sig")

    return valid_files


def extract_era5_one_month(nc_path: Path, station_meta: pd.DataFrame, out_path_no_ext: Path) -> None:
    with xr.open_dataset(nc_path) as ds:
        lat_name = find_coord_name(ds, ["latitude", "lat", "y"])
        lon_name = find_coord_name(ds, ["longitude", "lon", "x"])
        time_name = find_coord_name(ds, ["time", "valid_time", "date", "datetime"])

        vmax = find_var_name(ds, ["t2m_daily_max", "daily_max", "tmax", "tasmax"])
        vmin = find_var_name(ds, ["t2m_daily_min", "daily_min", "tmin", "tasmin"])
        vavg = find_var_name(ds, ["t2m_daily_mean", "daily_mean", "tmean", "tavg", "tas"])

        ds_lons = ds[lon_name].values
        station_lons = station_meta["lon"].values.astype("float64")
        if np.nanmin(ds_lons) >= 0 and np.nanmax(ds_lons) > 180:
            station_lons = np.where(station_lons < 0, station_lons + 360, station_lons)

        lat_da = xr.DataArray(station_meta["lat"].values.astype("float64"), dims="station", coords={"station": station_meta["station"].values})
        lon_da = xr.DataArray(station_lons.astype("float64"), dims="station", coords={"station": station_meta["station"].values})

        sub = xr.Dataset({
            "Tmax": ds[vmax].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest"),
            "Tmin": ds[vmin].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest"),
            "Tavg": ds[vavg].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest"),
        })

        df = sub.to_dataframe().reset_index()
        df = df.rename(columns={time_name: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["station"] = df["station"].astype(str)
        for c in ["Tmax", "Tmin", "Tavg"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

        df = df[["station", "date", "Tmax", "Tmin", "Tavg"]].copy()
        save_df(df, out_path_no_ext)


def prepare_era5_at_stations(station_meta: pd.DataFrame) -> pd.DataFrame:
    era5_cache = CACHE_DIR / "era5_daily_at_stations"
    if exists_df(era5_cache):
        return load_df(era5_cache)

    valid_nc_files = build_expected_era5_daily_file_list()
    if not valid_nc_files:
        raise FileNotFoundError(
            f"在 {ERA5_DAILY_DIR} 中没有找到任何有效的标准 ERA5 日文件。"
            f"\n请确认文件名形如：era5_daily_t2m_YYYYMM_utc{ERA5_LOCAL_UTC_OFFSET_HOURS:+d}.nc"
        )

    print(f"[INFO] 有效 ERA5 日文件数: {len(valid_nc_files)}")
    extra_nc_files = sorted(ERA5_DAILY_DIR.glob("*.nc"))
    skipped_extras = []
    valid_set = {str(p.resolve()) for p in valid_nc_files}
    for fp in extra_nc_files:
        if str(fp.resolve()) not in valid_set:
            skipped_extras.append({"file": str(fp), "reason": "非标准文件名或非 ERA5 日文件，已自动跳过"})
    if skipped_extras:
        pd.DataFrame(skipped_extras).to_csv(TABLE_DIR / "era5_skipped_extra_nc_files.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] 已跳过 {len(skipped_extras)} 个 daily 目录中的额外 nc 文件")

    for nc in tqdm(valid_nc_files, desc="ERA5 月文件提取到站点", unit="month"):
        month_key = monthly_key_from_name(nc)
        if month_key is None:

            continue
        out_no_ext = ERA5_MONTH_CACHE_DIR / f"era5_station_{month_key}"
        if exists_df(out_no_ext):
            continue
        try:
            extract_era5_one_month(nc, station_meta, out_no_ext)
        except Exception as e:
            print(f"[WARN] ERA5 提取失败 {nc.name}: {e}")

    parts = []
    monthly_pickles = sorted(ERA5_MONTH_CACHE_DIR.glob(f"*{CACHE_EXT}"))
    for fp in tqdm(monthly_pickles, desc="拼接 ERA5 站点月缓存", unit="file"):
        parts.append(pd.read_pickle(fp, compression="gzip"))

    if not parts:
        raise RuntimeError("ERA5 月缓存为空，无法继续")

    era5 = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    era5["year"] = era5["date"].dt.year.astype("int16")
    era5 = era5[(era5["year"] >= START_YEAR) & (era5["year"] <= END_YEAR)].copy()
    era5["month"] = era5["date"].dt.month.astype("int8")
    era5["doy"] = era5["date"].dt.dayofyear.astype("int16")
    era5 = era5.sort_values(["station", "date"]).reset_index(drop=True)

    save_df(era5, era5_cache)
    era5.to_csv(TABLE_DIR / "era5_daily_at_stations_preview.csv", index=False, encoding="utf-8-sig")
    return era5


def prepare_merged_daily(gsod: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    merged_cache = CACHE_DIR / "merged_daily"
    if exists_df(merged_cache):
        return load_df(merged_cache)

    merged = gsod.merge(
        era5[["station", "date", "Tmax", "Tmin", "Tavg"]],
        on=["station", "date"],
        how="inner",
        suffixes=("_gsod", "_era5")
    )

    merged = merged.sort_values(["station", "date"]).reset_index(drop=True)
    save_df(merged, merged_cache)
    return merged


def derive_hq_stations(gsod: pd.DataFrame) -> List[str]:
    hq_cache = TABLE_DIR / "hq_station_list.csv"
    if hq_cache.exists():
        return pd.read_csv(hq_cache)["station"].astype(str).tolist()

    tmp = gsod.copy()
    tmp["valid"] = tmp[["Tmax", "Tmin"]].notna().all(axis=1)
    ystat = tmp.groupby(["station", "year"])["valid"].sum().reset_index(name="valid_days")
    overall = ystat.groupby("station").agg(
        valid_years=("valid_days", lambda x: int((x >= HQ_MIN_VALID_DAYS_PER_YEAR).sum())),
        mean_valid_days=("valid_days", "mean")
    ).reset_index()
    overall["overall_completeness"] = overall["mean_valid_days"] / 365.25

    hq = overall[
        (overall["valid_years"] >= HQ_MIN_VALID_YEARS) &
        (overall["overall_completeness"] >= HQ_MIN_OVERALL_COMPLETENESS)
    ].copy()

    hq[["station", "valid_years", "mean_valid_days", "overall_completeness"]].to_csv(
        hq_cache, index=False, encoding="utf-8-sig"
    )
    return hq["station"].astype(str).tolist()


def get_source_daily(merged: pd.DataFrame, source: str, stations: Optional[List[str]] = None) -> pd.DataFrame:
    cols = ["station", "date", "year", "month", "doy", "lat", "lon"]
    src_cols = [f"Tmax_{source}", f"Tmin_{source}", f"Tavg_{source}"]
    df = merged[cols + src_cols].copy()
    df = df.rename(columns={f"Tmax_{source}": "Tmax", f"Tmin_{source}": "Tmin", f"Tavg_{source}": "Tavg"})
    if stations is not None:
        station_set = set(map(str, stations))
        df = df[df["station"].isin(station_set)].copy()
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    return df


def build_thresholds(source_daily: pd.DataFrame, scheme: dict, source: str) -> pd.DataFrame:
    thr_path = THR_DIR / f"thr_{source}_{scheme['id']}"
    if exists_df(thr_path):
        return load_df(thr_path)

    base = source_daily[
        (source_daily["year"] >= scheme["base_start"]) &
        (source_daily["year"] <= scheme["base_end"])
    ][["station", "doy", "Tmax", "Tmin"]].copy()

    q = float(scheme["q"])
    tmax_q = base.groupby(["station", "doy"])["Tmax"].quantile(q).reset_index(name="Tmax_thr")
    tmin_q = base.groupby(["station", "doy"])["Tmin"].quantile(q).reset_index(name="Tmin_thr")

    all_stations = sorted(base["station"].astype(str).unique().tolist())
    all_doys = np.arange(1, 367, dtype=np.int16)

    def complete_and_fill(dfq: pd.DataFrame, val_col: str) -> pd.DataFrame:
        wide = dfq.pivot(index="station", columns="doy", values=val_col).reindex(index=all_stations, columns=all_doys)
        wide = wide.astype("float32")
        wide = wide.interpolate(axis=1, limit_direction="both")
        wide = wide.ffill(axis=1).bfill(axis=1)
        out = wide.stack(dropna=False).rename(val_col).reset_index()
        out["doy"] = out["doy"].astype("int16")
        return out

    tmax_full = complete_and_fill(tmax_q, "Tmax_thr")
    tmin_full = complete_and_fill(tmin_q, "Tmin_thr")
    thr = tmax_full.merge(tmin_full, on=["station", "doy"], how="outer")
    save_df(thr, thr_path)
    return thr


def sen_slope(y: np.ndarray, x: Optional[np.ndarray] = None, per_decade: bool = True) -> float:
    y = np.asarray(y, dtype="float64")
    if x is None:
        x = np.arange(len(y), dtype="float64")
    else:
        x = np.asarray(x, dtype="float64")

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if n < 2:
        return np.nan

    idx_i, idx_j = np.triu_indices(n, k=1)
    slopes = (y[idx_j] - y[idx_i]) / (x[idx_j] - x[idx_i])
    slope = np.nanmedian(slopes)
    return float(slope * 10.0) if per_decade else float(slope)


def annual_metrics_for_one_station(stn_df: pd.DataFrame, thr_df_one: pd.DataFrame, min_run: int) -> pd.DataFrame:
    stn_df = stn_df.sort_values("date").copy()
    thr_map_max = thr_df_one.set_index("doy")["Tmax_thr"]
    thr_map_min = thr_df_one.set_index("doy")["Tmin_thr"]

    stn_df["Tmax_thr"] = stn_df["doy"].map(thr_map_max).astype("float32")
    stn_df["Tmin_thr"] = stn_df["doy"].map(thr_map_min).astype("float32")

    valid = stn_df[["Tmax", "Tmin", "Tmax_thr", "Tmin_thr"]].notna().all(axis=1).to_numpy()
    hot_max = valid & (stn_df["Tmax"].to_numpy() > stn_df["Tmax_thr"].to_numpy())
    hot_min = valid & (stn_df["Tmin"].to_numpy() > stn_df["Tmin_thr"].to_numpy())

    type_code = np.zeros(len(stn_df), dtype=np.int8)
    type_code[hot_max & ~hot_min] = 1
    type_code[hot_min & ~hot_max] = 2
    type_code[hot_max & hot_min] = 3

    exceed = np.zeros(len(stn_df), dtype="float32")
    tmax_exc = (stn_df["Tmax"].to_numpy() - stn_df["Tmax_thr"].to_numpy()).astype("float32")
    tmin_exc = (stn_df["Tmin"].to_numpy() - stn_df["Tmin_thr"].to_numpy()).astype("float32")
    exceed[type_code == 1] = tmax_exc[type_code == 1]
    exceed[type_code == 2] = tmin_exc[type_code == 2]
    exceed[type_code == 3] = ((tmax_exc[type_code == 3] + tmin_exc[type_code == 3]) / 2.0).astype("float32")

    day_gap = stn_df["date"].diff().dt.days.fillna(9999).to_numpy()
    grp_start = np.ones(len(stn_df), dtype=bool)
    if len(stn_df) > 1:
        grp_start[1:] = (type_code[1:] != type_code[:-1]) | (day_gap[1:] != 1)
    grp = np.cumsum(grp_start).astype("int32")

    tmp = stn_df[["station", "date", "year"]].copy()
    tmp["type_code"] = type_code
    tmp["exceed"] = exceed
    tmp["grp"] = grp

    grp_size = tmp.groupby("grp").size().rename("gsize")
    grp_type = tmp.groupby("grp")["type_code"].first().rename("gtype")
    grp_info = pd.concat([grp_size, grp_type], axis=1).reset_index()
    valid_grps = grp_info[(grp_info["gtype"] > 0) & (grp_info["gsize"] >= min_run)]["grp"]
    valid_set = set(valid_grps.tolist())

    ev = tmp[(tmp["type_code"] > 0) & (tmp["grp"].isin(valid_set))].copy()
    if ev.empty:
        years = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int16)
        out = pd.MultiIndex.from_product([[stn_df["station"].iloc[0]], years, TYPE_ORDER], names=["station", "year", "type"]).to_frame(index=False)
        out["HWF"] = 0.0
        out["HWD"] = 0.0
        out["HWI"] = 0.0
        return out

    ev["type"] = ev["type_code"].map(TYPE_CODE_TO_NAME)

    hwf = ev.groupby(["year", "type"]).size().rename("HWF").reset_index()
    hwi = ev.groupby(["year", "type"])["exceed"].mean().rename("HWI").reset_index()
    hwd = (
        ev.groupby(["year", "type", "grp"]).size().rename("runlen").reset_index()
        .groupby(["year", "type"])["runlen"].max().rename("HWD").reset_index()
    )

    out = hwf.merge(hwd, on=["year", "type"], how="outer").merge(hwi, on=["year", "type"], how="outer")
    out["station"] = stn_df["station"].iloc[0]

    years = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int16)
    skeleton = pd.MultiIndex.from_product([[stn_df["station"].iloc[0]], years, TYPE_ORDER], names=["station", "year", "type"]).to_frame(index=False)
    out = skeleton.merge(out, on=["station", "year", "type"], how="left")

    for c in ["HWF", "HWD", "HWI"]:
        out[c] = out[c].fillna(0).astype("float32")
    out["year"] = out["year"].astype("int16")
    return out


def run_one_scheme(source_daily: pd.DataFrame, scheme: dict, source: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scheme_root = SCHEME_DIR / scheme["id"]
    scheme_root.mkdir(parents=True, exist_ok=True)
    annual_path = scheme_root / f"annual_{source}"
    regional_path = scheme_root / f"regional_{source}"
    station_slope_path = scheme_root / f"station_slopes_{source}"

    if exists_df(annual_path) and exists_df(regional_path) and exists_df(station_slope_path):
        return load_df(annual_path), load_df(regional_path), load_df(station_slope_path)

    thr = build_thresholds(source_daily, scheme, source)
    thr_group = {stn: sub[["doy", "Tmax_thr", "Tmin_thr"]].copy() for stn, sub in thr.groupby("station")}

    annual_parts = []
    for stn, sub in tqdm(source_daily.groupby("station", sort=False), desc=f"{source.upper()}-{scheme['id']} 事件识别", unit="station"):
        if stn not in thr_group:
            continue
        annual_parts.append(annual_metrics_for_one_station(sub, thr_group[stn], int(scheme["min_run"])))

    annual_df = pd.concat(annual_parts, ignore_index=True)
    annual_df["source"] = source
    annual_df["scheme"] = scheme["id"]

    reg_year = annual_df.groupby(["year", "type"])[METRIC_ORDER].mean().reset_index()
    reg_summ = []
    for tp in TYPE_ORDER:
        sub = reg_year[reg_year["type"] == tp].sort_values("year")
        yrs = sub["year"].to_numpy()
        for m in METRIC_ORDER:
            reg_summ.append({
                "source": source,
                "scheme": scheme["id"],
                "type": tp,
                "metric": m,
                "sen_slope": sen_slope(sub[m].to_numpy(), yrs, per_decade=True)
            })
    regional_df = pd.DataFrame(reg_summ)

    slope_parts = []
    for (stn, tp), sub in tqdm(annual_df.groupby(["station", "type"], sort=False), desc=f"{source.upper()}-{scheme['id']} 站点趋势", unit="series"):
        sub = sub.sort_values("year")
        yrs = sub["year"].to_numpy()
        for m in METRIC_ORDER:
            slope_parts.append({
                "station": stn,
                "type": tp,
                "metric": m,
                "sen_slope": sen_slope(sub[m].to_numpy(), yrs, per_decade=True),
                "source": source,
                "scheme": scheme["id"]
            })
    station_slope_df = pd.DataFrame(slope_parts)

    save_df(annual_df, annual_path)
    save_df(regional_df, regional_path)
    save_df(station_slope_df, station_slope_path)
    return annual_df, regional_df, station_slope_df


def calc_type_shares(annual_df: pd.DataFrame, source: str, scheme_id: str) -> pd.DataFrame:
    out = []
    for pname, y0, y1 in [("P1", START_YEAR, P1_END), ("P2", P2_START, END_YEAR)]:
        sub = annual_df[(annual_df["year"] >= y0) & (annual_df["year"] <= y1)].copy()
        totals = sub.groupby("type")["HWF"].sum()
        denom = totals.sum()
        for tp in TYPE_ORDER:
            share = float(totals.get(tp, 0.0) / denom) if denom > 0 else np.nan
            out.append({"source": source, "scheme": scheme_id, "period": pname, "type": tp, "share": share})
    return pd.DataFrame(out)


def calc_station_agreement(s0_df: pd.DataFrame, sx_df: pd.DataFrame, meta: pd.DataFrame, source: str, scheme_id: str) -> pd.DataFrame:
    tmp = s0_df.merge(
        sx_df,
        on=["station", "type", "metric", "source"],
        suffixes=("_s0", "_sx"),
        how="inner"
    )

    def sign_equiv(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            return 1.0
        return float(np.sign(a) == np.sign(b))

    tmp["agree"] = [sign_equiv(a, b) for a, b in zip(tmp["sen_slope_s0"], tmp["sen_slope_sx"])]
    out = tmp.groupby("station", as_index=False)["agree"].mean()
    out = out.rename(columns={"agree": "agreement_frac"})
    out["scheme"] = scheme_id
    out["source"] = source
    out = out.merge(meta[["station", "lat", "lon"]], on="station", how="left")
    return out


def calc_skill_table(annual_gsod: pd.DataFrame, annual_era5: pd.DataFrame, scheme_id: str) -> pd.DataFrame:
    merged = annual_gsod.merge(
        annual_era5,
        on=["station", "year", "type", "scheme"],
        suffixes=("_gsod", "_era5"),
        how="inner"
    )

    rows = []
    for m in METRIC_ORDER:
        x = merged[f"{m}_gsod"].to_numpy(dtype="float64")
        y = merged[f"{m}_era5"].to_numpy(dtype="float64")
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            rows.append({"scheme": scheme_id, "metric": m, "r": np.nan, "Bias": np.nan, "RMSE": np.nan, "POD": np.nan, "FAR": np.nan, "CSI": np.nan})
            continue

        r = np.corrcoef(x, y)[0, 1] if (np.nanstd(x) > 0 and np.nanstd(y) > 0) else np.nan
        diff = y - x
        bias = np.nanmean(diff)
        rmse = np.sqrt(np.nanmean(diff ** 2))

        obs = x > 0
        sim = y > 0
        hit = np.sum(obs & sim)
        miss = np.sum(obs & (~sim))
        fa = np.sum((~obs) & sim)

        pod = hit / (hit + miss) if (hit + miss) > 0 else np.nan
        far = fa / (hit + fa) if (hit + fa) > 0 else np.nan
        csi = hit / (hit + miss + fa) if (hit + miss + fa) > 0 else np.nan

        rows.append({
            "scheme": scheme_id,
            "metric": m,
            "r": r,
            "Bias": bias,
            "RMSE": rmse,
            "POD": pod,
            "FAR": far,
            "CSI": csi
        })

    return pd.DataFrame(rows)


def plot_fig1_regional_heatmaps(regional_all: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.14, wspace=0.24, hspace=0.22)

    for i, source in enumerate(SOURCE_ORDER):
        for j, metric in enumerate(METRIC_ORDER):
            ax = axes[i, j]
            sub = regional_all[(regional_all["source"] == source) & (regional_all["metric"] == metric)].copy()
            mat = sub.pivot(index="scheme", columns="type", values="sen_slope").reindex(
                index=[s["id"] for s in SCHEMES],
                columns=TYPE_ORDER
            )

            vmax = np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1
            im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

            nrow, ncol = mat.shape
            for r in range(nrow):
                for c in range(ncol):
                    val = mat.values[r, c]
                    txt = "NA" if np.isnan(val) else (f"{val:.3f}" if metric == "HWI" else f"{val:.2f}")
                    ax.text(c, r, txt, ha="center", va="center", fontsize=8)

            ax.set_xticks(range(len(TYPE_ORDER)))
            ax.set_xticklabels(TYPE_ORDER)
            ax.set_yticks(range(len(SCHEMES)))
            ax.set_yticklabels([s["id"] for s in SCHEMES])
            ax.set_title(f"{source.upper()} - {metric}")
            if j == 0:
                ax.set_ylabel("Scheme")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4.2%", pad=0.16)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.set_ylabel("Sen slope / 10yr")

    scheme_text = " | ".join([
        f"{s['id']}: q={int(s['q']*100)}, base={s['base_start']}-{s['base_end']}, run≥{s['min_run']}d, set={s['station_set']}"
        for s in SCHEMES
    ])
    fig.suptitle("Fig.1 Robustness of regional-mean trends across schemes", fontsize=14, y=0.96)
    fig.text(0.5, 0.04, scheme_text, ha="center", va="top", fontsize=9)
    fig.savefig(FIG_DIR / "Fig1_regional_trend_robustness.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_spatial_agreement(agree_all: pd.DataFrame, region_gdf: gpd.GeoDataFrame):
    minx, miny, maxx, maxy = region_gdf.total_bounds
    xpad = (maxx - minx) * 0.04
    ypad = (maxy - miny) * 0.06
    mean_lat = (miny + maxy) / 2.0
    geo_aspect = 1.0 / np.cos(np.deg2rad(mean_lat))

    panel_order = [
        ("gsod", "S1"),
        ("gsod", "S4"),
        ("gsod", "S6"),
        ("era5", "S1"),
        ("era5", "S4"),
        ("era5", "S6"),
    ]
    panel_letters = ["a", "b", "c", "d", "e", "f"]

    for letter, (source, scheme_id) in zip(panel_letters, panel_order):
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), constrained_layout=False)
        plt.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.12)

        region_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

        sub = agree_all[(agree_all["source"] == source) & (agree_all["scheme"] == scheme_id)].copy()
        sc = ax.scatter(
            sub["lon"], sub["lat"],
            c=sub["agreement_frac"],
            s=POINT_SIZE,
            cmap="viridis",
            vmin=0,
            vmax=1
        )

        ax.set_xlim(minx - xpad, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)
        ax.set_aspect(geo_aspect, adjustable="box")
        ax.set_title(f"{source.upper()} {scheme_id} vs S0")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(ls=":", lw=0.3)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.2%", pad=0.10)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_ylabel("Sign agreement fraction")

        out_name = f"Fig2{letter}_{source.upper()}_{scheme_id}_vs_S0.png"
        fig.savefig(FIG2_DIR / out_name, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


def plot_fig3_type_share(type_share_all: pd.DataFrame):
    colors = {"CL": "#d73027", "DL": "#4575b4", "NL": "#74add1"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=False, sharey=True)
    plt.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.17, wspace=0.10, hspace=0.22)
    periods = ["P1", "P2"]
    legend_handles = None

    for i, source in enumerate(SOURCE_ORDER):
        for j, period in enumerate(periods):
            ax = axes[i, j]
            sub = type_share_all[(type_share_all["source"] == source) & (type_share_all["period"] == period)].copy()
            pv = sub.pivot(index="scheme", columns="type", values="share").reindex(
                index=[s["id"] for s in SCHEMES],
                columns=TYPE_ORDER
            )

            bottom = np.zeros(len(pv), dtype="float64")
            x = np.arange(len(pv))
            local_handles = []
            for tp in TYPE_ORDER:
                vals = pv[tp].fillna(0).to_numpy()
                bars = ax.bar(x, vals, bottom=bottom, color=colors[tp], label=tp)
                local_handles.append(bars[0])
                bottom += vals

            if legend_handles is None:
                legend_handles = local_handles

            ax.set_xticks(x)
            ax.set_xticklabels(pv.index)
            ax.set_ylim(0, 1)
            ax.set_title(f"{source.upper()} - {period}")
            ax.set_ylabel("Share in total HWF")
            ax.set_xlabel("Scheme")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            TYPE_ORDER,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.06)
        )

    fig.suptitle("Fig.3 Robustness of DL/NL/CL HWF shares before and after 2000", fontsize=14, y=0.96)
    fig.savefig(FIG_DIR / "Fig3_type_share_robustness.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_skill(skill_all: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=False)
    plt.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.12, wspace=0.34)
    stat_list = ["r", "Bias", "RMSE"]
    cmaps = {"r": "viridis", "Bias": "RdBu_r", "RMSE": "magma"}

    for ax, stat in zip(axes.flat, stat_list):
        mat = skill_all.pivot(index="scheme", columns="metric", values=stat).reindex(
            index=[s["id"] for s in SCHEMES],
            columns=METRIC_ORDER
        )

        if stat == "Bias":
            vmax = np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1
            im = ax.imshow(mat.values, aspect="auto", cmap=cmaps[stat], vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(mat.values, aspect="auto", cmap=cmaps[stat])

        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat.values[r, c]
                txt = "NA" if np.isnan(val) else f"{val:.2f}"
                txt_color = "black"
                if stat == "RMSE" and c in [1, 2]:
                    txt_color = "white"
                ax.text(c, r, txt, ha="center", va="center", fontsize=8, color=txt_color)

        ax.set_xticks(range(len(METRIC_ORDER)))
        ax.set_xticklabels(METRIC_ORDER)
        ax.set_yticks(range(len(SCHEMES)))
        ax.set_yticklabels([s["id"] for s in SCHEMES])
        ax.set_title(stat)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(stat)

    fig.suptitle("Fig.4 ERA5 skill relative to GSOD across robustness schemes", fontsize=14, y=0.95)
    fig.savefig(FIG_DIR / "Fig4_era5_vs_gsod_skill.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_matplotlib()
    region_gdf = prepare_study_area()

    print("\n[1/8] 读取并整理 GSOD 原始站点数据...")
    gsod, meta = prepare_gsod_and_meta(region_gdf)
    print(f"GSOD rows = {len(gsod):,}; stations = {gsod['station'].nunique()}")

    print("\n[2/8] 读取 ERA5 月 nc 并提取到站点...")
    era5 = prepare_era5_at_stations(meta)
    print(f"ERA5 extracted rows = {len(era5):,}; stations = {era5['station'].nunique()}")

    print("\n[3/8] 合并 GSOD 与 ERA5 日尺度数据...")
    merged = prepare_merged_daily(gsod, era5)
    print(f"Merged rows = {len(merged):,}; common stations = {merged['station'].nunique()}")

    print("\n[4/8] 生成高质量站点子集...")
    hq_stations = derive_hq_stations(gsod)
    print(f"HQ stations = {len(hq_stations)}")

    pd.DataFrame(SCHEMES).to_csv(TABLE_DIR / "scheme_design.csv", index=False, encoding="utf-8-sig")

    regional_all = []
    type_share_all = []
    skill_all = []
    station_slope_dict = {}

    print("\n[5/8] 按 scheme 运行事件识别与年尺度指标计算...")
    for scheme in SCHEMES:
        print(f"\n--- Running {scheme['id']} ---")
        scheme_stations = hq_stations if scheme["station_set"] == "hq" else None

        annual_store = {}
        for source in SOURCE_ORDER:
            source_daily = get_source_daily(merged, source, scheme_stations)
            annual_df, regional_df, station_slope_df = run_one_scheme(source_daily, scheme, source)

            regional_all.append(regional_df)
            type_share_all.append(calc_type_shares(annual_df, source, scheme["id"]))
            station_slope_dict[(scheme["id"], source)] = station_slope_df
            annual_store[source] = annual_df

            out_csv = TABLE_DIR / f"annual_metrics_{scheme['id']}_{source}.csv"
            if not out_csv.exists():
                annual_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            del source_daily, annual_df, regional_df, station_slope_df
            gc.collect()

        skill_all.append(calc_skill_table(annual_store["gsod"], annual_store["era5"], scheme["id"]))
        del annual_store
        gc.collect()

    regional_all = pd.concat(regional_all, ignore_index=True)
    type_share_all = pd.concat(type_share_all, ignore_index=True)
    skill_all = pd.concat(skill_all, ignore_index=True)

    regional_all.to_csv(TABLE_DIR / "regional_summary_all_schemes.csv", index=False, encoding="utf-8-sig")
    type_share_all.to_csv(TABLE_DIR / "type_share_all_schemes.csv", index=False, encoding="utf-8-sig")
    skill_all.to_csv(TABLE_DIR / "era5_skill_all_schemes.csv", index=False, encoding="utf-8-sig")

    print("\n[6/8] 计算空间符号一致率（S1/S4/S6 相对 S0）...")
    agree_parts = []
    for source in SOURCE_ORDER:
        s0 = station_slope_dict[("S0", source)]
        for sid in SELECTED_MAP_SCHEMES:
            sx = station_slope_dict[(sid, source)]
            agree_parts.append(calc_station_agreement(s0, sx, meta, source, sid))
    agree_all = pd.concat(agree_parts, ignore_index=True)
    agree_all.to_csv(TABLE_DIR / "station_sign_agreement.csv", index=False, encoding="utf-8-sig")

    print("\n[7/8] 绘制 4 张稳健性图...")
    plot_fig1_regional_heatmaps(regional_all)
    plot_fig2_spatial_agreement(agree_all, region_gdf)
    plot_fig3_type_share(type_share_all)
    plot_fig4_skill(skill_all)

    print("\n[8/8] 完成。输出路径如下：")
    print(f"  图件目录: {FIG_DIR}")
    print(f"  Fig2子图目录: {FIG2_DIR}")
    print(f"  表格目录: {TABLE_DIR}")
    print(f"  缓存目录: {CACHE_DIR}")

    summary = {
        "figures": [
            str(FIG_DIR / "Fig1_regional_trend_robustness.png"),
            str(FIG2_DIR / "Fig2a_GSOD_S1_vs_S0.png"),
            str(FIG2_DIR / "Fig2b_GSOD_S4_vs_S0.png"),
            str(FIG2_DIR / "Fig2c_GSOD_S6_vs_S0.png"),
            str(FIG2_DIR / "Fig2d_ERA5_S1_vs_S0.png"),
            str(FIG2_DIR / "Fig2e_ERA5_S4_vs_S0.png"),
            str(FIG2_DIR / "Fig2f_ERA5_S6_vs_S0.png"),
            str(FIG_DIR / "Fig3_type_share_robustness.png"),
            str(FIG_DIR / "Fig4_era5_vs_gsod_skill.png"),
        ],
        "tables": [
            str(TABLE_DIR / "scheme_design.csv"),
            str(TABLE_DIR / "regional_summary_all_schemes.csv"),
            str(TABLE_DIR / "type_share_all_schemes.csv"),
            str(TABLE_DIR / "era5_skill_all_schemes.csv"),
            str(TABLE_DIR / "station_sign_agreement.csv"),
            str(TABLE_DIR / "hq_station_list.csv"),
            str(TABLE_DIR / "era5_skipped_extra_nc_files.csv"),
            str(TABLE_DIR / "era5_missing_daily_files.csv"),
            str(TABLE_DIR / "era5_invalid_daily_files.csv"),
        ]
    }

    with open(LOG_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
