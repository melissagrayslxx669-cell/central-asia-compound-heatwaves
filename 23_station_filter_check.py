import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


STATION_DIR = r"G:\CYX\Analyse\normalized_by_station"
CA_DIR = r"G:\Central Asia"
OUT_DIR = r"G:\CYX\Analyse\figs\station_filter_check"

YEAR_MIN_REQUIRED = 1950


DATE_COL_CANDIDATES = ["date", "time", "datetime", "DATE", "TIME", "Date", "Time"]
LON_COL_CANDIDATES  = ["lon", "longitude", "LON", "LONGITUDE", "Lon", "Longitude", "x", "X"]
LAT_COL_CANDIDATES  = ["lat", "latitude", "LAT", "LATITUDE", "Lat", "Latitude", "y", "Y"]


def pick_first_existing_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def read_station_file_minimal(fp: Path):
    suffix = fp.suffix.lower()

    if suffix == ".csv":

        head = pd.read_csv(fp, nrows=5)
        date_col = pick_first_existing_col(head.columns, DATE_COL_CANDIDATES)
        lon_col  = pick_first_existing_col(head.columns, LON_COL_CANDIDATES)
        lat_col  = pick_first_existing_col(head.columns, LAT_COL_CANDIDATES)

        if date_col is None:
            raise ValueError(f"[{fp.name}] 未找到日期列。请在 DATE_COL_CANDIDATES 中补充你的列名。")

        usecols = [date_col]
        if lon_col is not None: usecols.append(lon_col)
        if lat_col is not None: usecols.append(lat_col)

        df = pd.read_csv(fp, usecols=list(dict.fromkeys(usecols)))

    elif suffix == ".parquet":
        df0 = pd.read_parquet(fp)
        date_col = pick_first_existing_col(df0.columns, DATE_COL_CANDIDATES)
        lon_col  = pick_first_existing_col(df0.columns, LON_COL_CANDIDATES)
        lat_col  = pick_first_existing_col(df0.columns, LAT_COL_CANDIDATES)

        if date_col is None:
            raise ValueError(f"[{fp.name}] 未找到日期列。请在 DATE_COL_CANDIDATES 中补充你的列名。")

        keep = [date_col]
        if lon_col is not None: keep.append(lon_col)
        if lat_col is not None: keep.append(lat_col)
        df = df0[keep].copy()

    else:
        raise ValueError(f"不支持的文件格式：{fp}")


    dt = pd.to_datetime(df[date_col], errors="coerce")
    years = dt.dt.year.dropna().astype(int)

    if years.empty:
        year_min = np.nan
        year_max = np.nan
        n_years_valid = 0
    else:
        year_min = int(years.min())
        year_max = int(years.max())
        n_years_valid = int(years.nunique())

    n_years_span = (year_max - year_min + 1) if pd.notna(year_min) and pd.notna(year_max) else 0


    lon = np.nan
    lat = np.nan
    if lon_col is not None:
        s = pd.to_numeric(df[lon_col], errors="coerce").dropna()
        if not s.empty:
            lon = float(s.iloc[0])
    if lat_col is not None:
        s = pd.to_numeric(df[lat_col], errors="coerce").dropna()
        if not s.empty:
            lat = float(s.iloc[0])

    return {
        "station_id": fp.stem,
        "file": str(fp),
        "year_min": year_min,
        "year_max": year_max,
        "n_years_span": int(n_years_span) if n_years_span else 0,
        "n_years_valid": int(n_years_valid),
        "lon": lon,
        "lat": lat,
        "n_rows": int(len(df)),
    }

def load_ca_outer_polygon(ca_dir: str):
    shp_list = sorted(glob.glob(os.path.join(ca_dir, "*.shp")))
    if not shp_list:
        raise FileNotFoundError(f"未在 {ca_dir} 找到 *.shp")

    ca = gpd.read_file(shp_list[0]).to_crs("EPSG:4326")

    _ca = ca.copy()
    _ca["__g__"] = 1
    ca_outer = _ca.dissolve(by="__g__")

    ca_outer["geometry"] = ca_outer.geometry.buffer(0)

    return ca_outer, shp_list[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)


    ca_outer, ca_shp_used = load_ca_outer_polygon(CA_DIR)
    print(f"[INFO] CA shp used: {ca_shp_used}")


    station_files = []
    station_files += sorted(Path(STATION_DIR).glob("*.csv"))
    station_files += sorted(Path(STATION_DIR).glob("*.parquet"))

    if not station_files:
        raise FileNotFoundError(f"在 {STATION_DIR} 未找到 *.csv 或 *.parquet 站点文件。")

    print(f"[INFO] Found station files: {len(station_files)}")


    rows = []
    bad = 0
    for i, fp in enumerate(station_files, 1):
        try:
            rows.append(read_station_file_minimal(fp))
        except Exception as e:
            bad += 1
            print(f"[WARN] skip {fp.name}: {e}")

        if i % 100 == 0:
            print(f"[INFO] processed {i}/{len(station_files)}")

    summary = pd.DataFrame(rows)
    summary_out = os.path.join(OUT_DIR, "station_summary_from_normalized.csv")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"[DONE] summary saved: {summary_out}")
    print(f"[INFO] bad files skipped: {bad}")


    df = summary.copy()
    df = df[df["year_min"].fillna(0) >= YEAR_MIN_REQUIRED].copy()
    df = df.dropna(subset=["lon", "lat"])
    df = df[df["n_years_span"].fillna(0) > 0].copy()

    print(f"[INFO] After time+coord filter: {len(df)}")


    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf_clip = gpd.clip(gdf, ca_outer)

    print(f"[RESULT] Stations within CA after clip: {len(gdf_clip)} (since {YEAR_MIN_REQUIRED})")


    out_csv = os.path.join(OUT_DIR, f"station_filtered_CA_since{YEAR_MIN_REQUIRED}.csv")
    gdf_clip.drop(columns="geometry").to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_gpkg = os.path.join(OUT_DIR, f"station_filtered_CA_since{YEAR_MIN_REQUIRED}.gpkg")
    gdf_clip.to_file(out_gpkg, layer="stations", driver="GPKG")

    print(f"[DONE] filtered list saved: {out_csv}")
    print(f"[DONE] gpkg saved: {out_gpkg}")


    cover = gdf_clip["n_years_span"].astype(float)
    if len(cover) > 0:
        print(f"[INFO] Coverage median (years): {np.nanmedian(cover):.0f}")

if __name__ == "__main__":
    main()
