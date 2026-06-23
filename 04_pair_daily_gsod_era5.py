import os, glob, re
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


GSOD_ROOT = r"G:\CYX\GSOD\NCEIgsod"
ERA5_ROOT = r"G:\CYX\ERA5"
OUT_DIR   = r"G:\CYX\Analyse\pair_daily"
os.makedirs(OUT_DIR, exist_ok=True)


YEAR_MIN = 1950
YEAR_MAX = 2024


ERA5_PATTERN = os.path.join(ERA5_ROOT, "*.nc")

ERA5_VARS = dict(
    tmax=["tmax","t2m_max","mx2t"],
    tmin=["tmin","t2m_min","mn2t"],
    tmean=["tmean","t2m_mean","t2m"],
)

ERA5_LAT_NAMES = ["lat","latitude"]
ERA5_LON_NAMES = ["lon","longitude"]


GSOD_DATE_CAND  = ["date","DATE","ymd","time","Time","datetime","DATE_time"]
GSOD_TMAX_CAND  = ["TMAX","tmax","TX","tx","maxtemp","maxt","TMAX_C","tmax_c"]
GSOD_TMIN_CAND  = ["TMIN","tmin","TN","tn","mintemp","mint","TMIN_C","tmin_c"]
GSOD_TMEAN_CAND = ["TAVG","tavg","TMEAN","tmean","TAVE","tave","TEMP","temp"]
GSOD_LAT_CAND   = ["LAT","lat","latitude","LATITUDE"]
GSOD_LON_CAND   = ["LON","lon","longitude","LONGITUDE"]
GSOD_ID_CAND    = ["STATION","station","USAF","USAF_WBAN","ID","station_id","WMO","wmo"]


def _pick(col_list, df_cols):
    lower = {c.lower():c for c in df_cols}
    for c in col_list:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _to_celsius(s: pd.Series):
    v = s.dropna().values
    if v.size == 0:
        return s.astype(float)
    med = np.nanmedian(v)

    if med > 80:
        return (s - 32.0) * (5.0/9.0)
    return s.astype(float)

def read_gsod_one_csv(fp):
    try:
        df = pd.read_csv(fp, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(fp, low_memory=False, encoding="latin1")

    c_date  = _pick(GSOD_DATE_CAND,  df.columns)
    c_tmax  = _pick(GSOD_TMAX_CAND,  df.columns)
    c_tmin  = _pick(GSOD_TMIN_CAND,  df.columns)
    c_tmean = _pick(GSOD_TMEAN_CAND, df.columns)
    c_lat   = _pick(GSOD_LAT_CAND,   df.columns)
    c_lon   = _pick(GSOD_LON_CAND,   df.columns)
    c_id    = _pick(GSOD_ID_CAND,    df.columns)

    if c_date is None:
        raise KeyError(f"{os.path.basename(fp)}: cannot find date column.")
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=[c_date]).copy()


    df = df[df[c_date].dt.year >= YEAR_MIN]
    if df.empty:
        return None


    if c_tmax in df:  df[c_tmax]  = _to_celsius(df[c_tmax])
    if c_tmin in df:  df[c_tmin]  = _to_celsius(df[c_tmin])
    if c_tmean in df: df[c_tmean] = _to_celsius(df[c_tmean])


    out = pd.DataFrame({
        "date": df[c_date].values
    })
    if c_tmax in df:  out["tmax_gsod"]  = df[c_tmax].astype(float).values
    if c_tmin in df:  out["tmin_gsod"]  = df[c_tmin].astype(float).values
    if c_tmean in df: out["tmean_gsod"] = df[c_tmean].astype(float).values
    if c_lat in df:   out["lat"]        = pd.to_numeric(df[c_lat], errors="coerce").values
    if c_lon in df:   out["lon"]        = pd.to_numeric(df[c_lon], errors="coerce").values
    if c_id in df:    out["station"]    = df[c_id].astype(str).values

    return out

def load_all_gsod():
    years = [d for d in glob.glob(os.path.join(GSOD_ROOT, "*")) if os.path.isdir(d)]

    yrs = []
    for d in years:
        m = re.findall(r"(\d{4})", os.path.basename(d))
        if m:
            y = int(m[0])
            if YEAR_MIN <= y <= YEAR_MAX:
                yrs.append((y, d))
    yrs.sort()
    by_station = {}
    for y, d in tqdm(yrs, desc="Reading GSOD yearly folders"):
        for fp in glob.glob(os.path.join(d, "*.csv")):
            dat = read_gsod_one_csv(fp)
            if dat is None:
                continue

            if "station" not in dat.columns:

                sid = os.path.splitext(os.path.basename(fp))[0]
                dat["station"] = sid

            for sid, dsi in dat.groupby(dat["station"].astype(str)):
                sub = dsi.dropna(subset=["date"]).copy()

                if "lat" not in sub or sub["lat"].isna().all():
                    lat = np.nan
                else:
                    lat = float(sub["lat"].dropna().median())
                if "lon" not in sub or sub["lon"].isna().all():
                    lon = np.nan
                else:
                    lon = float(sub["lon"].dropna().median())
                sub["lat"] = lat
                sub["lon"] = lon
                keep_cols = ["date","station","lat","lon","tmax_gsod","tmin_gsod","tmean_gsod"]
                sub = sub[[c for c in keep_cols if c in sub.columns]].copy()
                sub = sub.sort_values("date").drop_duplicates(subset=["date"])

                if sid not in by_station:
                    by_station[sid] = sub
                else:
                    by_station[sid] = pd.concat([by_station[sid], sub], ignore_index=True)

    for sid, df in by_station.items():
        df = df.sort_values("date").drop_duplicates(subset=["date"])
        by_station[sid] = df
    return by_station

def open_era5():
    files = sorted(glob.glob(ERA5_PATTERN))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {ERA5_ROOT}")
    ds = xr.open_mfdataset(files, combine="by_coords")

    lat_name = None; lon_name = None
    for nm in ERA5_LAT_NAMES:
        if nm in ds.coords: lat_name = nm; break
    for nm in ERA5_LON_NAMES:
        if nm in ds.coords: lon_name = nm; break
    if lat_name is None or lon_name is None:
        raise KeyError("Cannot find lat/lon coordinate names in ERA5 files.")


    def pick_var(cands):
        for v in cands:
            if v in ds.variables:
                return v
        return None
    v_tmax  = pick_var(ERA5_VARS["tmax"])
    v_tmin  = pick_var(ERA5_VARS["tmin"])
    v_tmean = pick_var(ERA5_VARS["tmean"])
    if v_tmax is None or v_tmin is None or v_tmean is None:
        raise KeyError("ERA5 variable names not found. Please update ERA5_VARS mapping.")


    def to_c(dsvar):

        meanv = float(dsvar.isel({lat_name: dsvar.sizes[lat_name]//2,
                                  lon_name: dsvar.sizes[lon_name]//2,
                                  "time": 0}).values)
        if meanv > 150:
            return dsvar - 273.15
        return dsvar

    ds = ds.rename({lat_name:"lat", lon_name:"lon"})
    tmax  = to_c(ds[v_tmax]).rename("tmax_era5")
    tmin  = to_c(ds[v_tmin]).rename("tmin_era5")
    tmean = to_c(ds[v_tmean]).rename("tmean_era5")
    ds2 = xr.merge([tmax, tmin, tmean])


    ds2 = ds2.sel(time=slice(f"{YEAR_MIN}-01-01", f"{YEAR_MAX}-12-31"))
    return ds2

def extract_era5_for_station(ds, lat, lon):
    if np.isnan(lat) or np.isnan(lon):
        return None

    if ds["lon"].max() > 180 and lon < 0:
        lon = lon % 360.0
    sub = ds.interp(lat=float(lat), lon=float(lon))
    df = sub.to_dataframe().reset_index()
    df = df.rename(columns={"time":"date"})
    return df[["date","tmax_era5","tmin_era5","tmean_era5"]]


if __name__ == "__main__":
    print("[INFO] Scanning GSOD (1950+) ...")
    by_station = load_all_gsod()
    print(f"[INFO] GSOD stations (>=1950): {len(by_station)}")

    print("[INFO] Opening ERA5 daily dataset ...")
    ds_era = open_era5()

    for sid, gdf in tqdm(by_station.items(), desc="Pairing stations"):
        if gdf.empty:
            continue

        lat = float(gdf["lat"].dropna().median()) if "lat" in gdf else np.nan
        lon = float(gdf["lon"].dropna().median()) if "lon" in gdf else np.nan
        gdf = gdf[ gdf["date"].dt.year >= YEAR_MIN ].copy()
        gdf = gdf.sort_values("date")
        gdf["station"] = str(sid)

        era = extract_era5_for_station(ds_era, lat, lon)
        if era is None or era.empty:
            continue

        pair = pd.merge(gdf, era, on="date", how="inner")

        pair["lat"] = lat
        pair["lon"] = lon

        cols = ["date","station","lat","lon",
                "tmax_gsod","tmin_gsod","tmean_gsod",
                "tmax_era5","tmin_era5","tmean_era5"]
        cols = [c for c in cols if c in pair.columns]
        pair = pair[cols].copy()

        out = os.path.join(OUT_DIR, f"{sid}.csv")
        pair.to_csv(out, index=False)
