import os
import glob
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

warnings.filterwarnings("ignore")


IN_ROOT  = r"G:\CYX\GSOD\NCEIgsod"
OUT_ROOT = r"G:\CYX\Analyse"


BASELINE_START = "1991-01-01"
BASELINE_END   = "2020-12-31"
WINDOW_DAYS    = 15
MIN_DUR_PCTL   = 3


TX_LEVELS = [(35.0, 5), (37.0, 4), (40.0, 3), (42.0, 2)]
TN_LEVELS = [(24.0, 3), (26.0, 2), (28.0, 2)]


YEAR_VALID_DAYS = 330
MISSING_BIG = {9999.9, 9999.99, 999.9, 99.99, 9999}
TEMP_COLS = ["TEMP", "MAX", "MIN"]
DATE_CAND = ["DATE","date","Date"]


def ensure_dirs():
    subdirs = [
        "normalized_by_station",
        "thresholds", "events", "annual",
        "events_fixed", "annual_fixed", "figs"
    ]
    for s in subdirs:
        os.makedirs(os.path.join(OUT_ROOT, s), exist_ok=True)


def _is_missing(x):
    try:
        return float(x) in MISSING_BIG
    except Exception:
        return pd.isna(x)

def f_to_c(x):
    if pd.isna(x) or _is_missing(x):
        return np.nan
    return (float(x) - 32.0) * 5.0 / 9.0

def read_one_csv(fp):
    df = pd.read_csv(fp)

    dcol = None
    for c in DATE_CAND:
        if c in df.columns:
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    df["DATE"] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()
    df["year"] = df["DATE"].dt.year
    df["doy"]  = df["DATE"].dt.dayofyear
    df.loc[df["doy"] == 366, "doy"] = 365


    need = ["STATION","LATITUDE","LONGITUDE","ELEVATION","NAME","TEMP","MAX","MIN",
            "TEMP_ATTRIBUTES","MAX_ATTRIBUTES","MIN_ATTRIBUTES"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan


    for c in TEMP_COLS:
        df[c] = df[c].apply(f_to_c)
    df = df.rename(columns={"TEMP":"Tavg", "MAX":"Tmax", "MIN":"Tmin"})


    def has_star(x): return ("*" in str(x)) if not pd.isna(x) else False
    df["flag_max_from_hourly"] = df.get("MAX_ATTRIBUTES", pd.Series([np.nan]*len(df))).apply(has_star).astype(bool)
    df["flag_min_from_hourly"] = df.get("MIN_ATTRIBUTES", pd.Series([np.nan]*len(df))).apply(has_star).astype(bool)

    keep = ["STATION","DATE","year","doy","LATITUDE","LONGITUDE","ELEVATION","NAME",
            "Tmax","Tmin","Tavg","TEMP_ATTRIBUTES","MAX_ATTRIBUTES","MIN_ATTRIBUTES",
            "flag_max_from_hourly","flag_min_from_hourly"]
    out = df[keep].copy().rename(columns={
        "STATION":"station","DATE":"date","LATITUDE":"lat","LONGITUDE":"lon",
        "ELEVATION":"elev","NAME":"name","TEMP_ATTRIBUTES":"n_temp_obs"
    })
    out = out.dropna(subset=["Tmax","Tmin","Tavg"], how="all")
    out = out.sort_values(["station","date"]).reset_index(drop=True)
    return out

def append_df_to_csv(path, df):
    head = (not os.path.exists(path))
    df.to_csv(path, index=False, encoding="utf-8-sig", mode="a", header=head)


def first_pass_normalize_and_dump():
    norm_dir = os.path.join(OUT_ROOT, "normalized_by_station")
    os.makedirs(norm_dir, exist_ok=True)
    stations_meta = {}
    years = sorted([d for d in os.listdir(IN_ROOT) if d.isdigit()])
    if not years:
        raise RuntimeError("输入目录下未找到年份文件夹。")

    print(f"[1/2] 第一遍：规范化并按站点落盘  (年份数={len(years)})")
    for y in tqdm(years, desc="Years"):
        ydir = os.path.join(IN_ROOT, y)
        if not os.path.isdir(ydir):
            continue
        files = glob.glob(os.path.join(ydir, "*.csv"))
        for fp in tqdm(files, desc=f"Files {y}", leave=False):
            try:
                part = read_one_csv(fp)
                if part.empty:
                    continue

                for st, sub in part.groupby("station"):
                    if st not in stations_meta:
                        row = sub.iloc[0]
                        stations_meta[st] = {
                            "station": st,
                            "name": str(row.get("name", "")),
                            "lat": float(row.get("lat", np.nan)),
                            "lon": float(row.get("lon", np.nan)),
                            "elev": float(row.get("elev", np.nan))
                        }

                for st, sub in part.groupby("station"):
                    out_csv = os.path.join(norm_dir, f"{st}.csv")
                    append_df_to_csv(out_csv, sub)
            except Exception as e:
                print(f"[WARN] 读取失败: {fp} -> {e}")


    meta_path = os.path.join(norm_dir, "stations_meta.csv")
    if stations_meta:
        pd.DataFrame(list(stations_meta.values())).to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 规范化完成，按站点落盘于：{norm_dir}")


def build_tx_tn_thresholds(df_station, p=0.90, window=WINDOW_DAYS, base_start=BASELINE_START, base_end=BASELINE_END):
    base = df_station[(df_station["date"]>=base_start) & (df_station["date"]<=base_end)].copy()
    if base.empty:
        base = df_station.copy()
    base = base.dropna(subset=["Tmax","Tmin"])
    q_tx, q_tn = {}, {}
    for doy in range(1, 366):
        lo = max(1, doy - window)
        hi = min(365, doy + window)
        mask = (base["doy"]>=lo) & (base["doy"]<=hi)
        tx = base.loc[mask,"Tmax"].dropna().values
        tn = base.loc[mask,"Tmin"].dropna().values
        def qcalc(arr):
            if len(arr) >= 30:   return np.quantile(arr, p)
            if len(arr) >= 10:   return np.quantile(arr, min(0.85, p))
            return np.nan
        q_tx[doy] = qcalc(tx)
        q_tn[doy] = qcalc(tn)
    s_tx, s_tn = pd.Series(q_tx), pd.Series(q_tn)
    s_tx = s_tx.fillna(method="ffill").fillna(method="bfill")
    s_tn = s_tn.fillna(method="ffill").fillna(method="bfill")
    return s_tx, s_tn

def detect_runs(series_bool):
    runs = []
    in_run = False
    start_i = None
    for i, v in enumerate(series_bool):
        if v and not in_run:
            in_run = True; start_i = i
        elif (not v or i == len(series_bool)-1) and in_run:
            end_i = i-1 if not v else i
            runs.append((start_i, end_i))
            in_run = False
    return runs

def detect_percentile_events(df_station, thr_tx, thr_tn, min_dur=MIN_DUR_PCTL):
    d = df_station.copy()
    d["thr_tx"] = d["doy"].map(thr_tx.to_dict())
    d["thr_tn"] = d["doy"].map(thr_tn.to_dict())
    d["is_hot_day"]   = (d["Tmax"] > d["thr_tx"]).astype(int)
    d["is_hot_night"] = (d["Tmin"] > d["thr_tn"]).astype(int)

    events = []


    for s, e in detect_runs(d["is_hot_day"].tolist()):
        dur = e - s + 1
        if dur >= min_dur:
            seg = d.iloc[s:e+1]
            events.append({
                "station": seg["station"].iloc[0],
                "start_date": seg["date"].iloc[0],
                "end_date": seg["date"].iloc[-1],
                "kind": "DL",
                "duration_days": dur,
                "mean_intensity": (seg["Tmax"] - seg["thr_tx"]).mean(),
                "max_intensity": (seg["Tmax"] - seg["thr_tx"]).max(),
                "cum_intensity": np.maximum(0, (seg["Tmax"] - seg["thr_tx"]).values).sum(),
                "days_over_thr": int(seg["is_hot_day"].sum())
            })


    for s, e in detect_runs(d["is_hot_night"].tolist()):
        dur = e - s + 1
        if dur >= min_dur:
            seg = d.iloc[s:e+1]
            events.append({
                "station": seg["station"].iloc[0],
                "start_date": seg["date"].iloc[0],
                "end_date": seg["date"].iloc[-1],
                "kind": "NL",
                "duration_days": dur,
                "mean_intensity": (seg["Tmin"] - seg["thr_tn"]).mean(),
                "max_intensity": (seg["Tmin"] - seg["thr_tn"]).max(),
                "cum_intensity": np.maximum(0, (seg["Tmin"] - seg["thr_tn"]).values).sum(),
                "days_over_thr": int(seg["is_hot_night"].sum())
            })


    d["is_compound"] = ((d["is_hot_day"]==1) & (d["is_hot_night"]==1)).astype(int)
    for s, e in detect_runs(d["is_compound"].tolist()):
        dur = e - s + 1
        if dur >= 2:
            seg = d.iloc[s:e+1]
            ex = ((seg["Tmax"]-seg["thr_tx"]) + (seg["Tmin"]-seg["thr_tn"])) / 2.0
            events.append({
                "station": seg["station"].iloc[0],
                "start_date": seg["date"].iloc[0],
                "end_date": seg["date"].iloc[-1],
                "kind": "CL",
                "duration_days": dur,
                "mean_intensity": ex.mean(),
                "max_intensity": ex.max(),
                "cum_intensity": np.maximum(0, ex.values).sum(),
                "days_over_thr": int(seg["is_compound"].sum())
            })

    return pd.DataFrame(events)

def detect_fixed_events(df_station, levels, var_name="Tmax"):
    out = []
    for thr, md in levels:
        mask = (df_station[var_name] > thr).astype(int)
        for s, e in detect_runs(mask.tolist()):
            dur = e - s + 1
            if dur >= md:
                seg = df_station.iloc[s:e+1]
                ex = seg[var_name] - thr
                out.append({
                    "station": seg["station"].iloc[0],
                    "start_date": seg["date"].iloc[0],
                    "end_date": seg["date"].iloc[-1],
                    "kind": f"FIX_{var_name}_{int(thr)}",
                    "duration_days": dur,
                    "mean_intensity": ex.mean(),
                    "max_intensity": ex.max(),
                    "cum_intensity": np.maximum(0, ex.values).sum(),
                    "days_over_thr": int((seg[var_name] > thr).sum())
                })
    return pd.DataFrame(out)

def summarize_annual(events_df):
    if events_df.empty:
        return pd.DataFrame(columns=["station","kind","year","events","hw_days","max_duration","mean_duration","mean_intensity","max_intensity","cum_intensity"])
    events_df["year"] = pd.to_datetime(events_df["start_date"]).dt.year
    g = events_df.groupby(["station","kind","year"], as_index=False).agg(
        events=("duration_days", "count"),
        hw_days=("days_over_thr", "sum"),
        max_duration=("duration_days", "max"),
        mean_duration=("duration_days", "mean"),
        mean_intensity=("mean_intensity", "mean"),
        max_intensity=("max_intensity", "max"),
        cum_intensity=("cum_intensity", "sum")
    )
    return g

def plot_series(ann_df, station, kind, y, ylabel, out_png):
    sub = ann_df[(ann_df["station"]==station) & (ann_df["kind"]==kind)].sort_values("year")
    if sub.empty:
        return
    plt.figure(figsize=(12,4))
    plt.plot(sub["year"], sub[y], marker="o", linewidth=1)
    plt.title(f"{station} - {kind} - {ylabel} (1929–2024)")
    plt.xlabel("Year"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
    if len(sub)>=10:
        mv = sub.set_index("year")[y].rolling(10, min_periods=1, center=True).mean()
        plt.plot(mv.index, mv.values, linewidth=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()


def second_pass_by_station():
    norm_dir = os.path.join(OUT_ROOT, "normalized_by_station")
    station_files = sorted([f for f in glob.glob(os.path.join(norm_dir, "*.csv")) if os.path.basename(f) != "stations_meta.csv"])
    if not station_files:
        raise RuntimeError("未找到按站点落盘的 CSV。请先执行第一遍。")

    print(f"[2/2] 第二遍：逐站识别热浪与统计  (站点数={len(station_files)})")
    for fp in tqdm(station_files, desc="Stations"):
        st = os.path.splitext(os.path.basename(fp))[0]
        try:
            sub = pd.read_csv(fp, parse_dates=["date"])
        except Exception as e:
            print(f"[WARN] 读取站点 {st} 失败：{e}")
            continue

        sub = sub.sort_values("date").reset_index(drop=True)
        if sub["date"].nunique() < 2000:

            continue


        thr_tx, thr_tn = build_tx_tn_thresholds(sub)
        thr_df = pd.DataFrame({"doy": np.arange(1,366), "TX90p": thr_tx.values, "TN90p": thr_tn.values})
        thr_out = os.path.join(OUT_ROOT, "thresholds", f"thresholds_{st}.csv")
        thr_df.to_csv(thr_out, index=False, encoding="utf-8-sig")


        events_pctl = detect_percentile_events(sub, thr_tx, thr_tn, min_dur=MIN_DUR_PCTL)
        if not events_pctl.empty:
            events_pctl.sort_values("start_date", inplace=True)
            evp_out = os.path.join(OUT_ROOT, "events", f"events_percentile_{st}.csv")
            events_pctl.to_csv(evp_out, index=False, encoding="utf-8-sig")

            ann_pctl = summarize_annual(events_pctl)
            anp_out = os.path.join(OUT_ROOT, "annual", f"annual_percentile_{st}.csv")
            ann_pctl.to_csv(anp_out, index=False, encoding="utf-8-sig")


            fig_dir = os.path.join(OUT_ROOT, "figs"); os.makedirs(fig_dir, exist_ok=True)
            for kind in ["DL","NL","CL"]:
                plot_series(ann_pctl, st, kind, "events", "Heatwave frequency (events/yr)",
                            os.path.join(fig_dir, f"{st}_{kind}_HWN.png"))
                plot_series(ann_pctl, st, kind, "max_duration", "Longest duration (days)",
                            os.path.join(fig_dir, f"{st}_{kind}_HWD.png"))
                plot_series(ann_pctl, st, kind, "cum_intensity", "Cumulative intensity (°C·days/yr)",
                            os.path.join(fig_dir, f"{st}_{kind}_CWD.png"))


        ev_fix_tx = detect_fixed_events(sub, TX_LEVELS, var_name="Tmax")
        ev_fix_tn = detect_fixed_events(sub, TN_LEVELS, var_name="Tmin")
        events_fix = pd.concat([ev_fix_tx, ev_fix_tn], ignore_index=True) if (not ev_fix_tx.empty or not ev_fix_tn.empty) else pd.DataFrame()
        if not events_fix.empty:
            events_fix.sort_values("start_date", inplace=True)
            evf_out = os.path.join(OUT_ROOT, "events_fixed", f"events_fixed_{st}.csv")
            events_fix.to_csv(evf_out, index=False, encoding="utf-8-sig")

            ann_fix = summarize_annual(events_fix)
            anf_out = os.path.join(OUT_ROOT, "annual_fixed", f"annual_fixed_{st}.csv")
            ann_fix.to_csv(anf_out, index=False, encoding="utf-8-sig")


def main():
    ensure_dirs()

    first_pass_normalize_and_dump()

    second_pass_by_station()
    print("[DONE] 全部输出已写入：", OUT_ROOT)

if __name__ == "__main__":
    main()
