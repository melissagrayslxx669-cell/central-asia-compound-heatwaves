import os
import glob
import re
import gzip
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        "normalized", "thresholds", "events", "annual",
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


    for c in ["STATION","LATITUDE","LONGITUDE","ELEVATION","NAME","TEMP","MAX","MIN",
              "TEMP_ATTRIBUTES","MAX_ATTRIBUTES","MIN_ATTRIBUTES"]:
        if c not in df.columns:
            df[c] = np.nan


    for c in TEMP_COLS:
        df[c] = df[c].apply(f_to_c)
    df = df.rename(columns={"TEMP":"Tavg", "MAX":"Tmax", "MIN":"Tmin"})


    def has_star(x):
        return "*" in str(x) if not pd.isna(x) else False
    df["flag_max_from_hourly"] = df.get("MAX_ATTRIBUTES", pd.Series([np.nan]*len(df))).apply(has_star).astype(bool)
    df["flag_min_from_hourly"] = df.get("MIN_ATTRIBUTES", pd.Series([np.nan]*len(df))).apply(has_star).astype(bool)


    keep = ["STATION","DATE","year","doy","LATITUDE","LONGITUDE","ELEVATION","NAME",
            "Tmax","Tmin","Tavg","TEMP_ATTRIBUTES","MAX_ATTRIBUTES","MIN_ATTRIBUTES",
            "flag_max_from_hourly","flag_min_from_hourly"]
    out = df[keep].copy()
    out = out.rename(columns={
        "STATION":"station","DATE":"date","LATITUDE":"lat","LONGITUDE":"lon","ELEVATION":"elev","NAME":"name",
        "TEMP_ATTRIBUTES":"n_temp_obs"
    })

    out = out.dropna(subset=["Tmax","Tmin","Tavg"], how="all")
    out = out.sort_values(["station","date"]).reset_index(drop=True)
    return out

def read_all_years(root_dir):
    years = sorted([d for d in os.listdir(root_dir) if d.isdigit()])
    parts = []
    for y in years:
        ydir = os.path.join(root_dir, y)
        if not os.path.isdir(ydir):
            continue
        files = glob.glob(os.path.join(ydir, "*.csv"))
        for fp in files:
            try:
                part = read_one_csv(fp)
                if not part.empty:
                    parts.append(part)
            except Exception as e:
                print(f"[WARN] 读取失败: {fp} -> {e}")
    if parts:
        big = pd.concat(parts, ignore_index=True)
        return big.sort_values(["station","date"]).reset_index(drop=True)
    return pd.DataFrame()

def year_valid_mask(df):

    g = df.groupby(["station","year"])["date"].count().reset_index(name="ndays")
    g["valid"] = g["ndays"] >= YEAR_VALID_DAYS
    return g


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
        tx, tn = base.loc[mask,"Tmax"].dropna().values, base.loc[mask,"Tmin"].dropna().values

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


    runs_dl = detect_runs(d["is_hot_day"].tolist())
    events_dl = []
    for s, e in runs_dl:
        dur = e - s + 1
        if dur >= min_dur:
            seg = d.iloc[s:e+1]
            events_dl.append({
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


    runs_nl = detect_runs(d["is_hot_night"].tolist())
    events_nl = []
    for s, e in runs_nl:
        dur = e - s + 1
        if dur >= min_dur:
            seg = d.iloc[s:e+1]
            events_nl.append({
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
    runs_cl = detect_runs(d["is_compound"].tolist())
    events_cl = []
    for s, e in runs_cl:
        dur = e - s + 1
        if dur >= 2:
            seg = d.iloc[s:e+1]

            ex = ((seg["Tmax"]-seg["thr_tx"]) + (seg["Tmin"]-seg["thr_tn"])) / 2.0
            events_cl.append({
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

    events_df = pd.DataFrame(events_dl + events_nl + events_cl)
    return d, events_df


def detect_fixed_events(df_station, levels, var_name="Tmax"):
    out = []
    for thr, md in levels:
        mask = (df_station[var_name] > thr).astype(int)
        runs = detect_runs(mask.tolist())
        for s, e in runs:
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


def main():
    ensure_dirs()
    print("[INFO] 读取全部年度数据...")
    df = read_all_years(IN_ROOT)
    if df.empty:
        raise RuntimeError("未读取到任何 GSOD CSV。请检查输入目录。")
    print("[INFO] 原始记录数:", len(df))


    val = year_valid_mask(df)
    df = df.merge(val[["station","year","valid"]], on=["station","year"], how="left")
    print("[INFO] 站-年有效天≥%d 的比例：%.1f%%" % (YEAR_VALID_DAYS, 100*val["valid"].mean()))


    norm_path = os.path.join(OUT_ROOT, "normalized", "daily_normalized.parquet")
    df.to_parquet(norm_path, index=False)
    print(f"[OK] 已保存规范化日值 -> {norm_path}")

    stations = df["station"].dropna().unique().tolist()
    print(f"[INFO] 站点数量：{len(stations)}")

    all_events_pctl = []
    all_ann_pctl = []
    all_events_fix  = []
    all_ann_fix  = []

    for si, st in enumerate(stations, 1):
        sub = df[df["station"]==st].copy().sort_values("date").reset_index(drop=True)
        if sub["date"].nunique() < 2000:
            print(f"[WARN] 站 {st} 有效日数较少，跳过百分位口径。")
            continue


        thr_tx, thr_tn = build_tx_tn_thresholds(sub)
        thr_df = pd.DataFrame({"doy": np.arange(1,366), "TX90p": thr_tx.values, "TN90p": thr_tn.values})
        thr_out = os.path.join(OUT_ROOT, "thresholds", f"thresholds_{st}.csv")
        thr_df.to_csv(thr_out, index=False, encoding="utf-8-sig")


        daily_with_thr, events_pctl = detect_percentile_events(sub, thr_tx, thr_tn, min_dur=MIN_DUR_PCTL)
        if not events_pctl.empty:
            events_pctl.sort_values("start_date", inplace=True)
            evp_out = os.path.join(OUT_ROOT, "events", f"events_percentile_{st}.csv")
            events_pctl.to_csv(evp_out, index=False, encoding="utf-8-sig")
            all_events_pctl.append(events_pctl)

            ann_pctl = summarize_annual(events_pctl)
            anp_out = os.path.join(OUT_ROOT, "annual", f"annual_percentile_{st}.csv")
            ann_pctl.to_csv(anp_out, index=False, encoding="utf-8-sig")
            all_ann_pctl.append(ann_pctl)


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
        events_fix = pd.concat([ev_fix_tx, ev_fix_tn], ignore_index=True) if not ev_fix_tx.empty or not ev_fix_tn.empty else pd.DataFrame()
        if not events_fix.empty:
            events_fix.sort_values("start_date", inplace=True)
            evf_out = os.path.join(OUT_ROOT, "events_fixed", f"events_fixed_{st}.csv")
            events_fix.to_csv(evf_out, index=False, encoding="utf-8-sig")
            all_events_fix.append(events_fix)

            ann_fix = summarize_annual(events_fix)
            anf_out = os.path.join(OUT_ROOT, "annual_fixed", f"annual_fixed_{st}.csv")
            ann_fix.to_csv(anf_out, index=False, encoding="utf-8-sig")
            all_ann_fix.append(ann_fix)

        print(f"[OK] ({si}/{len(stations)}) 完成站点 {st}")


    if all_events_pctl:
        pd.concat(all_events_pctl, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "events", "events_percentile_ALL.csv"), index=False, encoding="utf-8-sig")
    if all_ann_pctl:
        pd.concat(all_ann_pctl, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "annual", "annual_percentile_ALL.csv"), index=False, encoding="utf-8-sig")
    if all_events_fix:
        pd.concat(all_events_fix, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "events_fixed", "events_fixed_ALL.csv"), index=False, encoding="utf-8-sig")
    if all_ann_fix:
        pd.concat(all_ann_fix, ignore_index=True).to_csv(os.path.join(OUT_ROOT, "annual_fixed", "annual_fixed_ALL.csv"), index=False, encoding="utf-8-sig")

    print("[DONE] 全部输出已写入：", OUT_ROOT)

if __name__ == "__main__":
    main()
