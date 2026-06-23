import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


STATION_ID = "276790"
VAR        = "tmax"
IN_FP      = rf"G:\CYX\Analyse\pair_daily_real\{STATION_ID}.csv"
OUT_DIR    = r"G:\CYX\Analyse\figs"
os.makedirs(OUT_DIR, exist_ok=True)

PNG_FP = os.path.join(OUT_DIR, f"fig_03_scatter_{STATION_ID}_{VAR}_C-forced.png")
PDF_FP = os.path.join(OUT_DIR, f"fig_03_scatter_{STATION_ID}_{VAR}_C-forced.pdf")
CSV_FP = os.path.join(OUT_DIR, f"fig_03_metrics_{STATION_ID}_{VAR}_C-forced.csv")

GSOD_COLS = [f"{v}_gsod" for v in ("tmax","tmin","tmean")]
ERA5_COLS = [f"{v}_era5" for v in ("tmax","tmin","tmean")]

def set_sentinels_to_nan(df):

    for c in (set(GSOD_COLS + ERA5_COLS) & set(df.columns)):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] >= 900,  c] = np.nan
        df.loc[df[c] <= -900, c] = np.nan
    return df

def force_gsod_to_celsius(df):

    for c in (set(GSOD_COLS) & set(df.columns)):
        df[c] = (df[c] - 32.0) * (5.0/9.0)
    return df

def kelvin_to_c_if_needed(s):
    if s.notna().sum() == 0:
        return s
    med = float(np.nanmedian(s))
    return (s - 273.15) if med > 150 else s

def ensure_era5_celsius(df):

    for c in (set(ERA5_COLS) & set(df.columns)):
        df[c] = kelvin_to_c_if_needed(df[c])
    return df

def build_tmean_if_missing(df):
    if ("tmean_gsod" not in df.columns) and {"tmax_gsod","tmin_gsod"}.issubset(df.columns):
        df["tmean_gsod"] = (df["tmax_gsod"] + df["tmin_gsod"]) / 2.0
    if ("tmean_era5" not in df.columns) and {"tmax_era5","tmin_era5"}.issubset(df.columns):
        df["tmean_era5"] = (df["tmax_era5"] + df["tmin_era5"]) / 2.0
    return df

def remove_unrealistic(df):

    for c in (set(GSOD_COLS + ERA5_COLS) & set(df.columns)):
        df.loc[df[c] > 60,  c] = np.nan
        df.loc[df[c] < -80, c] = np.nan
    return df

def compute_metrics(x, y):
    d = y - x
    bias = float(np.nanmean(d))
    rmse = float(np.sqrt(np.nanmean(d**2)))
    mae  = float(np.nanmean(np.abs(d)))
    r    = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else np.nan
    slope, intercept, *_ = linregress(x, y) if len(x) >= 2 else (np.nan, np.nan, np.nan, np.nan, np.nan)
    return dict(n=len(x), r=r, bias=bias, rmse=rmse, mae=mae, slope=slope, intercept=intercept)


df = pd.read_csv(IN_FP, parse_dates=["date"])
df = set_sentinels_to_nan(df)
df = force_gsod_to_celsius(df)
df = ensure_era5_celsius(df)
df = build_tmean_if_missing(df)
df = remove_unrealistic(df)

gsod_col = f"{VAR}_gsod"
era5_col = f"{VAR}_era5"
if gsod_col not in df.columns or era5_col not in df.columns:
    raise KeyError(f"缺少必要列：{gsod_col} 或 {era5_col}")

dd = df.dropna(subset=[gsod_col, era5_col]).copy()
x = dd[gsod_col].to_numpy(dtype=float)
y = dd[era5_col].to_numpy(dtype=float)
if x.size < 10:
    raise RuntimeError(f"{STATION_ID}: 有效重叠日不足（n={x.size}），无法绘制散点。")

metrics = compute_metrics(x, y)


plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
ax.scatter(x, y, s=8, alpha=0.35)

vmin = float(np.nanmin([x.min(), y.min()]))
vmax = float(np.nanmax([x.max(), y.max()]))
pad  = 0.05 * (vmax - vmin) if np.isfinite(vmax - vmin) else 1.0
lo, hi = vmin - pad, vmax + pad


ax.plot([lo, hi], [lo, hi], lw=1.0)


if np.isfinite(metrics["slope"]) and np.isfinite(metrics["intercept"]):
    xx = np.array([lo, hi])
    yy = metrics["slope"] * xx + metrics["intercept"]
    ax.plot(xx, yy, lw=1.0)

ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")

label_map = dict(tmax="Tmax", tmin="Tmin", tmean="Tmean")
ax.set_xlabel(f"GSOD {label_map.get(VAR, VAR).title()} (°C)")
ax.set_ylabel(f"ERA5 {label_map.get(VAR, VAR).title()} (°C)")
ax.set_title(f"Daily Consistency — Station {STATION_ID} ({label_map.get(VAR, VAR).title()})")

txt = (f"n={metrics['n']}\n"
       f"r={metrics['r']:.3f}\n"
       f"Bias={metrics['bias']:.2f}°C\n"
       f"RMSE={metrics['rmse']:.2f}°C\n"
       f"MAE={metrics['mae']:.2f}°C")
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left")

plt.tight_layout()
fig.savefig(PNG_FP, dpi=300)
fig.savefig(PDF_FP)
plt.close(fig)


pd.DataFrame([{
    "station": STATION_ID,
    "var": VAR,
    **metrics
}]).to_csv(CSV_FP, index=False)

print(f"[OK] Fig3 saved:\n  {PNG_FP}\n  {PDF_FP}\n[OK] Metrics saved:\n  {CSV_FP}")
