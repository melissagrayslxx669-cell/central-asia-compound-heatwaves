import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


PAIR_DIR = r"G:\CYX\Analyse\pair_daily_real"
OUT_DIR  = r"G:\CYX\Analyse\figs"
os.makedirs(OUT_DIR, exist_ok=True)

VARS = ["tmax", "tmin", "tmean"]
MIN_OVERLAP_DAYS = 365
YEAR_MIN, YEAR_MAX = 1950, 2024

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"

GSOD_COLS = [f"{v}_gsod" for v in VARS]
ERA5_COLS = [f"{v}_era5" for v in VARS]


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

def ensure_era5_celsius(df):

    for c in (set(ERA5_COLS) & set(df.columns)):
        s = df[c]
        if s.notna().sum():
            med = float(np.nanmedian(s))
            if med > 150:
                df[c] = s - 273.15
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


rows = []
fps = glob.glob(os.path.join(PAIR_DIR, "*.csv"))
for fp in fps:
    sid = os.path.basename(fp)[:-4]
    try:
        df = pd.read_csv(fp, parse_dates=["date"])
    except Exception:
        continue
    if df.empty:
        continue


    df = df[(df["date"] >= f"{YEAR_MIN}-01-01") & (df["date"] <= f"{YEAR_MAX}-12-31")]


    df = set_sentinels_to_nan(df)
    df = force_gsod_to_celsius(df)
    df = ensure_era5_celsius(df)
    df = build_tmean_if_missing(df)
    df = remove_unrealistic(df)


    for var in VARS:
        gx = f"{var}_gsod"; ey = f"{var}_era5"
        if gx not in df.columns or ey not in df.columns:
            continue
        sub = df.dropna(subset=[gx, ey])
        if sub.shape[0] < MIN_OVERLAP_DAYS:
            continue
        m = compute_metrics(sub[gx].to_numpy(dtype=float), sub[ey].to_numpy(dtype=float))
        rows.append(dict(station=sid, var=var, **m))

met = pd.DataFrame(rows)
if met.empty:
    raise RuntimeError("没有满足条件的站点/变量（检查 MIN_OVERLAP_DAYS 或输入数据）。")


tab_all = os.path.join(OUT_DIR, "tab2_daily_consistency_allvars_C-forced.csv")
met.to_csv(tab_all, index=False)
for v in VARS:
    met[met["var"]==v].to_csv(os.path.join(OUT_DIR, f"tab2_daily_consistency_{v}_C-forced.csv"), index=False)
print(f"[OK] metrics saved -> {tab_all}")


def plot_single_metric(metric_key, y_label, out_stub):
    data = [met[met["var"]==v][metric_key].dropna().values for v in VARS]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=220)
    ax.boxplot(data, positions=np.arange(1, len(VARS)+1), widths=0.6, showfliers=True)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["Tmax","Tmin","Tmean"])
    ax.set_ylabel(y_label)

    if metric_key == "bias":
        lo, hi = ax.get_ylim()
        ax.plot([0.5, 3.5], [0, 0], lw=1)
        ax.set_ylim(lo, hi)

    title_map = dict(r="Correlation (r)", bias="Bias (ERA5 − GSOD)", rmse="RMSE")
    ax.set_title(f"{title_map.get(metric_key, metric_key)} — Variables: Tmax, Tmin, Tmean")
    plt.tight_layout()
    png_fp = os.path.join(OUT_DIR, f"{out_stub}_C-forced.png")
    pdf_fp = os.path.join(OUT_DIR, f"{out_stub}_C-forced.pdf")
    fig.savefig(png_fp, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_fp, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved -> {png_fp} ; {pdf_fp}")

plot_single_metric("r",    "r",        "fig_04_r")
plot_single_metric("bias", "Bias (°C)","fig_04_bias")
plot_single_metric("rmse", "RMSE (°C)","fig_04_rmse")
