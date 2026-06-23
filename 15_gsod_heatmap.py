import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


matrix_dir = r"G:\CYX\Analyse\figs_GSOD"


out_dir = matrix_dir


gsod_cmap = "viridis"


PLOT_YEAR_MIN = 1950
PLOT_YEAR_MAX = 2024

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def load_matrix(metric: str) -> pd.DataFrame:
    fname = f"GSOD_CA_{metric}_matrix.csv"
    fpath = os.path.join(matrix_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"找不到矩阵文件：{fpath}")

    df = pd.read_csv(fpath, index_col=0)

    df.index = df.index.astype(int)
    df.columns = [int(c) for c in df.columns]
    df = df.reindex(columns=sorted(df.columns))

    return df


def pad_to_full_years(df: pd.DataFrame,
                      year_min: int,
                      year_max: int) -> tuple[np.ndarray, np.ndarray]:
    full_years = np.arange(year_min, year_max + 1)
    matrix_full = np.full((len(full_years), 12), np.nan)

    for i, y in enumerate(full_years):
        if y in df.index:

            row = df.loc[y].values

            if len(row) == 12:
                matrix_full[i, :] = row
            else:

                temp = np.full(12, np.nan)
                temp[:min(12, len(row))] = row[:min(12, len(row))]
                matrix_full[i, :] = temp

    return matrix_full, full_years


def plot_from_matrix(metric: str):

    df = load_matrix(metric)


    matrix, years = pad_to_full_years(df, PLOT_YEAR_MIN, PLOT_YEAR_MAX)


    fig, ax = plt.subplots(figsize=(10, 12))

    if np.all(np.isnan(matrix)):
        vmin, vmax = 0, 1
    else:
        vmin = 0
        vmax = np.nanpercentile(matrix, 98)

    im = ax.imshow(
        matrix,
        origin="upper",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="Reds"
    )


    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        fontsize=10
    )


    yticks_idx = np.arange(0, len(years), 5)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(years[yticks_idx], fontsize=10)

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    ax.set_title(
        f"GSOD {metric} over Central Asia "
        f"({PLOT_YEAR_MIN}–{PLOT_YEAR_MAX})\n"
        f"(90th percentile, 3+ consecutive days)",
        fontsize=13
    )


    cb = fig.colorbar(im, ax=ax)
    if metric == "HWF":
        cb.set_label("HWF (days per month)", fontsize=12)
    elif metric == "HWD":
        cb.set_label("HWD (longest event days)", fontsize=12)
    elif metric == "HWI":
        cb.set_label("HWI (℃ excess)", fontsize=12)
    else:
        cb.set_label(metric, fontsize=12)

    plt.tight_layout()


    out_fig = os.path.join(out_dir, f"GSOD_CA_{metric}_heatmap_1950_2024.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"{metric} 热力图已保存：{out_fig}")


def main():
    for metric in ['HWD', 'HWF', 'HWI']:
        print(f"\n===== 使用中间矩阵绘制 {metric} 图 =====")
        plot_from_matrix(metric)


if __name__ == "__main__":
    main()
