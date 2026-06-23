import os
import glob
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
from scipy.ndimage import label


ERA5_DIR = r"G:\CYX\ERA5\daily"
GSOD_DIR = r"G:\CYX\Analyse\normalized_by_station"
CACHE_FILE = "heatwave_metrics_cache.pkl"


BASE_START, BASE_END = 1991, 2020


era5_results = {'HWD': {'CL': [], 'DL': [], 'NL': []},
                'HWF': {'CL': [], 'DL': [], 'NL': []},
                'HWI': {'CL': [], 'DL': [], 'NL': []}}

gsod_results = {'HWD': {'CL': [], 'DL': [], 'NL': []},
                'HWF': {'CL': [], 'DL': [], 'NL': []},
                'HWI': {'CL': [], 'DL': [], 'NL': []}}


def get_valid_heatwave_days_and_events(mask_array):
    labeled_array, num_features = label(mask_array)
    valid_mask = np.zeros_like(mask_array, dtype=bool)
    max_duration = 0

    if num_features > 0:
        unique, counts = np.unique(labeled_array, return_counts=True)
        counts_dict = dict(zip(unique[1:], counts[1:]))

        for val, length in counts_dict.items():
            if length >= 3:
                valid_mask[labeled_array == val] = True
                if length > max_duration:
                    max_duration = length

    return valid_mask, max_duration


def process_gsod_data():
    print(">>> [1/3] 开始处理 GSOD 站点数据...")
    csv_files = glob.glob(os.path.join(GSOD_DIR, "*.csv"))
    if not csv_files:
        print(f"未在 {GSOD_DIR} 找到 CSV 文件！")
        return

    bad_files_count = 0

    for file in tqdm(csv_files, desc="Processing GSOD Stations"):
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            if len(df.columns) <= 1 or 'date' not in [str(c).strip().lower() for c in df.columns]:
                df = pd.read_csv(file, sep=r'\s+', on_bad_lines='skip')
        except Exception:
            bad_files_count += 1
            continue

        df.columns = df.columns.str.strip().str.lower()

        if not all(col in df.columns for col in ['date', 'year', 'doy', 'tmax', 'tmin']):
            bad_files_count += 1
            continue

        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            bad_files_count += 1
            continue

        base_df = df[(df['year'] >= BASE_START) & (df['year'] <= BASE_END)]
        if base_df.empty:
            continue

        tx90 = base_df.groupby('doy')['tmax'].quantile(0.9).rename('TX90p')
        tn90 = base_df.groupby('doy')['tmin'].quantile(0.9).rename('TN90p')

        df['TX90p'] = df['doy'].map(tx90)
        df['TN90p'] = df['doy'].map(tn90)

        df['mask_CL'] = (df['tmax'] > df['TX90p']) & (df['tmin'] > df['TN90p'])
        df['mask_DL'] = (df['tmax'] > df['TX90p']) & (df['tmin'] <= df['TN90p'])
        df['mask_NL'] = (df['tmin'] > df['TN90p']) & (df['tmax'] <= df['TX90p'])

        df['int_CL'] = ((df['tmax'] - df['TX90p']) + (df['tmin'] - df['TN90p'])) / 2.0
        df['int_DL'] = df['tmax'] - df['TX90p']
        df['int_NL'] = df['tmin'] - df['TN90p']

        for year, group in df.groupby('year'):
            for hw_type in ['CL', 'DL', 'NL']:
                mask = group[f'mask_{hw_type}'].values
                intensity = group[f'int_{hw_type}'].values

                valid_mask, hwd = get_valid_heatwave_days_and_events(mask)
                hwf = valid_mask.sum()

                if hwf > 0:
                    hwi = np.nanmean(intensity[valid_mask])
                    gsod_results['HWD'][hw_type].append(hwd)
                    gsod_results['HWF'][hw_type].append(hwf)
                    gsod_results['HWI'][hw_type].append(hwi)

    if bad_files_count > 0:
        print(f"\n注意：共跳过了 {bad_files_count} 个格式损坏或数据不完整的异常 CSV 文件。")


def process_era5_data():
    print("\n>>> [2/3] 开始处理 ERA5 格点数据...")
    nc_files = glob.glob(os.path.join(ERA5_DIR, "*.nc"))
    if not nc_files:
        print(f"未在 {ERA5_DIR} 找到 NC 文件！")
        return

    ds = xr.open_mfdataset(nc_files, combine='by_coords')

    tmax = ds['t2m_daily_max']
    tmin = ds['t2m_daily_min']

    base_tmax = tmax.sel(time=slice(str(BASE_START), str(BASE_END)))
    base_tmin = tmin.sel(time=slice(str(BASE_START), str(BASE_END)))

    print("   正在计算 ERA5 基准期 90% 阈值 (内存消耗较大，请耐心等待)...")
    tx90p = base_tmax.groupby('time.dayofyear').quantile(0.9, dim='time').compute()
    tn90p = base_tmin.groupby('time.dayofyear').quantile(0.9, dim='time').compute()

    print("   正在逐网格提取热浪事件与指标...")
    lats = ds.latitude.values
    lons = ds.longitude.values

    times = ds.time.values
    years = pd.DatetimeIndex(times).year
    doys = pd.DatetimeIndex(times).dayofyear

    tx90_arr = tx90p.sel(dayofyear=doys).values
    tn90_arr = tn90p.sel(dayofyear=doys).values

    tmax_values = tmax.values
    tmin_values = tmin.values

    total_grids = len(lats) * len(lons)
    pbar = tqdm(total=total_grids, desc="Processing ERA5 Grids")

    for i in range(len(lats)):
        for j in range(len(lons)):
            tmax_1d = tmax_values[:, i, j]
            tmin_1d = tmin_values[:, i, j]

            if np.isnan(tmax_1d[0]):
                pbar.update(1)
                continue

            tx90_1d = tx90_arr[:, i, j]
            tn90_1d = tn90_arr[:, i, j]

            mask_CL = (tmax_1d > tx90_1d) & (tmin_1d > tn90_1d)
            mask_DL = (tmax_1d > tx90_1d) & (tmin_1d <= tn90_1d)
            mask_NL = (tmin_1d > tn90_1d) & (tmax_1d <= tx90_1d)

            int_CL = ((tmax_1d - tx90_1d) + (tmin_1d - tn90_1d)) / 2.0
            int_DL = tmax_1d - tx90_1d
            int_NL = tmin_1d - tn90_1d

            df_temp = pd.DataFrame({
                'YEAR': years,
                'mask_CL': mask_CL, 'mask_DL': mask_DL, 'mask_NL': mask_NL,
                'int_CL': int_CL, 'int_DL': int_DL, 'int_NL': int_NL
            })

            for year, group in df_temp.groupby('YEAR'):
                for hw_type in ['CL', 'DL', 'NL']:
                    mask = group[f'mask_{hw_type}'].values
                    intensity = group[f'int_{hw_type}'].values

                    valid_mask, hwd = get_valid_heatwave_days_and_events(mask)
                    hwf = valid_mask.sum()

                    if hwf > 0:
                        hwi = np.nanmean(intensity[valid_mask])
                        era5_results['HWD'][hw_type].append(hwd)
                        era5_results['HWF'][hw_type].append(hwf)
                        era5_results['HWI'][hw_type].append(hwi)

            pbar.update(1)
    pbar.close()


def plot_results(era5_data, gsod_data):
    print("\n>>> 开始绘制核密度估计 (KDE) 概率分布图...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), dpi=300)

    warm_colors = {'CL': '#d73027', 'DL': '#f46d43', 'NL': '#fdae61'}
    cool_colors = {'CL': '#313695', 'DL': '#4575b4', 'NL': '#74add1'}

    metrics = ['HWD', 'HWF', 'HWI']
    x_labels = ['HWD (days)', 'HWF (days)', 'HWI (°C)']
    row_sources = [era5_data, gsod_data]
    row_names = ['ERA5', 'GSOD']
    color_schemes = [warm_colors, cool_colors]

    pbar_plot = tqdm(total=18, desc="Drawing KDE Lines", unit="line")

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            metric = metrics[j]
            data_dict = row_sources[i][metric]
            colors = color_schemes[i]

            for hw_type in ['CL', 'DL', 'NL']:
                plot_data = np.array(data_dict[hw_type])
                plot_data = plot_data[~np.isnan(plot_data)]

                if len(plot_data) > 1:

                    bw = 2.0 if metric in ['HWD', 'HWF'] else 1.0

                    sns.kdeplot(
                        data=plot_data,
                        ax=ax,
                        color=colors[hw_type],
                        label=hw_type,
                        linewidth=2.5,
                        fill=False,
                        alpha=0.9,
                        bw_adjust=bw
                    )
                pbar_plot.update(1)


            if i == 0 and j == 0:
                ax.set_xlim(left=0, right=20)
            elif i == 0 and j == 1:
                ax.set_xlim(left=0, right=30)
            elif i == 1 and j == 0:
                ax.set_xlim(left=0, right=20)
            elif i == 1 and j == 1:
                ax.set_xlim(left=0, right=60)
            elif i == 1 and j == 2:
                ax.set_xlim(left=0, right=15)


            ax.grid(True, linestyle='--', alpha=0.5, color='gray')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            if i == 1:
                ax.set_xlabel(x_labels[j], fontsize=14)
            else:
                ax.set_xlabel('')

            if j == 0:
                ax.set_ylabel(f'Probability Density\n({row_names[i]})', fontsize=14)
            else:
                ax.set_ylabel('')

            if i == 0:
                ax.set_title(f'{metric}', fontsize=16, fontweight='bold', pad=15)

            ax.legend(loc='upper right', frameon=False, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)

    pbar_plot.close()
    plt.tight_layout()
    plt.savefig('Heatwave_KDE_Comparison_6panels.png', bbox_inches='tight', dpi=300)
    print("\n>>> 绘图全部完成！大图已保存至当前运行目录：Heatwave_KDE_Comparison_6panels.png")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        print(f"检测到缓存数据文件：{CACHE_FILE}。")
        print(">>> 正在直接加载数据，跳过漫长的计算步骤...")
        with open(CACHE_FILE, 'rb') as f:
            cached_data = pickle.load(f)
            era5_results = cached_data['era5']
            gsod_results = cached_data['gsod']
        print(">>> 缓存数据加载成功！直接进入绘图环节。")
    else:
        print("未检测到缓存数据，开始完整的数据提取计算流程...")
        process_gsod_data()
        process_era5_data()

        print(f"\n>>> 正在将提取的数据结果保存至 {CACHE_FILE} ...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'era5': era5_results, 'gsod': gsod_results}, f)
        print(">>> 数据缓存保存成功！")

    plot_results(era5_results, gsod_results)
