import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


daily_data_dir = r"G:\CYX\ERA5\daily"
out_matrix = r"G:\CYX\Analyse\ERA5_CA_HWI_matrix.csv"
out_fig = r"G:\CYX\Analyse\figs_ERA5\ERA5_CA_HWI_heatmap.png"

threshold_period = (1991, 2020)
percentile_threshold = 90
min_consecutive_days = 3
temperature_var = 't2m_daily_max'


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def load_daily_temperature(data_dir: str, temp_var: str) -> pd.DataFrame:
    all_files = []
    for year in range(1950, 2025):
        for month in range(1, 13):
            file_path = os.path.join(data_dir, f"era5_daily_t2m_{year}{month:02d}_utc+5.nc")
            if os.path.exists(file_path):
                all_files.append(file_path)

    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何日数据文件")

    print(f"找到 {len(all_files)} 个月度文件，开始读取...")

    data_list = []
    for file_path in sorted(all_files):
        try:
            ds = xr.open_dataset(file_path)
            if temp_var not in ds.data_vars:
                ds.close()
                continue

            time_coord = next((t for t in ['time', 'valid_time', 'datetime'] if t in ds.coords), None)
            if time_coord is None:
                ds.close()
                continue

            times = ds[time_coord].values
            temps = ds[temp_var].values

            if temps.ndim > 1:
                temps = temps.mean(axis=tuple(range(1, temps.ndim)))

            df = pd.DataFrame({'time': pd.to_datetime(times), 't2m': temps})
            data_list.append(df)
            print(f"  已加载: {os.path.basename(file_path)} ({len(df)} 天)")
            ds.close()

        except Exception as e:
            print(f"读取 {file_path} 时出错: {e}，跳过")
            continue

    daily_df = pd.concat(data_list, ignore_index=True)
    daily_df = daily_df.sort_values('time').reset_index(drop=True)
    print(f"总计加载 {len(daily_df)} 条日记录")
    return daily_df


def calculate_HWI(daily_df: pd.DataFrame, threshold_period: tuple,
                  percentile: int, min_days: int) -> pd.Series:
    daily_df = daily_df.set_index('time').sort_index().copy()


    baseline = daily_df[(daily_df.index.year >= threshold_period[0]) &
                        (daily_df.index.year <= threshold_period[1])].copy()
    baseline.loc[:, 'month_day'] = baseline.index.strftime('%m-%d')
    thresholds = baseline.groupby('month_day')['t2m'].quantile(percentile/100)

    daily_df.loc[:, 'month_day'] = daily_df.index.strftime('%m-%d')
    daily_df.loc[:, 'threshold'] = daily_df['month_day'].map(thresholds)
    daily_df.loc[:, 'excess'] = daily_df['t2m'] - daily_df['threshold']
    daily_df.loc[:, 'exceed'] = daily_df['excess'] > 0


    daily_df.loc[:, 'heatwave_event'] = 0
    group = daily_df['exceed'].ne(daily_df['exceed'].shift()).cumsum()
    event_id = 1

    for group_id, group_df in daily_df.groupby(group):
        if group_df['exceed'].iloc[0] and len(group_df) >= min_days:
            daily_df.loc[group_df.index, 'heatwave_event'] = event_id
            event_id += 1


    hw_days = daily_df[daily_df['heatwave_event'] > 0].copy()
    monthly_hwi = hw_days.resample('ME')['excess'].mean()

    full_index = pd.date_range(start=daily_df.index.min(),
                               end=daily_df.index.max(),
                               freq='ME')
    monthly_hwi_complete = pd.Series(np.nan, index=full_index, dtype=float)
    monthly_hwi_complete.update(monthly_hwi)
    monthly_hwi_complete = monthly_hwi_complete.fillna(0)

    return monthly_hwi_complete


def main():
    print("步骤1: 读取日数据...")
    daily_df = load_daily_temperature(daily_data_dir, temperature_var)

    print("\n步骤2: 计算HWI...")
    monthly_hwi = calculate_HWI(daily_df, threshold_period,
                                percentile_threshold,
                                min_consecutive_days)

    print("\n步骤3: 构建矩阵...")
    matrix_data = np.full((75, 12), np.nan)
    year_index = {year: i for i, year in enumerate(range(1950, 2025))}

    for date, intensity in monthly_hwi.items():
        year = date.year
        month = date.month
        if year in year_index:
            matrix_data[year_index[year], month-1] = intensity

    os.makedirs(os.path.dirname(out_matrix), exist_ok=True)
    matrix_df = pd.DataFrame(matrix_data,
                             index=range(1950, 2025),
                             columns=range(1, 13))
    matrix_df.to_csv(out_matrix)
    print(f"HWI矩阵已保存至: {out_matrix}")

    print("\n步骤4: 绘制HWI热力图...")
    fig, ax = plt.subplots(figsize=(10, 12))

    vmin = 0
    vmax = np.nanpercentile(matrix_data, 98) if not np.isnan(matrix_data).all() else 2


    im = ax.imshow(
        matrix_data,
        origin="upper",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="YlOrRd"
    )

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        fontsize=10
    )

    yticks = np.arange(0, 75, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(range(1950, 2025, 5), fontsize=10)

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    ax.set_title(
        f"ERA5 Heatwave Intensity (HWI) over Central Asia (1950–2024)\n"
        f"({temperature_var}, {percentile_threshold}th percentile, {min_consecutive_days}+ consecutive days)",
        fontsize=14
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("HWI (℃ excess)", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig, dpi=300)
    print(f"HWI热力图已保存至: {out_fig}")
    plt.show()

if __name__ == "__main__":
    main()
