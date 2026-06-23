from __future__ import annotations

import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


WORLD_SHP = r"G:\CYX\world shp\world shp\World_Country.shp"
CA_DIR = r"G:\Central Asia"
OUT_DIR = r"G:\CYX\world shp\Figs"


OUT_PNG = os.path.join(OUT_DIR, "Fig1_locator_Eurasia_CentralAsia.png")
OUT_TIF = os.path.join(OUT_DIR, "Fig1_locator_Eurasia_CentralAsia.tif")
OUT_WORLD_GPKG = os.path.join(OUT_DIR, "locator_eurasia_base_clip.gpkg")
OUT_CA_GPKG = os.path.join(OUT_DIR, "locator_central_asia_outline.gpkg")


EURASIA_XLIM = (-15, 155)
EURASIA_YLIM = (5, 80)


LAND_FACE = "#d9d9d9"
LAND_EDGE = "#9e9e9e"
CA_EDGE = "#d7301f"
CA_FACE = "none"
SEA_FACE = "white"


FIGSIZE = (8.6, 4.8)
DPI = 600
TITLE = ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_central_asia_shp(ca_dir: str) -> str:
    shp_files = sorted(glob.glob(os.path.join(ca_dir, "*.shp")))
    if not shp_files:
        raise FileNotFoundError(f"未在目录中找到 shp 文件：{ca_dir}")

    preferred_keywords = ["central", "asia", "admin", "boundary", "region", "ca"]

    ranked = []
    for shp in shp_files:
        name = os.path.basename(shp).lower()
        score = sum(k in name for k in preferred_keywords)
        ranked.append((score, shp))
    ranked.sort(key=lambda x: (-x[0], x[1]))

    for _, shp in ranked:
        try:
            gdf = gpd.read_file(shp)
            if not gdf.empty and gdf.geometry.notna().any():
                geom_types = set(gdf.geometry.geom_type.dropna().unique())
                if geom_types & {"Polygon", "MultiPolygon"}:
                    return shp
        except Exception:
            continue

    raise RuntimeError(f"{ca_dir} 中存在 shp，但未找到可用的面状研究区 shp。")


def read_and_prepare_world(world_shp: str) -> gpd.GeoDataFrame:
    if not os.path.exists(world_shp):
        raise FileNotFoundError(f"世界 shp 不存在：{world_shp}")

    world = gpd.read_file(world_shp)
    if world.empty:
        raise RuntimeError("世界 shp 读取成功，但为空。")
    if world.crs is None:
        raise RuntimeError("世界 shp 缺少坐标系信息（.prj），无法稳妥叠加。")

    world = world.to_crs("EPSG:4326")


    clip_geom = box(EURASIA_XLIM[0], EURASIA_YLIM[0], EURASIA_XLIM[1], EURASIA_YLIM[1])
    clip_box = gpd.GeoDataFrame(geometry=[clip_geom], crs="EPSG:4326")
    world_clip = gpd.clip(world, clip_box)

    if world_clip.empty:
        raise RuntimeError("世界 shp 在亚欧范围裁剪后为空，请检查世界边界数据。")


    world_clip = world_clip[world_clip.geometry.notna() & (~world_clip.geometry.is_empty)].copy()
    if world_clip.empty:
        raise RuntimeError("世界 shp 裁剪后仅剩空几何。")

    return world_clip


def read_and_prepare_ca(ca_shp: str) -> gpd.GeoDataFrame:
    ca = gpd.read_file(ca_shp)
    if ca.empty:
        raise RuntimeError("中亚研究区 shp 读取成功，但为空。")
    if ca.crs is None:
        raise RuntimeError("中亚 shp 缺少坐标系信息（.prj），无法稳妥叠加。")

    ca = ca.to_crs("EPSG:4326")
    ca = ca[ca.geometry.notna() & (~ca.geometry.is_empty)].copy()
    if ca.empty:
        raise RuntimeError("中亚研究区 shp 中没有有效几何。")

    ca_outline = gpd.GeoDataFrame(geometry=[ca.unary_union], crs="EPSG:4326")
    return ca_outline


def export_layers(world_clip: gpd.GeoDataFrame, ca_outline: gpd.GeoDataFrame) -> None:
    if os.path.exists(OUT_WORLD_GPKG):
        os.remove(OUT_WORLD_GPKG)
    if os.path.exists(OUT_CA_GPKG):
        os.remove(OUT_CA_GPKG)

    world_clip.to_file(OUT_WORLD_GPKG, driver="GPKG")
    ca_outline.to_file(OUT_CA_GPKG, driver="GPKG")


def draw_locator_map(world_clip: gpd.GeoDataFrame, ca_outline: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(SEA_FACE)
    ax.set_facecolor(SEA_FACE)

    world_clip.plot(
        ax=ax,
        facecolor=LAND_FACE,
        edgecolor=LAND_EDGE,
        linewidth=0.35,
        zorder=1,
    )

    ca_outline.boundary.plot(
        ax=ax,
        color=CA_EDGE,
        linewidth=1.8,
        zorder=3,
    )

    ax.set_xlim(*EURASIA_XLIM)
    ax.set_ylim(*EURASIA_YLIM)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    if TITLE:
        ax.set_title(TITLE, fontsize=12)

    plt.tight_layout(pad=0.2)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_TIF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ensure_dir(OUT_DIR)

    print("[1/5] 检查世界边界 shp ...")
    world_clip = read_and_prepare_world(WORLD_SHP)

    print("[2/5] 自动寻找中亚研究区 shp ...")
    ca_shp = find_central_asia_shp(CA_DIR)
    print(f"       使用中亚 shp: {ca_shp}")

    print("[3/5] 读取并处理研究区边界 ...")
    ca_outline = read_and_prepare_ca(ca_shp)

    print("[4/5] 导出 ArcGIS 可微调中间图层 ...")
    export_layers(world_clip, ca_outline)

    print("[5/5] 绘制并导出定位图 ...")
    draw_locator_map(world_clip, ca_outline)

    print("完成。输出目录：")
    print(f"  {OUT_DIR}")
    print("输出文件：")
    print(f"  - {OUT_PNG}")
    print(f"  - {OUT_TIF}")
    print(f"  - {OUT_WORLD_GPKG}")
    print(f"  - {OUT_CA_GPKG}")


if __name__ == "__main__":
    main()
