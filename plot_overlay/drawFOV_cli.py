# -*- coding: utf-8 -*-
"""
Overlay ROIs onto a Virtual H&E image with labels and a legend.

Example:
python drawFOV_cli.py \
  --base "/people/m344313/Map_FOV_Back_On_vHnE" \
  --outdir "/people/m344313/Map_FOV_Back_On_vHnE/overlays/moran_I" \
  --labels "./labels.csv" \
  --targets Mel43_BMS Mel44_BMS \
  --moves-file "MOVES.xml" \
  --vhe-pattern "S001_VHE_region_001.tif" \
  --static-size 1210 1140 \
  --downscale 0.10 \
  --font-path "./arial.ttf" \
  --font-size 320 \
  --outline-width 40

Switch to ROI-native width/height:
  --use-xml-size

Continuous mode example:
  --cont-col moran_I --vmin -1 --vmax 1 --alpha-fill 200 
"""

import os, math, glob, argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageFont, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors


COLOR_BORDER: Dict[str, Tuple[int, int, int, int]] = {
    'Interface': (0, 171, 0, 164),
    'infiltrate': (90, 255, 106, 164),
    'Infiltrate': (90, 255, 106, 164),
    'Tumor': (122, 0, 0, 164),
    'Normal': (15, 255, 238, 164),
    'Lymphoid Tissue': (15, 255, 238, 164),
    'unknown': (240, 240, 240, 164),
    'Other': (240, 240, 240, 164),
    'unknown2': (42, 246, 33, 164),
    'SCS': (0, 122, 10, 164),
    'Sinus': (243, 166, 0, 164),
    'Cortex': (122, 0, 0, 164),
    'Follicle': (15, 255, 238, 164),
    'Medulla': (194, 2, 153, 164),
    'Margin': (183, 81, 0, 164),
    'Regressions': (176, 160, 160, 164),
    'Tumor Bed': (146, 89, 207, 164),
    'Regression': (176, 160, 160, 164),
    'Reactive Stroma': (15, 255, 238, 164),
    'Fibrotic': (170, 170, 170, 164),
    'Necrotic': (170, 170, 170, 164),
    'Tumor Bed Interface': (188, 0, 191, 164),
    'Tumor Interface': (227, 69, 5, 164),
    'interface': (0, 171, 0, 164),
    'tumor': (122, 0, 0, 164),
    'stroma': (15, 255, 238, 164),
}
COLOR_FILL: Dict[str, Tuple[int, int, int, int]] = {
    'Interface': (0, 84, 0, 100),
    'infiltrate': (163, 247, 170, 100),
    'Infiltrate': (163, 247, 170, 100),
    'Tumor': (210, 14, 14, 100),
    'Normal': (77, 255, 242, 100),
    'Lymphoid Tissue': (77, 255, 242, 100),
    'unknown': (150, 155, 151, 124),
    'Other': (150, 155, 151, 124),
    'unknown2': (200, 254, 196, 124),
    'SCS': (34, 204, 45, 100),
    'Sinus': (228, 170, 0, 100),
    'Cortex': (210, 14, 14, 100),
    'Follicle': (77, 255, 242, 100),
    'Medulla': (245, 117, 218, 100),
    'Margin': (241, 117, 19, 100),
    'Regressions': (77, 70, 70, 100),
    'Tumor Bed': (102, 0, 211, 100),
    'Regression': (77, 70, 70, 100),
    'Reactive Stroma': (77, 255, 242, 100),
    'Fibrotic': (88, 88, 88, 164),
    'Necrotic': (88, 88, 88, 164),
    'Tumor Bed Interface': (240, 85, 242, 164),
    'Tumor Interface': (223, 110, 63, 100),
    'interface': (0, 84, 0, 100),
    'tumor': (210, 14, 14, 100),
    'stroma': (77, 255, 242, 100),
}


def parse_xml(xmlfile: str):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    xform_node = root.find('./canvas_xform')
    if xform_node is None:
        raise ValueError(f"No <canvas_xform> in {xmlfile}")
    xf = {
        'a': float(xform_node.attrib['a']),
        'b': float(xform_node.attrib['b']),
        'c': float(xform_node.attrib['c']),
        'd': float(xform_node.attrib['d']),
        'du': float(xform_node.attrib['du']),
        'dv': float(xform_node.attrib['dv']),
    }
    views = []
    for roi in root.findall('./roi'):
        entry = {}
        for child in roi:
            if child.tag == 'label':
                entry['name'] = child.text
            elif child.tag == 'roi_rect':
                entry['x'] = float(child.attrib['x'])
                entry['y'] = float(child.attrib['y'])
                entry['width'] = float(child.attrib['width'])
                entry['height'] = float(child.attrib['height'])
            elif child.tag == 'moves' and len(child) > 0:
                entry['xp'] = int(child[0].attrib['xp'])
                entry['yp'] = int(child[0].attrib['yp'])
        if {'name','x','y','width','height','xp','yp'} <= set(entry.keys()):
            views.append(entry)
    return views, xf


def _det(xf: dict) -> float:
    return (xf['a'] * xf['d']) - (xf['b'] * xf['c'])


def stage_to_pixel(items: List[dict], xf: dict) -> List[dict]:
    out = []
    det = _det(xf)
    for u in items:
        xNew = xf['du'] + (u['x'] * det + xf['c'] * (u['yp'] - xf['dv'])) / xf['d']
        yNew = xf['dv'] + (u['y'] * det + xf['b'] * (u['xp'] - xf['du'])) / xf['a']
        out.append({'name': u['name'], 'x': xNew, 'y': yNew, 'width': u['width'], 'height': u['height']})
    return out


def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=font_size)
    try:
        return ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        return ImageFont.load_default()


def _get_cmap(cmap_name: str):
    try:
        return plt.get_cmap(cmap_name)
    except Exception:
        return plt.get_cmap("viridis")


def _rgba_from_val(val: float, norm: "mcolors.Normalize", cmap) -> Tuple[int,int,int,int]:
    r, g, b, a = cmap(norm(val))
    return int(r*255), int(g*255), int(b*255), int(a*255)


def _draw_continuous_legend(
    img: Image.Image,
    drawer: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    vmin: float,
    vmax: float,
    cmap,
    x: int = 100,
    y: int = 100,
    width: int = 1600,
    height: int = 200,
    n_ticks: int = 3
):
    grad = Image.new("RGBA", (width, height), (0,0,0,0))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    pix = grad.load()
    for i in range(width):
        val = vmin + (vmax - vmin) * (i / max(1, width-1))
        r, g, b, a = cmap(norm(val))
        rgba = (int(r*255), int(g*255), int(b*255), 255)
        for j in range(height):
            pix[i, j] = rgba
    img.paste(grad, (x, y), grad)
    drawer.rectangle(((x, y), (x+width, y+height)), outline=(0,0,0,200), width=6)
    for k in range(n_ticks):
        frac = k / (n_ticks - 1) if n_ticks > 1 else 0
        xi = int(x + frac * width)
        drawer.line([(xi, y+height), (xi, y+height+40)], fill=(0,0,0,200), width=6)
        val = vmin + frac*(vmax - vmin)
        lbl = f"{val:.3g}"
        tw, th = drawer.textbbox((0,0), lbl, font=font)[2:]
        drawer.text((xi - tw//2, y+height+50), lbl, fill=(0,0,0,255), font=font)


def draw_overlays(
    rois_px: List[dict],
    image_path: str,
    outdir: str,
    sample: str,
    df_labels: pd.DataFrame,
    label_col: str = "Pathology",
    skip_labels: Optional[List[str]] = None,
    use_xml_size: bool = False,
    static_w: int = 1210,
    static_h: int = 1140,
    outline_width: int = 40,
    font_path: Optional[str] = None,
    font_size: int = 320,
    downscale: float = 0.10,
    legend: bool = True,
    cont_col: Optional[str] = None,
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha_fill: int = 100,
    alpha_border: int = 164,
):
    os.makedirs(outdir, exist_ok=True)
    img = Image.open(image_path)
    font = _load_font(font_path, font_size)
    drawer = ImageDraw.Draw(img, 'RGBA')

    if 'Sample' in df_labels.columns:
        df_labels = df_labels.copy()
        df_labels['Sample'] = df_labels['Sample'].str.replace(f"{sample}_", "", regex=False)

    used_labels: List[str] = []
    using_continuous = cont_col is not None and cont_col in df_labels.columns

    if using_continuous:
        col_vals = pd.to_numeric(df_labels[cont_col], errors="coerce")
        auto_vmin = float(col_vals.min(skipna=True)) if col_vals.notna().any() else 0.0
        auto_vmax = float(col_vals.max(skipna=True)) if col_vals.notna().any() else 1.0
        if auto_vmax == auto_vmin:
            auto_vmax = auto_vmin + 1e-9
        vmin_eff = auto_vmin if vmin is None else float(vmin)
        vmax_eff = auto_vmax if vmax is None else float(vmax)
        if vmax_eff == vmin_eff:
            vmax_eff = vmin_eff + 1e-9
        cmap = _get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff, clip=True)

    for ele in rois_px:
        name_full = ele['name']
        label_val = 'unknown'
        cont_val = None

        try:
            short = name_full.split(sample + "_", 1)[-1] if (sample + "_") in name_full else name_full
            if 'Sample' in df_labels.columns:
                rowmask = (df_labels['Sample'] == short)
                if using_continuous:
                    if rowmask.any():
                        tmp = pd.to_numeric(df_labels.loc[rowmask, cont_col], errors="coerce")
                        if len(tmp) > 0:
                            cont_val = tmp.values[0]
                else:
                    if label_col in df_labels.columns and rowmask.any():
                        m = df_labels.loc[rowmask, label_col]
                        if len(m) > 0:
                            label_val = m.values[0]
        except Exception:
            pass

        if using_continuous:
            if cont_val is None or (isinstance(cont_val, float) and math.isnan(cont_val)):
                continue
            r, g, b, _ = _rgba_from_val(cont_val, norm, cmap)
            fill = (r, g, b, alpha_fill)
            border = (r, g, b, alpha_border)
        else:
            if (isinstance(label_val, float) and math.isnan(label_val)) or label_val is None:
                label_val = 'unknown'
            if skip_labels and label_val in skip_labels:
                continue
            if label_val == 'unknown':
                continue
            border = COLOR_BORDER.get(label_val, COLOR_BORDER['unknown2'])
            fill = COLOR_FILL.get(label_val, COLOR_FILL['unknown2'])
            used_labels.append(label_val)

        x, y = ele['x'], ele['y']
        if use_xml_size:
            w, h = ele['width'], ele['height']
        else:
            w, h = static_w, static_h

        drawer.rectangle(((x, y), (x + w, y + h)), width=outline_width, outline=border, fill=fill)
        label_text = name_full.replace(sample + "_", "").replace("region_", " ")
        drawer.text((x - 10, y + 50), label_text, fill=(0, 0, 0), font=font)

    if legend:
        if using_continuous:
            _draw_continuous_legend(
                img=img,
                drawer=drawer,
                font=font,
                vmin=vmin_eff,
                vmax=vmax_eff,
                cmap=cmap,
                x=100,
                y=100,
                width=2000,
                height=220,
                n_ticks=6
            )
            title = f"{cont_col}"
            tb = drawer.textbbox((0,0), title, font=font)
            drawer.text((100, 100 - (tb[3]-tb[1]) - 40), title, fill=(0,0,0,255), font=font)
        else:
            if used_labels:
                xRect, yRect = 100, 100
                boxW, boxH = 2500, 600
                for kk in sorted(set(used_labels)):
                    if kk == 'unknown' or (skip_labels and kk in skip_labels):
                        continue
                    bcol = COLOR_BORDER.get(kk, COLOR_BORDER['unknown2'])
                    drawer.rectangle(((xRect, yRect), (xRect + boxW, yRect + boxH)), fill=bcol)
                    drawer.text((xRect + 40, yRect + 10), kk, fill=(0, 0, 0), font=font)
                    yRect += 700

    if downscale and 0 < downscale < 1:
        new_w = int(img.size[0] * downscale)
        new_h = int(img.size[1] * downscale)
        try:
            from PIL import Image as _PILImage
            resample = getattr(_PILImage, "Resampling", _PILImage).LANCZOS
        except Exception:
            resample = Image.LANCZOS
        out_img = img.resize((new_w, new_h), resample)
    else:
        out_img = img

    out_path = os.path.join(outdir, f"{sample}_vHE.jpg")
    out_img.save(out_path, "JPEG", quality=90)
    print(f"[✓] Saved: {out_path}")


def read_labels_table(path: str, sep: Optional[str]) -> pd.DataFrame:
    if sep is None:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=sep)


def main():
    p = argparse.ArgumentParser(description="Overlay ROI rectangles onto a Virtual H&E image.")
    p.add_argument("--base", required=True, help="Base directory containing per-sample folders.")
    p.add_argument("--outdir", required=True, help="Output directory for JPEG overlays.")
    p.add_argument("--labels", required=True, help="Path to TSV/CSV with columns: Slide, Sample, Pathology (default).")
    p.add_argument("--sep", default=None, help="Field separator for labels file (default: auto-infer).")
    p.add_argument("--label-col", default="Pathology", help="Column from labels to color by (categorical).")

    tg = p.add_mutually_exclusive_group(required=True)
    tg.add_argument("--targets", nargs="+", help="List of sample folder names under --base.")
    tg.add_argument("--targets-file", help="Text file with one sample per line.")

    p.add_argument("--moves-file", default="moves.xml", help="Relative path/name of the moves XML within each sample folder.")
    p.add_argument("--vhe-pattern", default="VirtualStains/S001_VHE_region_001.tif",
                   help="Relative path/pattern to the VHE TIFF (glob allowed) within each sample folder.")

    p.add_argument("--use-xml-size", action="store_true", help="Use ROI width/height from XML instead of static size.")
    p.add_argument("--static-size", nargs=2, type=int, metavar=("W", "H"), default=[1210, 1140],
                   help="Static rectangle size (ignored if --use-xml-size). Default: 1210 1140")

    p.add_argument("--outline-width", type=int, default=40, help="Outline stroke width (default: 40).")
    p.add_argument("--font-path", default=None, help="Path to a .ttf font (default: try arial.ttf, else bitmap default).")
    p.add_argument("--font-size", type=int, default=320, help="Font size (default: 320).")
    p.add_argument("--downscale", type=float, default=0.10, help="Final downscale factor 0..1 (default: 0.10).")
    p.add_argument("--no-legend", action="store_true", help="Do not draw legend.")
    p.add_argument("--skip-label", action="append", default=["unknown", "DROP BAD FOV"],
                   help="Labels to skip. Can be used multiple times. Default: unknown, DROP BAD FOV")

    # Continuous options
    p.add_argument("--cont-col", default=None,
                   help="Column in labels file to treat as CONTINUOUS; overrides categorical coloring when provided.")
    p.add_argument("--cmap", default="viridis",
                   help="Matplotlib colormap name for continuous mode (default: viridis).")
    p.add_argument("--vmin", type=float, default=None,
                   help="Optional min value for continuous normalization (default: auto from data per slide).")
    p.add_argument("--vmax", type=float, default=None,
                   help="Optional max value for continuous normalization (default: auto from data per slide).")
    p.add_argument("--alpha-fill", type=int, default=100,
                   help="Alpha (0-255) for ROI fill in continuous mode (default: 100).")
    p.add_argument("--alpha-border", type=int, default=164,
                   help="Alpha (0-255) for ROI border in continuous mode (default: 164).")

    args = p.parse_args()

    if args.targets:
        targets = args.targets
    else:
        with open(args.targets_file, "r", encoding="utf-8") as fh:
            targets = [ln.strip() for ln in fh if ln.strip()]

    df = read_labels_table(args.labels, args.sep)
    required_cols = {"Slide", "Sample", args.label_col}
    missing = required_cols - set(df.columns)
    if missing and (args.cont_col is None or args.cont_col not in df.columns):
        raise SystemExit(f"Labels file missing columns: {missing}. Present: {list(df.columns)}")

    for sample in targets:
        sample_dir = os.path.join(args.base, sample)
        moves_xml = os.path.join(sample_dir, args.moves_file)
        if not os.path.exists(moves_xml):
            print(f"[!] Missing moves.xml for {sample}: {moves_xml}")
            continue
        vhe_glob = glob.glob(os.path.join(sample_dir, args.vhe_pattern))
        if not vhe_glob:
            print(f"[!] No VHE image matched for {sample}: {os.path.join(sample_dir, args.vhe_pattern)}")
            continue
        vhe_img = vhe_glob[0]

        rois, xf = parse_xml(moves_xml)
        rois_px = stage_to_pixel(rois, xf)
        df_slide = df[df['Slide'] == sample].copy()

        draw_overlays(
            rois_px=rois_px,
            image_path=vhe_img,
            outdir=args.outdir,
            sample=sample,
            df_labels=df_slide,
            label_col=args.label_col,
            skip_labels=args.skip_label,
            use_xml_size=args.use_xml_size,
            static_w=args.static_size[0],
            static_h=args.static_size[1],
            outline_width=args.outline_width,
            font_path=args.font_path,
            font_size=args.font_size,
            downscale=args.downscale,
            legend=not args.no_legend,
            cont_col=args.cont_col,
            cmap_name=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            alpha_fill=args.alpha_fill,
            alpha_border=args.alpha_border
        )


if __name__ == "__main__":
    main()
