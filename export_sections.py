#!/usr/bin/env python
"""Export small, calibrated section images from Zeiss Axioscan 7 .czi slide scans.

The raw scans are 20x brightfield mosaics (0.172 um/px, 8-bit BGR, JPEG-XR tiles,
one "scene" per detected tissue section, 7-level pyramid).  For lesion-area work
the full resolution is not needed, so this reads one coarse pyramid level per
scene and writes it as an 8-bit RGB TIFF with the pixel size stored in the file.

Usage (env: histology_stroke):
    python export_sections.py "/media/.../stroke_8s" --out out --downsample 16

Outputs in --out (one folder per slide):
    inventory.csv              one row per .czi (scan / label / preview / focusmap / other)
    sections.csv               one row per exported section (pixel size, size in mm, position)
    <slide>/<slide>_sNN.tif    section image, RGB uint8, deflate, um/px in TIFF tags
    <slide>/<slide>_qc.jpg     slide label + slide preview + whole-slide overview with scene numbers
    <slide>/<slide>_label.png  handwritten slide label (file names carry no slide identity)
    <slide>/<slide>_preview.png  low-mag photo of the whole slide
Slides whose status in the --labels manifest starts with "superseded" go under
    _superseded_partial_scans/<slide>/ instead.

Reader notes:
    pylibCZIrw returns BGR;  czifile returns RGB.  Everything written here is RGB.
    Use --downsample 2/4/8/16/32/64: these are the pyramid levels stored in the file.
"""
import argparse
import csv
import os
import re
import sys

import cv2
import czifile
import numpy as np
import tifffile
from pylibCZIrw import czi as pyczi

WHITE = (1.0, 1.0, 1.0)  # fill for pixels outside the scanned region


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def find_czi(sources):
    files = []
    for src in sources:
        if os.path.isdir(src):
            for dirpath, _, names in os.walk(src):
                files += [os.path.join(dirpath, n) for n in names if n.lower().endswith(".czi")]
        else:
            files.append(src)
    return sorted(set(files), key=natural_key)


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def as_list(x):
    return [] if x is None else (x if isinstance(x, list) else [x])


def slide_id(path):
    """'9_2_2026-RecognizedCode-16.czi' -> '20260902_16' (no suffix -> '_0')."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(\d{1,2})_(\d{1,2})_(\d{4})[-_]?(.*)", stem)
    date, rest = (f"{m[3]}{int(m[1]):02d}{int(m[2]):02d}", m[4]) if m else ("nodate", stem)
    rest = re.sub(r"RecognizedCode", "", rest)
    rest = re.sub(r"[^A-Za-z0-9]+", "-", rest).strip("-")
    return f"{date}_{rest or '0'}"


def file_info(path):
    """Header + XML metadata only; no pixel data is read."""
    with pyczi.open_czi(path) as c:
        rect = c.total_bounding_rectangle
        scenes = dict(c.scenes_bounding_rectangle)
        pixel_types = sorted(set(c.pixel_types.values()))
        md = c.metadata["ImageDocument"]["Metadata"]
    img = dig(md, "Information", "Image") or {}
    um = {d.get("@Id"): float(d["Value"]) * 1e6 for d in as_list(dig(md, "Scaling", "Items", "Distance"))}
    um_px = um.get("X", float("nan"))
    if scenes:
        kind = "scan"
    elif rect.w <= 0:
        kind = "focusmap"
    elif um_px > 20:
        kind = "label" if rect.h > rect.w else "preview"
    else:
        kind = "other"
    info = {
        "file": path,
        "kind": kind,
        "slide_id": slide_id(path) if kind == "scan" else "",
        "size_MB": round(os.path.getsize(path) / 1e6, 1),
        "width_px": rect.w,
        "height_px": rect.h,
        "n_scenes": len(scenes),
        "n_tiles": img.get("SizeM", ""),
        "pixel_type": ",".join(pixel_types),
        "um_per_px": round(um_px, 5),
        "objective": dig(md, "Scaling", "AutoScaling", "ObjectiveName") or "",
        "channels": ",".join(str(c.get("@Name")) for c in as_list(dig(img, "Dimensions", "Channels", "Channel"))),
        "compression": img.get("OriginalCompressionMethod", ""),
        "acquired_utc": str(img.get("AcquisitionDateAndTime", ""))[:19],
        "scan_profile": dig(md, "AttachmentInfos", "AttachmentInfo", "Profile", "ProfileName") or "",
        "microscope": dig(md, "Information", "Instrument", "Microscopes", "Microscope", "@Name") or "",
    }
    return info, scenes, rect


def read_attachments(path):
    """Slide label and slide preview stored inside each scan (RGB order, uint8)."""
    out = {}
    with czifile.CziFile(path) as c:
        for a in c.attachments():
            name = a.attachment_entry.name
            if name in ("Label", "SlidePreview"):
                arr = np.squeeze(a.data())
                if arr.dtype == np.uint16:
                    arr = (arr >> 8).astype(np.uint8)
                out[name] = arr
    return out


def count_tissue_blobs(preview_rgb, um_per_px=35.6, min_mm2=5.0):
    """Rough count of tissue pieces on the slide preview (stained tissue is far more saturated than glass)."""
    hsv = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] > 110).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_px = min_mm2 / (um_per_px / 1000) ** 2
    return int(sum(stats[i, cv2.CC_STAT_AREA] >= min_px for i in range(1, n)))


def write_tif(path, rgb, um_per_px):
    px_per_cm = 1e4 / um_per_px
    tifffile.imwrite(path, rgb, photometric="rgb", compression="zlib", predictor=True,
                     resolution=(px_per_cm, px_per_cm), resolutionunit="CENTIMETER",
                     metadata=None, description=f"um_per_px={um_per_px:.5f}")


def qc_sheet(label, preview, overview_bgr, scenes, rect, ov_ds, title):
    """Label + preview on top, annotated whole-slide overview below."""
    ov = overview_bgr.copy()
    for i, r in scenes.items():
        x0, y0 = int((r.x - rect.x) / ov_ds), int((r.y - rect.y) / ov_ds)
        x1, y1 = int((r.x - rect.x + r.w) / ov_ds), int((r.y - rect.y + r.h) / ov_ds)
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.putText(ov, f"s{i:02d}", (x0 + 12, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 10)
        cv2.putText(ov, f"s{i:02d}", (x0 + 12, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    width = 2400
    ov = cv2.resize(ov, (width, max(1, round(ov.shape[0] * width / ov.shape[1]))), interpolation=cv2.INTER_AREA)
    top_h = 600
    parts = []
    for arr in (label, preview):
        if arr is None:
            continue
        bgr = arr[..., ::-1]
        parts.append(cv2.resize(bgr, (round(bgr.shape[1] * top_h / bgr.shape[0]), top_h), interpolation=cv2.INTER_CUBIC))
    top = np.full((top_h, width, 3), 255, np.uint8)
    x = 0
    for p in parts:
        p = p[:, : max(0, width - x)]
        top[:, x:x + p.shape[1]] = p
        x += p.shape[1] + 20
    bar = np.full((60, width, 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return np.vstack([bar, top, ov])


def export_slide(path, info, scenes, rect, out, rel_dir, downsample, overview_ds=64):
    """Write everything belonging to one slide into out/rel_dir/."""
    sid = info["slide_id"]
    slide_dir = os.path.join(out, rel_dir)
    os.makedirs(slide_dir, exist_ok=True)
    um_px = info["um_per_px"] * downsample
    att = read_attachments(path)
    label, preview = att.get("Label"), att.get("SlidePreview")
    if label is not None:
        cv2.imwrite(os.path.join(slide_dir, f"{sid}_label.png"), label[..., ::-1])
    if preview is not None:
        cv2.imwrite(os.path.join(slide_dir, f"{sid}_preview.png"), preview[..., ::-1])
    rows = []
    with pyczi.open_czi(path) as c:
        overview = c.read(roi=(rect.x, rect.y, rect.w, rect.h), plane={"C": 0}, zoom=1 / overview_ds, background_pixel=WHITE)
        for i, r in sorted(scenes.items()):
            bgr = c.read(scene=i, plane={"C": 0}, zoom=1 / downsample, background_pixel=WHITE)
            name = f"{sid}_s{i:02d}.tif"
            write_tif(os.path.join(slide_dir, name), np.ascontiguousarray(bgr[..., ::-1]), um_px)
            rows.append({
                "slide_id": sid, "scene": i, "image": f"{rel_dir}/{name}",
                "um_per_px": round(um_px, 5), "width_px": bgr.shape[1], "height_px": bgr.shape[0],
                "width_mm": round(r.w * info["um_per_px"] / 1000, 3), "height_mm": round(r.h * info["um_per_px"] / 1000, 3),
                "x0_fullres": r.x, "y0_fullres": r.y, "w_fullres": r.w, "h_fullres": r.h,
                "source_czi": path,
            })
    n_blobs = count_tissue_blobs(preview) if preview is not None else ""
    title = f"{sid}   {os.path.basename(path)}   scenes scanned: {len(scenes)}   tissue pieces on preview (auto): {n_blobs}"
    sheet = qc_sheet(label, preview, overview, scenes, rect, overview_ds, title)
    cv2.imwrite(os.path.join(slide_dir, f"{sid}_qc.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rows, n_blobs


def superseded_slides(manifest):
    """slide_ids marked 'superseded...' in the hand-edited slide manifest (if there is one)."""
    if not manifest or not os.path.isfile(manifest):
        return set()
    with open(manifest, newline="") as fh:
        return {r["slide_id"] for r in csv.DictReader(fh) if r.get("status", "").startswith("superseded")}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help=".czi files and/or folders (searched recursively)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--downsample", type=int, default=16, choices=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--inventory-only", action="store_true", help="write inventory.csv and stop")
    ap.add_argument("--labels", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "slide_labels.csv"),
                    help="slide manifest; slides with status 'superseded...' are filed separately")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    superseded = superseded_slides(args.labels)

    inventory, sections = [], []
    for path in find_czi(args.sources):
        try:
            info, scenes, rect = file_info(path)
        except Exception as exc:  # unreadable / truncated file: record it and carry on
            inventory.append({"file": path, "kind": f"ERROR {exc!r}"})
            print(f"ERROR  {path}: {exc!r}", file=sys.stderr)
            continue
        info["tissue_pieces_on_preview"] = ""
        if info["kind"] == "scan" and not args.inventory_only:
            sid = info["slide_id"]
            rel_dir = f"_superseded_partial_scans/{sid}" if sid in superseded else sid
            rows, n_blobs = export_slide(path, info, scenes, rect, args.out, rel_dir, args.downsample)
            sections += rows
            info["tissue_pieces_on_preview"] = n_blobs
            print(f"{rel_dir:54s} {len(rows):2d} sections  (preview shows ~{n_blobs})")
        inventory.append(info)

    keys = list(max(inventory, key=len).keys())
    write_csv(os.path.join(args.out, "inventory.csv"), [{k: r.get(k, "") for k in keys} for r in inventory])
    write_csv(os.path.join(args.out, "sections.csv"), sections)
    kinds = {}
    for r in inventory:
        kinds[r["kind"].split()[0]] = kinds.get(r["kind"].split()[0], 0) + 1
    print(f"\n{len(inventory)} czi files: {kinds};  {len(sections)} sections exported to {args.out}/")


if __name__ == "__main__":
    main()
