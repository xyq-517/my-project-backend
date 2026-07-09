import argparse
import csv
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import pydicom
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pydicom\n"
        "Please install it first, for example:\n"
        "  pip install pydicom Pillow numpy"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw LIDC-IDRI CT + XML annotations into VOC-style 2D slices."
    )
    parser.add_argument(
        "--lidc-root",
        default=r"D:\unet-pytorch\LIDC-IDRI\manifest-1600709154662\LIDC-IDRI",
        help="Root directory that contains case folders such as LIDC-IDRI-0001.",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\unet-pytorch\VOCdevkit\VOC2007",
        help="VOC output root. JPEGImages/ and SegmentationClass/ will be created here.",
    )
    parser.add_argument(
        "--min-readers",
        type=int,
        default=1,
        help="Minimum number of reader votes per pixel. 1 means union of all XML contours.",
    )
    parser.add_argument(
        "--include-empty-slices",
        action="store_true",
        help="Export slices without nodules. By default only slices with positive masks are exported.",
    )
    parser.add_argument(
        "--window-center",
        type=float,
        default=-600.0,
        help="CT window center used to export JPEG slices.",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=1500.0,
        help="CT window width used to export JPEG slices.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional limit for debugging. 0 means export all cases.",
    )
    parser.add_argument(
        "--z-tolerance",
        type=float,
        default=1.5,
        help="Maximum allowed distance between XML roi z-position and CT slice z-position in mm.",
    )
    return parser.parse_args()


def strip_tag(tag):
    return tag.split("}", 1)[-1]


def iter_children_by_name(node, name):
    for child in node:
        if strip_tag(child.tag) == name:
            yield child


def find_first_child(node, name):
    for child in iter_children_by_name(node, name):
        return child
    return None


def get_text(node, name, default=None):
    child = find_first_child(node, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def ensure_dirs(output_root):
    jpeg_dir = output_root / "JPEGImages"
    mask_dir = output_root / "SegmentationClass"
    set_dir = output_root / "ImageSets" / "Segmentation"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    set_dir.mkdir(parents=True, exist_ok=True)
    return jpeg_dir, mask_dir, set_dir


def discover_case_dirs(lidc_root):
    return sorted([p for p in lidc_root.iterdir() if p.is_dir() and p.name.startswith("LIDC-IDRI-")])


def collect_series_dirs(case_dir):
    series_dirs = []
    for study_dir in case_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if series_dir.is_dir():
                series_dirs.append(series_dir)
    return series_dirs


def is_ct_series(series_dir):
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if len(dcm_files) < 10:
        return False
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return False
    return getattr(ds, "Modality", "") == "CT"


def collect_xml_files(case_dir):
    return sorted(case_dir.glob("*/*/*.xml"))


def choose_ct_series(case_dir):
    ct_candidates = [p for p in collect_series_dirs(case_dir) if is_ct_series(p)]
    if not ct_candidates:
        return None
    ct_candidates.sort(key=lambda p: len(list(p.glob("*.dcm"))), reverse=True)
    return ct_candidates[0]


def load_ct_volume(ct_series_dir):
    slices = []
    for dcm_path in sorted(ct_series_dir.glob("*.dcm")):
        ds = pydicom.dcmread(str(dcm_path), force=True)
        if getattr(ds, "Modality", "") != "CT":
            continue
        if not hasattr(ds, "PixelData"):
            continue
        image_position = getattr(ds, "ImagePositionPatient", None)
        if image_position is not None and len(image_position) >= 3:
            z = float(image_position[2])
        elif hasattr(ds, "SliceLocation"):
            z = float(ds.SliceLocation)
        else:
            z = float(getattr(ds, "InstanceNumber", len(slices)))
        instance_number = int(getattr(ds, "InstanceNumber", len(slices)))
        slices.append(
            {
                "path": dcm_path,
                "dataset": ds,
                "z": z,
                "instance_number": instance_number,
            }
        )

    slices.sort(key=lambda x: (x["z"], x["instance_number"]))
    if not slices:
        raise RuntimeError(f"No readable CT slices found in {ct_series_dir}")

    first = slices[0]["dataset"]
    height = int(first.Rows)
    width = int(first.Columns)

    z_positions = np.array([s["z"] for s in slices], dtype=np.float32)
    return slices, z_positions, height, width


def convert_to_hu(ds):
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels * slope + intercept


def window_to_uint8(hu_image, center, width):
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(hu_image, lower, upper)
    scaled = (clipped - lower) / max(upper - lower, 1e-6)
    return (scaled * 255.0).astype(np.uint8)


def image_to_rgb_jpeg_array(hu_image, center, width):
    gray = window_to_uint8(hu_image, center, width)
    return np.stack([gray, gray, gray], axis=-1)


def parse_lidc_xml(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    rois = []
    for session in root.iter():
        if strip_tag(session.tag) != "readingSession":
            continue
        for nodule in iter_children_by_name(session, "unblindedReadNodule"):
            nodule_id = get_text(nodule, "noduleID", default="unknown")
            for roi in iter_children_by_name(nodule, "roi"):
                inclusion_text = get_text(roi, "inclusion", default="TRUE")
                inclusion = inclusion_text.upper() != "FALSE"
                z_text = get_text(roi, "imageZposition")
                if z_text is None:
                    continue
                points = []
                for edge_map in iter_children_by_name(roi, "edgeMap"):
                    x_text = get_text(edge_map, "xCoord")
                    y_text = get_text(edge_map, "yCoord")
                    if x_text is None or y_text is None:
                        continue
                    points.append((int(float(x_text)), int(float(y_text))))
                if len(points) >= 3 and inclusion:
                    rois.append(
                        {
                            "xml_path": xml_path,
                            "nodule_id": nodule_id,
                            "z": float(z_text),
                            "points": points,
                        }
                    )
    return rois


def match_roi_to_slice_index(roi_z, slice_z_positions, tolerance):
    distances = np.abs(slice_z_positions - roi_z)
    best_index = int(np.argmin(distances))
    if float(distances[best_index]) > tolerance:
        return None
    return best_index


def polygon_to_mask(height, width, points):
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    drawer.polygon(points, outline=1, fill=1)
    return np.array(canvas, dtype=np.uint8)


def build_vote_masks(xml_files, slice_z_positions, height, width, z_tolerance):
    per_slice_votes = defaultdict(list)
    skipped_rois = 0

    for xml_path in xml_files:
        try:
            rois = parse_lidc_xml(xml_path)
        except Exception as exc:
            logging.warning("Skip unreadable XML %s: %s", xml_path, exc)
            continue

        for roi in rois:
            slice_index = match_roi_to_slice_index(roi["z"], slice_z_positions, z_tolerance)
            if slice_index is None:
                skipped_rois += 1
                continue
            per_slice_votes[slice_index].append(polygon_to_mask(height, width, roi["points"]))

    return per_slice_votes, skipped_rois


def export_case(
    case_dir,
    jpeg_dir,
    mask_dir,
    records,
    min_readers,
    include_empty_slices,
    window_center,
    window_width,
    z_tolerance,
):
    ct_series_dir = choose_ct_series(case_dir)
    if ct_series_dir is None:
        logging.warning("Case %s has no CT series, skipped.", case_dir.name)
        return 0

    xml_files = collect_xml_files(case_dir)
    if not xml_files:
        logging.warning("Case %s has no XML annotations, skipped.", case_dir.name)
        return 0

    slices, z_positions, height, width = load_ct_volume(ct_series_dir)
    vote_masks, skipped_rois = build_vote_masks(xml_files, z_positions, height, width, z_tolerance)

    exported = 0
    for slice_index, slice_info in enumerate(slices):
        vote_stack = vote_masks.get(slice_index, [])
        if vote_stack:
            votes = np.sum(np.stack(vote_stack, axis=0), axis=0)
            mask = (votes >= min_readers).astype(np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        if not include_empty_slices and mask.max() == 0:
            continue

        hu_image = convert_to_hu(slice_info["dataset"])
        image_rgb = image_to_rgb_jpeg_array(hu_image, window_center, window_width)

        sample_id = f"{case_dir.name}_{slice_index:04d}"
        Image.fromarray(image_rgb, mode="RGB").save(jpeg_dir / f"{sample_id}.jpg", quality=95)
        Image.fromarray(mask, mode="L").save(mask_dir / f"{sample_id}.png")

        records.append(
            {
                "sample_id": sample_id,
                "case_id": case_dir.name,
                "slice_index": slice_index,
                "instance_number": slice_info["instance_number"],
                "z_position": slice_info["z"],
                "mask_pixels": int(mask.sum()),
                "annotation_xml_count": len(xml_files),
                "skipped_rois_due_to_z_mismatch": skipped_rois,
                "ct_series_dir": str(ct_series_dir),
            }
        )
        exported += 1

    logging.info(
        "Case %s exported %d slices from %s using %d XML files.",
        case_dir.name,
        exported,
        ct_series_dir.name,
        len(xml_files),
    )
    return exported


def write_split_files(sample_ids, set_dir, train_ratio=0.8, val_ratio=0.2):
    rng = np.random.default_rng(42)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = len(shuffled) - train_count
    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count:]

    (set_dir / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""), encoding="utf-8")
    (set_dir / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""), encoding="utf-8")
    (set_dir / "trainval.txt").write_text("\n".join(shuffled) + ("\n" if shuffled else ""), encoding="utf-8")
    (set_dir / "test.txt").write_text("", encoding="utf-8")

    logging.info("Split files written: train=%d, val=%d, test=0", train_count, val_count)


def write_records_csv(records, output_root):
    csv_path = output_root / "lidc_export_records.csv"
    if not records:
        return

    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logging.info("Export metadata written to %s", csv_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    lidc_root = Path(args.lidc_root)
    output_root = Path(args.output_root)
    jpeg_dir, mask_dir, set_dir = ensure_dirs(output_root)

    if not lidc_root.exists():
        raise FileNotFoundError(f"LIDC root not found: {lidc_root}")

    case_dirs = discover_case_dirs(lidc_root)
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]

    logging.info("Found %d case directories under %s", len(case_dirs), lidc_root)
    logging.info("Output JPEG directory: %s", jpeg_dir)
    logging.info("Output mask directory: %s", mask_dir)
    logging.info("Minimum reader votes per pixel: %d", args.min_readers)
    logging.info("Include empty slices: %s", args.include_empty_slices)

    records = []
    total_exported = 0
    for case_dir in case_dirs:
        total_exported += export_case(
            case_dir=case_dir,
            jpeg_dir=jpeg_dir,
            mask_dir=mask_dir,
            records=records,
            min_readers=args.min_readers,
            include_empty_slices=args.include_empty_slices,
            window_center=args.window_center,
            window_width=args.window_width,
            z_tolerance=args.z_tolerance,
        )

    write_records_csv(records, output_root)
    write_split_files([r["sample_id"] for r in records], set_dir)

    logging.info("Done. Exported %d samples into VOC format.", total_exported)
    logging.info("Next steps:")
    logging.info("1. Run voc_annotation.py only if you want to regenerate split files from masks.")
    logging.info("2. In train.py set num_classes = 2.")
    logging.info("3. In get_miou.py / predict.py set class names to ['background', 'nodule'].")


if __name__ == "__main__":
    main()
