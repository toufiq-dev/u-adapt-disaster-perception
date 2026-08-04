#!/usr/bin/env python
"""download_datasets.py — Milestone 1: download and organize LADD + D-Fire.

Fetches the two primary detection benchmarks into the repo's raw-data layout
(which is gitignored — raw data is NEVER committed):

    data/raw/ladd/{train,val,test}/             images only
    data/raw/dfire/{train,val,test}/            images only
    data/annotations/{ladd,dfire}_{split}.json  COCO-style GT (gitignored)
    data/download_scripts/{ladd,dfire}/sha256sums.txt   archive checksums

Usage:

    # verify all configured URLs are reachable (no download)
    python download_datasets.py --check-only

    # D-Fire pilot: 10 images per split (train/test) + COCO conversion
    python download_datasets.py --dataset dfire --subset 10

    # full D-Fire download + conversion
    python download_datasets.py --dataset dfire

    # LADD (manual step — official repo is offline, see README): the script
    # prints step-by-step instructions. Once you have a working URL, either
    # pass it or set LADD_URL, then re-run:
    #   python download_datasets.py --dataset ladd --ladd-url <URL> --subset 10

Notes
-----
* **--subset N** downloads the full official archive (OneDrive shares cannot
  be ranged/partial-downloaded) but copies only the FIRST N sorted images per
  split into ``data/raw`` and writes subset-filtered COCO annotations. This
  keeps the pilot small while the script stays correct for full runs.
* **D-Fire annotations are YOLO format** (normalized class xc yc w h; classes
  0=fire, 1=smoke) — this script converts them to COCO-style JSON so the
  evaluation pipeline (``demo_mode_a_end_to_end.py`` / ``04_evaluate.py``)
  can consume them directly.
* **LADD** — the official GitHub repository (``huyhieupham/LADD``) returned
  404 on 2026-08-04 and no verified live URL could be found. Per the
  Milestone-1 policy (see ``data/download_scripts/README.md``) the script
  does NOT guess a download URL: it prints manual instructions and uses a
  clearly-marked placeholder that you replace once a link is confirmed.
* **License status** is tracked in ``docs/licenses.md`` (verified 2026-08-04).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Verified sources (2026-08-04) — from the official D-Fire README:
#   https://github.com/gaia-solutions-on-demand/DFireDataset
# ---------------------------------------------------------------------------
DFIRE_DRIVE_URL = (
    "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw"
)
DFIRE_KAGGLE_URL = "https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo"
DFIRE_PAPER_URL = "https://link.springer.com/article/10.1007/s00521-022-07467-z"

# Placeholder — replace with a verified LADD download URL when available
# (official repo huyhieupham/LADD is offline as of 2026-08-04).
LADD_URL_PLACEHOLDER = "https://REPLACE_WITH_VERIFIED_LADD_DOWNLOAD_URL"

DFIRE_CLASSES = {0: "fire", 1: "smoke"}  # per the D-Fire paper

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = ROOT / "data" / "raw"
DEFAULT_ANN_ROOT = ROOT / "data" / "annotations"
DEFAULT_STAGING = ROOT / "data" / "raw" / ".staging"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tar.gz", ".tgz"}


# ---------------------------------------------------------------------------
# Small stdlib-only helpers
# ---------------------------------------------------------------------------
def _png_size(data: bytes) -> Optional[Tuple[int, int]]:
    """PNG dimensions from the IHDR chunk (big-endian uint32 w, h)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
        return None
    return struct.unpack(">II", data[16:24])


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    """JPEG dimensions by scanning SOF markers (stdlib-only)."""
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return (w, h)
        i += 2 + seg_len
    return None


def _image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) reading only the image header (stdlib-only)."""
    with open(path, "rb") as fh:
        head = fh.read(64 * 1024)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        wh = _png_size(head)
        if wh:
            return wh
    if head[:2] == b"\xff\xd8":
        wh = _jpeg_size(head)
        if wh:
            return wh
    raise ValueError(
        f"could not read image dimensions from {path} (PNG/JPEG header parse "
        "failed); install Pillow/cv2 for other formats"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffix in (".tar", ".tgz") or archive.name.endswith(".tar.gz"):
        import tarfile

        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"unsupported archive type: {archive}")


def _walk_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _find_yolo_label(image: Path) -> Optional[Path]:
    """Locate the YOLO .txt label for an image.

    Candidates, in order:
      1. a sibling ``<stem>.txt`` next to the image;
      2. a ``labels/`` directory mirroring the ``images/`` tree
         (labels/a/b/img.txt for images/a/b/img.jpg);
      3. a sibling ``labels/<stem>.txt`` directory.
    """
    cand = image.with_suffix(".txt")
    if cand.exists():
        return cand
    # labels/ mirror of the images/ tree
    parts = image.parts
    for i, part in enumerate(parts[:-1]):
        if part.lower() in ("images", "img", "image"):
            mirrored = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")
            if mirrored.exists():
                return mirrored
            break
    # sibling labels/<stem>.txt
    alt = image.parent / "labels" / (image.stem + ".txt")
    return alt if alt.exists() else None


def _yolo_to_coco(images: Sequence[Path], split_name: str) -> Dict:
    """Convert D-Fire YOLO labels to a COCO-style annotation dict.

    Each image is emitted with its width/height (read from headers). Missing
    label files are allowed (D-Fire has ~9.8k 'none' images) and yield an
    empty annotation list. Returns the COCO dict.
    """
    categories = [
        {"id": cls_id + 1, "name": name} for cls_id, name in sorted(DFIRE_CLASSES.items())
    ]
    coco_images, coco_anns = [], []
    ann_id = 1
    for img_id, img_path in enumerate(images):
        w, h = _image_size(img_path)
        coco_images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "height": h,
                "width": w,
            }
        )
        label = _find_yolo_label(img_path)
        if label is None:
            continue
        for line in label.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"malformed YOLO line in {label}: {line!r} "
                    "(expected 'class xc yc w h')"
                )
            cls_id, xc, yc, bw, bh = (float(parts[0]), *map(float, parts[1:]))
            if int(cls_id) not in DFIRE_CLASSES:
                raise ValueError(
                    f"{label}: unknown class id {cls_id} "
                    f"(known: {sorted(DFIRE_CLASSES)})"
                )
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            box_w, box_h = bw * w, bh * h
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                }
            )
            ann_id += 1
    return {
        "info": {
            "description": f"D-Fire YOLO->COCO conversion (download_datasets.py)",
            "split": split_name,
            "source": DFIRE_PAPER_URL,
        },
        "images": coco_images,
        "annotations": coco_anns,
        "categories": categories,
    }


def _pick_subset(sorted_images: List[Path], n: Optional[int]) -> List[Path]:
    if n is None:
        return sorted_images
    return sorted_images[:n]


# ---------------------------------------------------------------------------
# D-Fire
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download ``url`` to ``dest``; returns False if the server answered HTML."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype or "application/json" in ctype:
                return False
            shutil.copyfileobj(resp, out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  download error: {exc}", file=sys.stderr)
        return False
    tmp.replace(dest)
    return True


def run_dfire(args: argparse.Namespace) -> int:
    raw_ds = args.raw_root / "dfire"
    if _dataset_present(raw_ds):
        print("  D-Fire images already present — skipping download.")
        if not (args.annotations_root / "dfire_test.json").exists():
            print(
                "  WARNING: images exist but data/annotations/dfire_test.json is "
                "missing — the pipeline GT gate will fail. Re-run this script "
                "(e.g. with --dfire-archive after clearing data/raw/dfire) to "
                "regenerate the COCO annotations.",
                file=sys.stderr,
            )
    else:
        archive = args.dfire_archive
        if archive is not None:
            archive = Path(archive)
            if not archive.exists():
                print(f"ERROR: --dfire-archive not found: {archive}", file=sys.stderr)
                return 1
        else:
            url = args.dfire_url
            archive = args.staging / "dfire" / "DFire_images_labels.zip"
            print(f"  downloading D-Fire archive (OneDrive): {url}")
            url = url + ("&download=1" if "?" in url else "?download=1")
            if not _download(url, archive):
                print(
                    "  ERROR: could not download automatically (OneDrive may serve "
                    "a page or require a browser).\n"
                    f"  Manual fallback:\n"
                    f"    1. open {args.dfire_url}\n"
                    f"    2. download the archive and re-run with\n"
                    f"       --dfire-archive /path/to/DFire_images_labels.zip\n"
                    f"  Alternate mirror: {DFIRE_KAGGLE_URL}",
                    file=sys.stderr,
                )
                return 1
            print(f"  downloaded {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

        extracted = args.staging / "dfire" / "extracted"
        print(f"  extracting {archive} -> {extracted} ...")
        _extract_archive(archive, extracted)

        all_images = _walk_images(extracted)
        if not all_images:
            print(
                "ERROR: no images found inside the D-Fire archive. Check the "
                "archive contents and the extraction path.",
                file=sys.stderr,
            )
            return 1
        print(f"  found {len(all_images)} images")

        # Splits: honor embedded train/val/test dirs, else deterministic 80/20.
        splits: Dict[str, List[Path]] = {}
        for split in ("train", "val", "test"):
            cand = extracted / split
            sub = [p for p in _walk_images(cand)] if cand.is_dir() else []
            if sub:
                splits[split] = sub
        if not splits:
            n = len(all_images)
            tr = all_images[: int(n * 0.8)]
            te = all_images[int(n * 0.8) :]
            splits = {"train": tr, "test": te}
            print(
                "  archive has no embedded split dirs — using deterministic "
                f"80/20 train/test split ({len(tr)}/{len(te)})"
            )

        for split, images in splits.items():
            subset = _pick_subset(sorted(images), args.subset)
            dest_dir = raw_ds / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for img in subset:
                shutil.copy2(img, dest_dir / img.name)
                copied += 1
            print(f"  {split}: {copied} images -> {dest_dir}")

            coco = _yolo_to_coco(subset, split)
            ann_out = args.annotations_root / f"dfire_{split}.json"
            ann_out.parent.mkdir(parents=True, exist_ok=True)
            ann_out.write_text(json.dumps(coco, indent=2))
            print(
                f"  {split}: wrote {ann_out} "
                f"({len(coco['images'])} images, {len(coco['annotations'])} boxes)"
            )

        # Integrity checksums (policy: scripts + checksums only in git).
        sums_dir = args.sums_dir or (ROOT / "data" / "download_scripts" / "dfire")
        sums_dir.mkdir(parents=True, exist_ok=True)
        sums = sums_dir / "sha256sums.txt"
        with open(sums, "w") as fh:
            fh.write(f"{_sha256(archive)}  {archive.name}\n")
        print(f"  checksum recorded -> {sums}")
    return 0


# ---------------------------------------------------------------------------
# LADD (placeholder + manual steps — official repo offline)
# ---------------------------------------------------------------------------
def _dataset_present(raw_ds: Path) -> bool:
    return any(_walk_images(raw_ds))


def run_ladd(args: argparse.Namespace) -> int:
    raw_ds = args.raw_root / "ladd"
    if _dataset_present(raw_ds):
        print("  LADD images already present — skipping download.")
        return 0

    archive = Path(args.ladd_archive) if args.ladd_archive else None
    if archive is None:
        if not args.ladd_url or args.ladd_url == LADD_URL_PLACEHOLDER:
            print(
                "LADD manual download required (official repo huyhieupham/LADD is "
                "offline as of 2026-08-04).\n"
                "\n"
                "  Steps:\n"
                "    1. Obtain a verified LADD download link (author page, paper "
                "supplementary, or Zenodo record).\n"
                "    2. Either pass it here:\n"
                "         python download_datasets.py --dataset ladd --ladd-url <URL> "
                "--subset 10\n"
                "       or set it in the script constant LADD_URL_PLACEHOLDER.\n"
                "    3. Provide the COCO-style GT JSON with --ladd-gt (or place it "
                "at data/annotations/ladd_test.json) once the archive is in place.\n"
                "\n"
                "  No URL was guessed: replace the placeholder with a verified link "
                "before running (Milestone-1 policy, see README.md).",
                file=sys.stderr,
            )
            return 1

        # User supplied a verified URL — attempt download (zip/tar auto-detected).
        url = args.ladd_url
        suffix = Path(url.split("?")[0]).suffix or ".zip"
        archive = args.staging / "ladd" / f"ladd_archive{suffix}"
        print(f"  downloading LADD archive: {url}")
        if not _download(url, archive):
            print(
                "  ERROR: download failed. Check the URL and try the manual fallback "
                "(browser download, then re-run with --ladd-archive).",
                file=sys.stderr,
            )
            return 1
    else:
        if not archive.exists():
            print(f"ERROR: --ladd-archive not found: {archive}", file=sys.stderr)
            return 1

    extracted = args.staging / "ladd" / "extracted"
    _extract_archive(archive, extracted)
    sums_dir = args.sums_dir or (ROOT / "data" / "download_scripts" / "ladd")
    sums_dir.mkdir(parents=True, exist_ok=True)
    sums_file = sums_dir / "sha256sums.txt"
    with open(sums_file, "w") as fh:
        fh.write(f"{_sha256(archive)}  {archive.name}\n")
    print(f"  checksum recorded -> {sums_file}")
    all_images = _walk_images(extracted)
    if not all_images:
        print("ERROR: no images found in the LADD archive.", file=sys.stderr)
        return 1
    n = len(all_images)
    splits = {
        "train": all_images[: int(n * 0.8)],
        "test": all_images[int(n * 0.8) :],
    }
    for split, images in splits.items():
        subset = _pick_subset(sorted(images), args.subset)
        dest_dir = raw_ds / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy2(img, dest_dir / img.name)
        print(f"  {split}: {len(subset)} images -> {dest_dir}")
    if args.ladd_gt:
        gt = Path(args.ladd_gt)
        out = args.annotations_root / "ladd_test.json"
        shutil.copy2(gt, out)
        print(f"  GT copied -> {out}")
    else:
        print(
            "  NOTE: no --ladd-gt given — place the COCO GT JSON at "
            "data/annotations/ladd_test.json before running the pipeline."
        )
    return 0


# ---------------------------------------------------------------------------
# Link check (no download)
# ---------------------------------------------------------------------------
def check_links(args: argparse.Namespace) -> int:
    import urllib.request as u

    def head(url: str, extra: str = "") -> str:
        try:
            req = u.Request(url + extra, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
            with u.urlopen(req, timeout=15) as resp:
                return f"HTTP {resp.status} ({resp.headers.get('Content-Type', '?')})"
        except Exception as exc:  # noqa: BLE001 — report any failure
            return f"FAIL ({exc})"

    print(f"D-Fire OneDrive (images+labels): {head(DFIRE_DRIVE_URL)}")
    print(f"D-Fire OneDrive (?download=1):   {head(DFIRE_DRIVE_URL, '?download=1')}")
    print(f"D-Fire Kaggle mirror:            {head(DFIRE_KAGGLE_URL)}")
    print(
        "LADD: placeholder only — no verified URL yet. "
        "Replace LADD_URL_PLACEHOLDER and re-run --check-only."
    )
    return 0


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 1: download and organize LADD + D-Fire "
        "(raw data stays out of git; see data/download_scripts/README.md)."
    )
    parser.add_argument(
        "--dataset", choices=["ladd", "dfire"], default="dfire",
        help="which dataset to prepare (default: dfire)",
    )
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="pilot mode: copy only the first N images per split "
        "(the full official archive is still downloaded — OneDrive cannot "
        "do ranged downloads — but only the subset is copied to data/raw)",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANN_ROOT)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--dfire-url", default=DFIRE_DRIVE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--dfire-archive", default=None, type=Path,
                        help="use a pre-downloaded D-Fire archive instead of downloading")
    parser.add_argument("--ladd-url", default=LADD_URL_PLACEHOLDER,
                        help="verified LADD download URL (replace placeholder)")
    parser.add_argument("--ladd-archive", default=None, type=Path,
                        help="use a pre-downloaded LADD archive")
    parser.add_argument("--ladd-gt", default=None, type=Path,
                        help="COCO-style LADD GT JSON to copy to data/annotations/")
    parser.add_argument("--check-only", action="store_true",
                        help="verify configured URLs are reachable; do not download")
    parser.add_argument("--sums-dir", default=None, type=Path,
                        help="dir for sha256sums.txt (default: "
                             "data/download_scripts/<dataset>)")
    args = parser.parse_args(argv)
    args.staging.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        return check_links(args)
    if args.dataset == "dfire":
        return run_dfire(args)
    return run_ladd(args)


if __name__ == "__main__":
    raise SystemExit(main())
