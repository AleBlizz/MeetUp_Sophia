"""
KO/OK Image Detector — Elasticsearch Inference API + Elasticsearch
-------------------------------------------------------------------
Same KO/OK strategy as ko_ok_detector_jina.py but embeddings are generated
via the Elasticsearch inference endpoint (e.g. .jina-clip-v2) instead of
calling the Jina API directly.

Inference endpoint call:
  POST {ES_HOST}/_inference/embedding/{INFERENCE_ENDPOINT_ID}
  {
    "input": [
      {
        "content": {
          "type": "image",
          "format": "base64",
          "value": "data:image/jpg;base64,<base64_data>"
        }
      }
    ]
  }

Commands:
  python3 ko_ok_detector_es_inference.py ingest-ko --folder ./ko_images
  python3 ko_ok_detector_es_inference.py ingest-ok --folder ./ok_images
  python3 ko_ok_detector_es_inference.py check     --image  ./new_image.jpg
  python3 ko_ok_detector_es_inference.py check     --folder ./test_images
"""

import csv
import os
import base64
import argparse
from pathlib import Path
from collections import Counter

import requests
from dotenv import load_dotenv

load_dotenv()

ES_HOST             = os.getenv("ES_HOST")
ES_API_KEY          = os.getenv("ES_API_KEY")
INFERENCE_ES_HOST   = os.getenv("INFERENCE_ES_HOST", os.getenv("ES_HOST"))
INFERENCE_ES_API_KEY = os.getenv("INFERENCE_ES_API_KEY", os.getenv("ES_API_KEY"))
INFERENCE_ENDPOINT  = os.getenv("INFERENCE_ENDPOINT", ".jina-clip-v2")
KO_INDEX            = os.getenv("KO_INDEX_ES_INFERENCE", "transistor_defects_ko_library")
KO_THRESHOLD        = float(os.getenv("KO_THRESHOLD", "0.90"))
KO_MARGIN           = float(os.getenv("KO_MARGIN", "0.01"))
KO_MIN_VOTES        = int(os.getenv("KO_MIN_VOTES", "2"))
KNN_K               = int(os.getenv("KNN_K", "5"))
KNN_CANDIDATES      = int(os.getenv("KNN_CANDIDATES", "200"))
EMBEDDING_DIMS      = int(os.getenv("EMBEDDING_DIMS", "2048"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ES_HEADERS = {
    "Authorization": f"ApiKey {ES_API_KEY}",
    "Content-Type": "application/json",
}
INFERENCE_HEADERS = {
    "Authorization": f"ApiKey {INFERENCE_ES_API_KEY}",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# Embedding — ES inference endpoint
# ---------------------------------------------------------------------------

def _mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    return "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"


def _b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_embedding(image_path: Path) -> list[float]:
    url = f"{INFERENCE_ES_HOST}/_inference/embedding/{INFERENCE_ENDPOINT}"
    payload = {
        "input": [
            {
                "content": {
                    "type":   "image",
                    "format": "base64",
                    "value":  f"data:{_mime(image_path)};base64,{_b64(image_path)}",
                }
            }
        ]
    }
    resp = requests.post(url, headers=INFERENCE_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]["embedding"]


# ---------------------------------------------------------------------------
# Elasticsearch helpers
# ---------------------------------------------------------------------------

def ensure_index():
    url = f"{ES_HOST}/{KO_INDEX}"
    if requests.head(url, headers=ES_HEADERS, timeout=10).status_code == 404:
        mapping = {
            "mappings": {
                "properties": {
                    "filename":     {"type": "keyword"},
                    "verdict":      {"type": "keyword"},
                    "defect_label": {"type": "keyword"},
                    "image_b64":    {"type": "binary", "doc_values": False},
                    "embedding": {
                        "type":       "dense_vector",
                        "dims":       EMBEDDING_DIMS,
                        "index":      True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        resp = requests.put(url, headers=ES_HEADERS, json=mapping, timeout=10)
        resp.raise_for_status()
        print(f"Created index '{KO_INDEX}' ({EMBEDDING_DIMS} dims)")
    else:
        # Add image_b64 field to existing index if not present
        mapping_url = f"{url}/_mapping"
        current = requests.get(mapping_url, headers=ES_HEADERS, timeout=10).json()
        props = current.get(KO_INDEX, {}).get("mappings", {}).get("properties", {})
        if "image_b64" not in props:
            patch = {"properties": {"image_b64": {"type": "binary", "doc_values": False}}}
            requests.put(mapping_url, headers=ES_HEADERS, json=patch, timeout=10)


def index_doc(doc_id: str, filename: str, verdict: str,
              defect_label: str | None, embedding: list[float],
              image_b64: str | None = None):
    url = f"{ES_HOST}/{KO_INDEX}/_doc/{doc_id}"
    doc = {
        "filename":     filename,
        "verdict":      verdict,
        "defect_label": defect_label,
        "embedding":    embedding,
    }
    if image_b64:
        doc["image_b64"] = image_b64
    resp = requests.put(url, headers=ES_HEADERS, json=doc, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]


def knn_filtered(embedding: list[float], verdict: str,
                 include_image: bool = False) -> list[dict]:
    url = f"{ES_HOST}/{KO_INDEX}/_search"
    source_fields = ["filename", "verdict", "defect_label"]
    if include_image:
        source_fields.append("image_b64")
    payload = {
        "knn": {
            "field":          "embedding",
            "query_vector":   embedding,
            "k":              KNN_K,
            "num_candidates": KNN_CANDIDATES,
            "filter": {"term": {"verdict": verdict}},
        },
        "_source": source_fields,
    }
    resp = requests.post(url, headers=ES_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["hits"]["hits"]


# ---------------------------------------------------------------------------
# Collect (image_path, defect_label) pairs from a folder
# ---------------------------------------------------------------------------

def collect_images(folder: Path, label_override: str | None) -> list[tuple[Path, str | None]]:
    pairs: list[tuple[Path, str | None]] = []

    if label_override is not None:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((p, label_override))
        return pairs

    subdirs = [d for d in sorted(folder.iterdir()) if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            for p in sorted(subdir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    pairs.append((p, subdir.name))
    else:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((p, None))

    return pairs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ingest(folder: Path, verdict: str, label_override: str | None):
    pairs = collect_images(folder, label_override)
    if not pairs:
        print(f"No images found in {folder.resolve()}")
        return

    ensure_index()

    if verdict == "KO":
        label_counts = Counter(label for _, label in pairs)
        print(f"Ingesting {len(pairs)} KO image(s) — defect labels: {dict(label_counts)}\n")
    else:
        print(f"Ingesting {len(pairs)} OK image(s)\n")

    for image_path, defect_label in pairs:
        prefix = defect_label or verdict
        doc_id = f"{prefix}__{image_path.stem}"
        tag    = f"[{defect_label}]" if defect_label else "[OK]"
        print(f"  {tag}  {image_path.name}  →  id={doc_id}")
        try:
            embedding = get_embedding(image_path)
            result    = index_doc(doc_id, image_path.name, verdict, defect_label,
                                  embedding, _b64(image_path))
            print(f"    dims={len(embedding)}  ES={result}")
        except requests.HTTPError as e:
            print(f"    HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"    Error: {e}")


def cmd_check(images: list[Path]):
    print(f"Index          : {KO_INDEX}")
    print(f"Inference host : {INFERENCE_ES_HOST}")
    print(f"Endpoint       : {INFERENCE_ENDPOINT}")
    print(f"Threshold      : {KO_THRESHOLD}\n")

    for image_path in images:
        if not image_path.exists():
            print(f"Not found: {image_path}\n")
            continue

        print(f"Checking: {image_path.name}")
        try:
            embedding    = get_embedding(image_path)
            ko_hits      = knn_filtered(embedding, "KO")
            ok_hits      = knn_filtered(embedding, "OK")

            top_ko_score = ko_hits[0]["_score"] if ko_hits else 0.0
            top_ok_score = ok_hits[0]["_score"] if ok_hits else 0.0
            ko_votes     = sum(1 for h in ko_hits if h["_score"] >= KO_THRESHOLD)
            avg_ko       = sum(h["_score"] for h in ko_hits) / len(ko_hits) if ko_hits else 0.0
            avg_ok       = sum(h["_score"] for h in ok_hits) / len(ok_hits) if ok_hits else 0.0
            margin       = avg_ko - avg_ok

            is_ko   = (avg_ko >= KO_THRESHOLD
                       and ko_votes >= KO_MIN_VOTES
                       and margin >= KO_MARGIN)
            verdict = "KO" if is_ko else "OK"

            matched_defects = [
                h["_source"]["defect_label"]
                for h in ko_hits
                if h["_score"] >= KO_THRESHOLD and h["_source"]["defect_label"]
            ]
            defects = ", ".join(sorted(set(matched_defects))) if matched_defects else "—"

            print(f"  Verdict         : {verdict}")
            print(f"  Defect(s)       : {defects}")
            print(f"  Top KO score    : {top_ko_score:.4f}")
            print(f"  Avg KO score    : {avg_ko:.4f}  (threshold={KO_THRESHOLD})")
            print(f"  KO votes        : {ko_votes}/{KNN_K}  (min={KO_MIN_VOTES})")
            print(f"  Margin avg KO-OK: {margin:+.4f}  (min={KO_MARGIN})")
            print(f"  Avg OK score    : {avg_ok:.4f}")
            print(f"  Top OK score    : {top_ok_score:.4f}")

            if ko_hits:
                print(f"  KO neighbours :")
                for hit in ko_hits:
                    marker = "★" if hit["_score"] >= KO_THRESHOLD else " "
                    label  = hit["_source"]["defect_label"] or "KO"
                    print(f"    {marker} [{label:<12}]  {hit['_source']['filename']:<45}  score={hit['_score']:.4f}")

            if ok_hits:
                print(f"  OK neighbours :")
                for hit in ok_hits:
                    print(f"      [OK          ]  {hit['_source']['filename']:<45}  score={hit['_score']:.4f}")

            print()

        except requests.HTTPError as e:
            print(f"  HTTP error {e.response.status_code}: {e.response.text}\n")
        except Exception as e:
            print(f"  Error: {e}\n")


# ---------------------------------------------------------------------------
# Debug scores — CSV export for threshold analysis
# ---------------------------------------------------------------------------

def _score_image(image_path: Path) -> dict | None:
    """Return raw score metrics for one image, or None on error."""
    try:
        embedding    = get_embedding(image_path)
        ko_hits      = knn_filtered(embedding, "KO")
        ok_hits      = knn_filtered(embedding, "OK")

        top_ko       = ko_hits[0]["_score"] if ko_hits else 0.0
        top_ok       = ok_hits[0]["_score"] if ok_hits else 0.0
        avg_ko       = sum(h["_score"] for h in ko_hits) / len(ko_hits) if ko_hits else 0.0
        avg_ok       = sum(h["_score"] for h in ok_hits) / len(ok_hits) if ok_hits else 0.0
        ko_votes     = sum(1 for h in ko_hits if h["_score"] >= KO_THRESHOLD)
        margin       = avg_ko - avg_ok
        is_ko        = avg_ko >= KO_THRESHOLD and ko_votes >= KO_MIN_VOTES and margin >= KO_MARGIN

        return {
            "filename":    image_path.name,
            "top_ko":      round(top_ko, 6),
            "avg_ko":      round(avg_ko, 6),
            "ko_votes":    ko_votes,
            "avg_ok":      round(avg_ok, 6),
            "top_ok":      round(top_ok, 6),
            "margin":      round(margin, 6),
            "predicted":   "KO" if is_ko else "OK",
        }
    except Exception as e:
        print(f"  [skip] {image_path.name}: {e}")
        return None


def cmd_debug_scores(ko_folder: Path | None, ok_folder: Path | None, output: Path):
    rows: list[dict] = []

    def process(folder: Path, true_label: str):
        pairs = collect_images(folder, label_override=None)
        for image_path, subfolder_label in pairs:
            actual_label = subfolder_label if subfolder_label else true_label
            print(f"  scoring [{actual_label}]  {image_path.name} ...", end=" ", flush=True)
            row = _score_image(image_path)
            if row:
                row["true_label"] = actual_label
                row["correct"]    = row["predicted"] == true_label
                rows.append(row)
                print(f"avg_ko={row['avg_ko']:.4f}  avg_ok={row['avg_ok']:.4f}  → {row['predicted']}")

    if ko_folder:
        print(f"\nScoring KO images from: {ko_folder}")
        process(ko_folder, "KO")

    if ok_folder:
        print(f"\nScoring OK images from: {ok_folder}")
        process(ok_folder, "OK")

    if not rows:
        print("No results to write.")
        return

    fieldnames = ["filename", "true_label", "predicted", "correct",
                  "avg_ko", "top_ko", "ko_votes", "avg_ok", "top_ok", "margin"]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved → {output.resolve()}")

    # Summary stats per true label
    ko_group = [r for r in rows if r["true_label"] == "KO"]
    ok_group = [r for r in rows if r["true_label"] == "OK"]

    for label, group in (("KO", ko_group), ("OK", ok_group)):
        if not group:
            continue
        avg_ko_vals = [r["avg_ko"] for r in group]
        margin_vals = [r["margin"] for r in group]
        correct     = sum(1 for r in group if r["correct"])
        print(f"\n  {label} group ({len(group)} images)")
        print(f"    avg_ko  min={min(avg_ko_vals):.4f}  max={max(avg_ko_vals):.4f}"
              f"  mean={sum(avg_ko_vals)/len(avg_ko_vals):.4f}")
        print(f"    margin  min={min(margin_vals):.4f}  max={max(margin_vals):.4f}"
              f"  mean={sum(margin_vals)/len(margin_vals):.4f}")
        print(f"    accuracy at current settings: {correct}/{len(group)} correct")

    # Threshold & margin recommendation (only when both groups present)
    if ko_group and ok_group:
        ko_avg_ko_vals  = [r["avg_ko"] for r in ko_group]
        ok_avg_ko_vals  = [r["avg_ko"] for r in ok_group]
        ko_margin_vals  = [r["margin"] for r in ko_group]
        ok_margin_vals  = [r["margin"] for r in ok_group]

        ko_min   = min(ko_avg_ko_vals)   # lowest KO score — must be above threshold
        ok_max   = max(ok_avg_ko_vals)   # highest OK score — must be below threshold

        print("\n" + "─" * 55)
        print("  THRESHOLD RECOMMENDATION")
        print("─" * 55)

        if ok_max < ko_min:
            suggested_threshold = round((ko_min + ok_max) / 2, 4)
            gap = round(ko_min - ok_max, 4)
            print(f"  Clean separation found — gap = {gap:.4f}")
            print(f"    Lowest  KO avg_ko : {ko_min:.4f}")
            print(f"    Highest OK avg_ko : {ok_max:.4f}")
            print(f"  → Suggested KO_THRESHOLD = {suggested_threshold}")
        else:
            overlap = round(ok_max - ko_min, 4)
            print(f"  WARNING: groups overlap by {overlap:.4f} — no perfect threshold exists.")
            print(f"    Lowest  KO avg_ko : {ko_min:.4f}")
            print(f"    Highest OK avg_ko : {ok_max:.4f}")
            print(f"  → Use the midpoint as a starting point: {round((ko_min + ok_max) / 2, 4)}")
            print(f"    Then inspect overlapping rows in the CSV manually.")

        # Margin recommendation: min margin seen in KO group (safe lower bound)
        suggested_margin = round(max(0.0, min(ko_margin_vals)), 4)
        print(f"\n  Lowest margin in KO group : {min(ko_margin_vals):.4f}")
        print(f"  Lowest margin in OK group : {min(ok_margin_vals):.4f}")
        print(f"  → Suggested KO_MARGIN     = {suggested_margin}")
        print("─" * 55)
        print(f"\n  Add to your .env:")
        print(f"    KO_THRESHOLD={suggested_threshold}")
        print(f"    KO_MARGIN={suggested_margin}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KO/OK image detector via ES inference endpoint + Elasticsearch"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ko = sub.add_parser("ingest-ko", help="Index known-bad (KO) images")
    ko.add_argument("--folder", type=Path, required=True,
                    help="Root folder (subfolders = defect labels) or flat folder")
    ko.add_argument("--label", default=None,
                    help="Override defect label for all images in a flat folder")

    ok = sub.add_parser("ingest-ok", help="Index known-good (OK) images")
    ok.add_argument("--folder", type=Path, required=True,
                    help="Folder of OK images (flat, no subfolders needed)")

    chk = sub.add_parser("check", help="Check one image or a whole folder")
    grp = chk.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image",  type=Path, help="Single image to check")
    grp.add_argument("--folder", type=Path, help="Folder of images to check")

    dbg = sub.add_parser("debug-scores",
                         help="Score labelled images and export a CSV for threshold analysis")
    dbg.add_argument("--ko-folder", type=Path, default=None,
                     help="Folder of known-KO images (subfolders = defect labels)")
    dbg.add_argument("--ok-folder", type=Path, default=None,
                     help="Folder of known-OK images")
    dbg.add_argument("--output",    type=Path, default=Path("scores.csv"),
                     help="Output CSV file (default: scores.csv)")

    args = parser.parse_args()

    if args.command == "ingest-ko":
        cmd_ingest(args.folder, "KO", args.label)
    elif args.command == "ingest-ok":
        cmd_ingest(args.folder, "OK", None)
    elif args.command == "check":
        if args.image:
            cmd_check([args.image])
        else:
            images = [
                p for p in sorted(args.folder.iterdir())
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            cmd_check(images)
    elif args.command == "debug-scores":
        if not args.ko_folder and not args.ok_folder:
            parser.error("debug-scores requires at least --ko-folder or --ok-folder")
        cmd_debug_scores(args.ko_folder, args.ok_folder, args.output)


if __name__ == "__main__":
    main()
