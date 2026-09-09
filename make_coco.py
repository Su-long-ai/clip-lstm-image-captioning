from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_KEYS = ("img_path", "image_path", "image_name", "img_name", "file_name")


def convert_records(records: list[dict]) -> dict[str, dict[str, list[dict]]]:
    outputs = {split: {"images": [], "annotations": []} for split in SPLITS}
    annotation_id = 0
    for index, item in enumerate(records):
        split = str(item.get("split", "train")).lower()
        if split not in outputs:
            raise ValueError(f"Unsupported split {split!r} at record {index}")
        image_id = item.get("id", item.get("img_id", index))
        filename = next((item.get(key) for key in IMAGE_KEYS if item.get(key)), None)
        if filename is None:
            raise ValueError(f"Record {index} has no image filename; checked keys {IMAGE_KEYS}")
        filename = str(filename)
        outputs[split]["images"].append({"id": image_id, "file_name": filename})
        for caption in item.get("captions", []):
            outputs[split]["annotations"].append(
                {"id": annotation_id, "image_id": image_id, "caption": str(caption)}
            )
            annotation_id += 1
    return outputs


def find_image(source_root: Path, relative_name: str) -> Path | None:
    direct = source_root / relative_name
    if direct.is_file():
        return direct
    basename = Path(relative_name).name
    return next((path for path in source_root.rglob(basename) if path.is_file()), None)


def export_dataset(source_json: Path, source_images: Path, output_root: Path) -> dict[str, int]:
    records = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Source JSON must be a list of image/caption records")
    converted = convert_records(records)
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    copied = {split: 0 for split in SPLITS}
    for split in SPLITS:
        image_dir = output_root / f"{split}2014"
        image_dir.mkdir(parents=True, exist_ok=True)
        for image in converted[split]["images"]:
            source = find_image(source_images, image["file_name"])
            if source is None:
                raise FileNotFoundError(f"Image not found for {image['file_name']!r}")
            destination = image_dir / Path(image["file_name"]).name
            shutil.copy2(source, destination)
            image["file_name"] = destination.name
            copied[split] += 1
        annotation_path = annotations_dir / f"captions_{split}2014.json"
        annotation_path.write_text(json.dumps(converted[split], ensure_ascii=False, indent=2), encoding="utf-8")
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert split image/caption records to a COCO-style directory layout")
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = export_dataset(args.source_json, args.source_images, args.output)
    print(" ".join(f"{split}={count}" for split, count in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
