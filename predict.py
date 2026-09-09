from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console

console = Console()
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate captions for a folder of images")
    parser.add_argument("--testfolder", type=Path, required=True, help="Folder containing test images")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Refactored training checkpoint (.pth)")
    parser.add_argument("--test-json", type=Path, required=True, help="COCO-style reference captions JSON")
    parser.add_argument("--output-json", type=Path, default=Path("gen_testcaption.json"))
    parser.add_argument("--reference-json", type=Path, default=Path("refer_testcaption.json"))
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--max-caption-length", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--debug", "-d", action="store_true", help="Process only the first 100 images")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_reference_map(test_captions: dict, image_files: list[str]) -> dict[str, list[str]]:
    """Map COCO-style annotations to local filenames, including repeated image IDs."""

    references = {filename: [] for filename in image_files}
    id_to_filenames: dict[object, list[str]] = {}
    for image in test_captions.get("images", []):
        if "id" not in image or "file_name" not in image:
            continue
        filename = os.path.basename(str(image["file_name"]))
        id_to_filenames.setdefault(image["id"], []).append(filename)

    for annotation in test_captions.get("annotations", []):
        if "image_id" not in annotation or "caption" not in annotation:
            continue
        for filename in id_to_filenames.get(annotation["image_id"], []):
            if filename in references:
                references[filename].append(str(annotation["caption"]))

    if all(not captions for captions in references.values()):
        for annotation in test_captions.get("annotations", []):
            if "image_id" not in annotation or "caption" not in annotation:
                continue
            annotation_id = str(annotation["image_id"])
            for filename in image_files:
                stem = Path(filename).stem
                if annotation_id in {stem, filename}:
                    references[filename].append(str(annotation["caption"]))
    return references


def main() -> int:
    args = parse_args()
    if args.beam_size <= 0 or args.max_caption_length <= 0:
        raise ValueError("beam-size and max-caption-length must be positive")
    if not args.testfolder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {args.testfolder}")

    device = resolve_device(args.device)
    from model import CaptionClipModel

    model = CaptionClipModel.from_checkpoint(args.checkpoint, device=device)
    image_files = sorted(
        path.name for path in args.testfolder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.debug:
        image_files = image_files[:100]

    with args.test_json.open("r", encoding="utf-8") as handle:
        test_captions = json.load(handle)
    references = build_reference_map(test_captions, image_files)

    generated: dict[str, list[str]] = {}
    with torch.no_grad():
        for filename in image_files:
            try:
                with Image.open(args.testfolder / filename) as image:
                    captions = model.generate(
                        image.convert("RGB"),
                        beam_size=args.beam_size,
                        max_caption_length=args.max_caption_length,
                    )
                generated[filename] = [captions[0] if captions else ""]
                if args.debug:
                    console.log(f"{filename}: {captions}", style="blue")
            except Exception as exc:
                console.log(f"Failed to process {filename}: {exc}", style="red bold")

    final_results = {name: generated[name] for name in generated if references.get(name)}
    final_references = {name: references[name] for name in final_results}
    unmatched = len(generated) - len(final_results)
    if unmatched:
        console.log(f"Skipped {unmatched} images without matched reference captions.", style="yellow")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.reference_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    args.reference_json.write_text(json.dumps(final_references, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(
        f"Saved {len(final_results)} caption pairs to {args.output_json} and {args.reference_json}",
        style="green bold",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
