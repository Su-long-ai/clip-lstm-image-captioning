import argparse
import json
import os

import torch
from PIL import Image
from rich.console import Console


console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions for a folder of images")
    parser.add_argument("--testfolder", required=True, help="Folder containing test images")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint (.pth)")
    parser.add_argument("--test-json", required=True, help="COCO-style reference captions JSON")
    parser.add_argument("--output-json", default="gen_testcaption.json", help="Generated captions output")
    parser.add_argument("--reference-json", default="refer_testcaption.json", help="Matched reference captions output")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size used during generation")
    parser.add_argument("--debug", "-d", action="store_true", help="Process only the first 100 images and print details")
    return parser.parse_args()


def build_reference_map(test_captions, image_files):
    """Map COCO-style annotations to local filenames, including one-to-many image IDs."""
    refer_results = {f: [] for f in image_files}

    if test_captions.get("images"):
        id_to_filenames = {}
        for img in test_captions["images"]:
            fname = os.path.basename(img["file_name"])
            id_to_filenames.setdefault(img["id"], []).append(fname)

        for annotation in test_captions.get("annotations", []):
            for fname in id_to_filenames.get(annotation["image_id"], []):
                if fname in refer_results:
                    refer_results[fname].append(annotation["caption"])

    # Fallback for datasets where annotations use a filename-like image_id.
    if all(not captions for captions in refer_results.values()):
        for annotation in test_captions.get("annotations", []):
            annotation_id = str(annotation["image_id"])
            for fname in image_files:
                stem = os.path.splitext(fname)[0]
                if annotation_id in {stem, fname}:
                    refer_results[fname].append(annotation["caption"])

    return refer_results


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = checkpoint["model"]
    model.eval()

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(
        f for f in os.listdir(args.testfolder)
        if os.path.isfile(os.path.join(args.testfolder, f)) and f.lower().endswith(image_extensions)
    )
    if args.debug:
        image_files = image_files[:100]

    with open(args.test_json, "r", encoding="utf-8") as file:
        test_captions = json.load(file)
    references = build_reference_map(test_captions, image_files)

    results = {}
    with torch.no_grad():
        for filename in image_files:
            try:
                image = Image.open(os.path.join(args.testfolder, filename)).convert("RGB")
                captions = model.generate(image, beam_size=args.beam_size)
                caption = captions[0][:-4] if captions else ""
                results[filename] = [caption]
                if args.debug:
                    console.log(f"{filename}: {captions}", style="blue")
            except Exception as exc:
                console.log(f"Failed to process {filename}: {exc}", style="red bold")

    # Keep only samples with a matched reference caption so evaluation is well-defined.
    final_results = {name: results[name] for name in results if references.get(name)}
    final_references = {name: references[name] for name in final_results}
    unmatched = len(results) - len(final_results)
    if unmatched:
        console.log(f"Skipped {unmatched} images without matched reference captions.", style="yellow")

    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(final_results, file, ensure_ascii=False, indent=2)
    with open(args.reference_json, "w", encoding="utf-8") as file:
        json.dump(final_references, file, ensure_ascii=False, indent=2)

    console.log(
        f"Saved {len(final_results)} caption pairs to {args.output_json} and {args.reference_json}",
        style="green bold",
    )


if __name__ == "__main__":
    main()
