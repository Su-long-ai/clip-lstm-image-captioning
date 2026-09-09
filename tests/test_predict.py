from __future__ import annotations

from predict import build_reference_map


def test_reference_map_uses_coco_image_ids() -> None:
    payload = {
        "images": [{"id": 7, "file_name": "nested/a.jpg"}],
        "annotations": [
            {"image_id": 7, "caption": "first"},
            {"image_id": 7, "caption": "second"},
        ],
    }
    assert build_reference_map(payload, ["a.jpg", "b.jpg"]) == {"a.jpg": ["first", "second"], "b.jpg": []}


def test_reference_map_has_filename_fallback() -> None:
    payload = {"annotations": [{"image_id": "sample", "caption": "caption"}]}
    assert build_reference_map(payload, ["sample.jpg"]) == {"sample.jpg": ["caption"]}
