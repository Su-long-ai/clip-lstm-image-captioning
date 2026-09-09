from __future__ import annotations

from make_coco import convert_records


def test_convert_records_preserves_splits_and_captions() -> None:
    result = convert_records(
        [
            {"id": 1, "split": "train", "file_name": "a.jpg", "captions": ["one", "two"]},
            {"img_id": 2, "split": "val", "img_name": "b.jpg", "captions": ["three"]},
        ]
    )
    assert result["train"]["images"] == [{"id": 1, "file_name": "a.jpg"}]
    assert [item["caption"] for item in result["train"]["annotations"]] == ["one", "two"]
    assert result["val"]["annotations"][0]["id"] == 2
