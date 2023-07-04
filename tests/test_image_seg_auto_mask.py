"""Test image segmentation with automatic mask prediction."""

from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor as cv_cvtColor
from cv2 import imread as cv_imread

from metaseg import SegAutoMaskPredictor


def test_seg_auto_mask_vit_l() -> None:
    """Test test_seg_auto_mask_vit_l function."""
    SegAutoMaskPredictor().image_predict(
        source="tests/images/dog.jpg",
        model_type="vit_l",  # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=0,
        output_path="output.jpg",
        show=False,
        save=False,
    )


def test_seg_auto_mask_vit_b() -> None:
    """Test test_seg_auto_mask_vit_b function."""
    SegAutoMaskPredictor().image_predict(
        source="tests/images/truck.jpg",
        model_type="vit_b",  # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=0,
        output_path="output.jpg",
        show=False,
        save=False,
    )


def test_seg_auto_mask_vit_h() -> None:
    """Test test_seg_auto_mask_vit_h function."""
    SegAutoMaskPredictor().image_predict(
        source="tests/images/groceries.jpg",
        model_type="vit_h",  # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=0,
        output_path="output.jpg",
        show=False,
        save=False,
    )


def test_cv_seg_auto_mask_vit_l() -> None:
    """Test test_cv_seg_auto_mask_vit_l function."""
    image = cv_imread("tests/images/truck.jpg")
    image = cv_cvtColor(image, COLOR_BGR2RGB)
    SegAutoMaskPredictor().image_predict(
        source=image,
        model_type="vit_l",  # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=0,
        output_path="output.jpg",
        show=False,
        save=False,
    )
