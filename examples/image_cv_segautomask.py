from cv2 import COLOR_BGR2RGB, Mat
from cv2 import cvtColor as cv_cvtColor
from cv2 import imread as cv_imread

from metaseg import SegAutoMaskPredictor


# If gpu memory is not enough, reduce the points_per_side and points_per_batch.
def main(src: Mat) -> None:
    SegAutoMaskPredictor().image_predict(
        source=src,
        model_type="vit_l",  # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=0,
        output_path="output.jpg",
        show=True,
        save=False,
    )


if __name__ == "__main__":
    image = cv_imread("image.png")
    image = cv_cvtColor(image, COLOR_BGR2RGB)
    main(image)
