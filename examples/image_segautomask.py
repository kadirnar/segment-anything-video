from metaseg import SegAutoMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.


def main(src: str = "image.png") -> None:
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
    main()
