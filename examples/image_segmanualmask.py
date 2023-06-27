from metaseg import SegManualMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.


def main(src: str = "image.png") -> None:
    SegManualMaskPredictor().image_predict(
        source=src,
        model_type="vit_l",  # vit_l, vit_h, vit_b
        input_point=[[100, 100], [200, 200]],
        input_label=[0, 1],
        input_box=[
            100,
            100,
            200,
            200,
        ],  # or [[100, 100, 200, 200], [100, 100, 200, 200]]
        multimask_output=False,
        random_color=False,
        show=True,
        save=False,
    )


if __name__ == "__main__":
    main()
