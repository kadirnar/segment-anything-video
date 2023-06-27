from metaseg import SegManualMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.


def main(src: str = "video.mp4") -> None:
    SegManualMaskPredictor().video_predict(
        source=src,
        model_type="vit_l",  # vit_l, vit_h, vit_b
        input_point=[0, 0, 100, 100],
        input_label=[0, 1],
        input_box=None,
        multimask_output=False,
        random_color=False,
        output_path="output.mp4",
    )


if __name__ == "__main__":
    main()
