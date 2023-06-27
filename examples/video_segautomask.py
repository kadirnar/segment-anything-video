from metaseg import SegAutoMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.


# For video
def main(src: str = "video.mp4") -> None:
    SegAutoMaskPredictor().video_predict(
        source=src,
        model_type="vit_l",  # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=1000,
        output_path="output.mp4",
    )


if __name__ == "__main__":
    main()
