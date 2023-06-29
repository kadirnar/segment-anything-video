from metaseg import SahiAutoSegmentation, sahi_sliced_predict


def main(src: str = "image.png") -> None:
    img_path = src
    boxes = sahi_sliced_predict(
        image_path=img_path,
        detection_model_type="yolov5",  # yolov8, detectron2, mmdetection, torchvision
        detection_model_path="yolov5l6.pt",
        conf_th=0.25,
        image_size=1280,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    SahiAutoSegmentation().image_predict(
        source=img_path,
        model_type="vit_b",
        input_box=boxes,
        multimask_output=False,
        random_color=False,
        show=True,
        save=False,
    )


if __name__ == "__main__":
    main()
