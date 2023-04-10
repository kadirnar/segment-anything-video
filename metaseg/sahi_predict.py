from metaseg import SegManualMaskPredictor


def sahi_predictor(image_path, model_type, model_path, conf_th, device):

    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_prediction, get_sliced_prediction, predict
    except ImportError:
        raise ImportError("Please install SAHI library using 'pip install sahi'.")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf_th,
        device=device,
    )
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    result = get_prediction(image_path, detection_model)
    output = result.object_prediction_list
    boxes = []
    for i in output:
        boxes.append(i.bbox.to_xyxy())

    seg_manual_mask_generator = SegManualMaskPredictor().save_image(
        source=image_path,
        model_type="vit_l",
        input_point=None,
        input_label=None,
        input_box=boxes,
        multimask_output=False,
    )
    return seg_manual_mask_generator
