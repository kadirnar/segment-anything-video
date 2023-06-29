"""Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import warnings

import torch
from numpy import ndarray
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from metaseg.generator import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from metaseg.utils.onnx import SamOnnxModel

parser = argparse.ArgumentParser(
    description="Export the SAM prompt encoder and mask decoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM model checkpoint.",
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="In ['default', 'vit_b', 'vit_l']. Which type of SAM model to export.",
)

parser.add_argument(
    "--return-single-mask",
    action="store_true",
    help=(
        "If true, the exported ONNX model will only return the best mask, "
        "instead of returning multiple masks. For high resolution images "
        "this can improve runtime when upscaling masks is expensive."
    ),
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic "
        "from onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--return-extra-metrics",
    action="store_true",
    help=(
        "The model will return five results: (masks, scores, stability_scores, "
        "areas, low_res_logits) instead of the usual three. This can be "
        "significantly slower for high resolution outputs."
    ),
)


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    """Export a PyTorch model to ONNX format.

    Args:
        model_type: The type of model to export
            (e.g. "vit_b", "vit_l", etc.).
        checkpoint: The path to the PyTorch checkpoint file to
            load the model weights from.
        output: The path to save the exported ONNX model to.
        opset: The ONNX operator set version to use.
        return_single_mask: Whether to return a single
            attention mask for all layers.
        gelu_approximate: Whether to use the approximate GELU function.
        use_stability_score: Whether to use the stability
            score to select attention heads.
        return_extra_metrics: Whether to return extra metrics during export.

    Returns:
        None

    Raises:
        ValueError: If an invalid model type is specified.
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if model_type == "vit_b":
        sam = build_sam_vit_b(checkpoint)
    elif model_type == "vit_l":
        sam = build_sam_vit_l(checkpoint)
    else:
        sam = build_sam_vit_h(checkpoint)

    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(
            low=0, high=1024, size=(1, 5, 2), dtype=torch.float
        ),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
    ort_session = InferenceSession(output)
    _ = ort_session.run(None, ort_inputs)


def to_numpy(tensor: torch.Tensor) -> ndarray:
    """Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor: The PyTorch tensor to convert.

    Returns:
        The NumPy array representation of the input tensor.

    Raises:
        TypeError: If the input tensor is not a PyTorch tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    quantize_dynamic(
        model_input=args.output,
        model_output=args.quantize_out,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )
