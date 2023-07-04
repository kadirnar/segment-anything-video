"""Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
from torch import nn
from torch.nn import functional as F

from metaseg.modeling import Sam
from metaseg.utils.amg import calculate_stability_score


class SamOnnxModel(nn.Module):
    """This model should not be called directly, but is used in ONNX export.

    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        """Constructor for the SamOnnxModel class, which is used for ONNX export.

        Args:
            model (Sam): A Sam model instance,
                which is used to decode masks.
            return_single_mask (bool): Whether to return a single mask
                or multiple masks.
            use_stability_score (bool, optional): Whether to use a stability score.
                Defaults to False.
            return_extra_metrics (bool, optional): Whether to return extra metrics.
                Defaults to False.
        """
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    @staticmethod
    def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        """Resizes the input image size tensor so that.

            its longest side is equal to the given longest side.

        Args:
            input_image_size (torch.Tensor): A tensor of shape (2,)
                representing the input image size (height, width).
            longest_side (int): The desired length of the longest side of the image.

        Returns:
            torch.Tensor: A tensor of shape (2,)
                representing the resized image size (height, width).
        """
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:
        """Embeds the given points using the given embeddings.

        Args:
            point_coords (torch.Tensor): A tensor of shape (N, 2)
                representing the input points.
            point_labels (torch.Tensor): A tensor of shape (N, D)
                representing the embeddings.

        Returns:
            torch.Tensor: A tensor of shape (N, 2 + D) representing the embedded points.
        """
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.model.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        """Embeds the given input mask using the given has_mask_input tensor.

        Args:
            input_mask (torch.Tensor): A tensor of shape (N, H, W)
                representing the input mask.
            has_mask_input (torch.Tensor): A tensor of shape (N,)
                representing whether each input has a mask.

        Returns:
            torch.Tensor: A tensor of shape (N, D, H, W)
                representing the embedded masks.
        """
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def mask_postprocessing(
        self, masks: torch.Tensor, orig_im_size: torch.Tensor
    ) -> torch.Tensor:
        """Postprocesses the given masks tensor by resizing it and.

            cropping it to the original image size.

        Args:
            masks (torch.Tensor): A tensor of shape (N, C, H, W)
                representing the masks.
            orig_im_size (torch.Tensor): A tensor of shape (N, 2)
                representing the original image size (height, width).

        Returns:
            torch.Tensor: A tensor of shape (N, C, H', W')
                representing the postprocessed masks.
        """
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size)
        masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]

        orig_im_size = orig_im_size.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks

    def select_masks(
        self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if we should return the multi click.

        mask or not from the number of points.
        The reweighting is used to avoid control flow.
        """
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the forward pass of the SamOnnxModel.

        Args:
            image_embeddings (torch.Tensor): A tensor of shape (N, D)
                representing the image embeddings.
            point_coords (torch.Tensor): A tensor of shape (N, P, 2)
                representing the point coordinates.
            point_labels (torch.Tensor): A tensor of shape (N, P)
                representing the point labels.
            mask_input (torch.Tensor): A tensor of shape (N, H, W)
                representing the input mask.
            has_mask_input (torch.Tensor): A tensor of shape (N,)
                representing whether each input has a mask.
            orig_im_size (torch.Tensor): A tensor of shape (N, 2)
                representing the original image size (height, width).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing
                the predicted masks (shape: (N, 1, H', W'))
                and the corresponding scores (shape: (N, 1)).
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold, self.stability_score_offset
            )
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks
