#!/usr/bin/env python3
"""Run the released SAM 3D Objects pipeline for the CA-1M context.

This runner decodes only the mesh. CA-1M already provides an aligned depth
map, so it is used as SAM 3D's supported point-map input instead of keeping
the MoGe depth network resident beside the two shape generators. This makes
the released model fit on a 24 GiB card without changing the generators or
their sampling schedules.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import trimesh
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image


def ca1m_pointmap(depth_path: Path, context_path: Path, view_index: int) -> torch.Tensor:
    """Back-project registered CA-1M depth into SAM 3D's PyTorch3D camera frame."""
    context = json.loads(context_path.read_text())
    view = context["views"][view_index]
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
    width, height = Image.open(context_path.parent / view["rgb_path"]).size
    if depth.shape != (height, width):
        depth = np.asarray(
            Image.fromarray(depth).resize((width, height), Image.Resampling.NEAREST),
            dtype=np.float32,
        )

    intrinsic = np.asarray(view["rgb_intrinsic"], dtype=np.float32)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    z = depth
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    # OpenCV (right, down, forward) -> PyTorch3D (left, up, forward).
    points = np.stack((-x, -y, z), axis=-1)
    points[~np.isfinite(z) | (z <= 0)] = np.nan
    return torch.from_numpy(points)


def export_mesh(mesh_result, output: Path) -> None:
    vertices = mesh_result.vertices.detach().float().cpu().numpy()
    faces = mesh_result.faces.detach().long().cpu().numpy()
    colors = None
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.detach().float().cpu().numpy()
        if attrs.ndim == 2 and attrs.shape[1] >= 3:
            rgb = np.clip(attrs[:, :3], 0.0, 1.0)
            colors = np.concatenate((rgb, np.ones((len(rgb), 1))), axis=1)
            colors = np.round(colors * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("depth", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    sys.path.insert(0, str(args.repo))
    project_root = Path(__file__).resolve().parents[1]
    pytorch3d_sources = sorted((project_root / "data" / "vendor").glob("pytorch3d-75ebeea*"))
    pytorch3d_sources = [path for path in pytorch3d_sources if path.is_dir()]
    if not pytorch3d_sources:
        raise FileNotFoundError("pinned PyTorch3D source tree is missing from data/vendor")
    sys.path.insert(0, str(pytorch3d_sources[0]))

    # model/io.py uses Lightning only for legacy checkpoint type checks. The
    # released inference checkpoints are plain torch modules, so provide the
    # tiny interface needed by that loader instead of installing the trainer.
    lightning = types.ModuleType("lightning")
    lightning_pytorch = types.ModuleType("lightning.pytorch")

    class _UnusedLightningModule(torch.nn.Module):
        pass

    lightning_pytorch.LightningModule = _UnusedLightningModule
    lightning.pytorch = lightning_pytorch
    lightning_utilities = types.ModuleType("lightning.pytorch.utilities")
    lightning_consolidate = types.ModuleType(
        "lightning.pytorch.utilities.consolidate_checkpoint"
    )
    lightning_consolidate._format_checkpoint = _unused_lightning = lambda value: value
    lightning_consolidate._load_distributed_checkpoint = lambda *_args, **_kwargs: (
        _unused_lightning({})
    )
    sys.modules.setdefault("lightning", lightning)
    sys.modules.setdefault("lightning.pytorch", lightning_pytorch)
    sys.modules.setdefault("lightning.pytorch.utilities", lightning_utilities)
    sys.modules.setdefault(
        "lightning.pytorch.utilities.consolidate_checkpoint", lightning_consolidate
    )

    # FlexiCubes uses only Kaolin's small tensor-shape assertion helper.
    kaolin = types.ModuleType("kaolin")
    kaolin_utils = types.ModuleType("kaolin.utils")
    kaolin_testing = types.ModuleType("kaolin.utils.testing")

    def _check_tensor(value, shape, throw=True):
        valid = value.ndim == len(shape) and all(
            expected is None or actual == expected
            for actual, expected in zip(value.shape, shape)
        )
        if throw and not valid:
            raise ValueError(f"expected tensor shape {shape}, got {tuple(value.shape)}")
        return valid

    kaolin_testing.check_tensor = _check_tensor
    kaolin_utils.testing = kaolin_testing
    kaolin.utils = kaolin_utils
    sys.modules.setdefault("kaolin", kaolin)
    sys.modules.setdefault("kaolin.utils", kaolin_utils)
    sys.modules.setdefault("kaolin.utils.testing", kaolin_testing)

    # Both released generator checkpoints contain their complete DINO
    # condition-embedder weights. Avoid fetching the same 1.13 GiB public
    # pretraining checkpoint only to overwrite it moments later.
    torch_hub_load = torch.hub.load

    def _load_backbone_from_checkpoint(repo_or_dir, model, *load_args, **load_kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            load_kwargs["pretrained"] = False
        return torch_hub_load(repo_or_dir, model, *load_args, **load_kwargs)

    torch.hub.load = _load_backbone_from_checkpoint

    # The point-map pipeline imports one camera helper through the renderer
    # package, whose package initializer otherwise loads compiled rasterizers.
    # This is the exact fixed OpenCV-to-PyTorch3D view used upstream.
    p3d_renderer = types.ModuleType("pytorch3d.renderer")

    def _look_at_view_transform(*, eye, at, up, device="cpu", **_kwargs):
        count = len(eye)
        rotation = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
        rotation = rotation.unsqueeze(0).repeat(count, 1, 1)
        translation = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(count, 1)
        return rotation, translation

    p3d_renderer.look_at_view_transform = _look_at_view_transform
    sys.modules.setdefault("pytorch3d.renderer", p3d_renderer)

    # Mesh postprocessing and layout optimization are imported eagerly by the
    # upstream package. They are not part of this mesh-only run, so keep those
    # optional stacks (Open3D, PyVista, gsplat) out of the environment.
    postprocess_name = (
        "sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils"
    )
    sys.modules.setdefault(postprocess_name, types.ModuleType(postprocess_name))
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))

    layout_name = "sam3d_objects.pipeline.layout_post_optimization_utils"
    layout = types.ModuleType(layout_name)

    def _unused_optional(*_args, **_kwargs):
        raise RuntimeError("optional layout/renderer code was called in a mesh-only run")

    for name in (
        "run_ICP compute_iou set_seed apply_transform get_mesh get_mask_renderer "
        "run_alignment run_render_compare check_occlusion get_gs_transformed "
        "get_gs_mask_renderer run_gs_alignment run_gs_ICP apply_gs_transform "
        "run_gs_render_compare_rgb_mask compute_iou_gs "
        "copy_and_update_gaussian_positions "
        "apply_icp_transformation_to_gaussian prepare_rgb_for_supervision "
        "flip_coords_pytorch3d_to_opencv safe_copy_gaussian "
        "get_mask_colors_for_gs"
    ).split():
        setattr(layout, name, _unused_optional)
    sys.modules.setdefault(layout_name, layout)

    # Retain an explicit gsplat guard in case upstream changes its import path.
    if "gsplat" not in sys.modules:
        gsplat = types.ModuleType("gsplat")
        gsplat.rasterization = _unused_optional
        sys.modules["gsplat"] = gsplat

    config = OmegaConf.load(args.checkpoint_dir / "pipeline.yaml")
    config.workspace_dir = str(args.checkpoint_dir)
    config.compile_model = False
    config.decode_formats = ["mesh"]
    config.depth_model = None
    config.slat_decoder_gs_config_path = None
    config.slat_decoder_gs_ckpt_path = None
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    pipeline = instantiate(config)

    # Export MeshExtractResult directly, avoiding Gaussian-assisted baking.
    pipeline.postprocess_slat_output = lambda outputs, *_args, **_kwargs: outputs
    image = np.asarray(Image.open(args.image).convert("RGB"))
    # Keep alpha in the uint8 [0, 255] range expected by image_to_float.
    mask = np.asarray(Image.open(args.mask).convert("L"))
    pointmap = ca1m_pointmap(args.depth, args.context, args.view_index)
    with torch.inference_mode():
        result = pipeline.run(
            image,
            mask,
            seed=args.seed,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            pointmap=pointmap,
            decode_formats=["mesh"],
        )
    mesh_result = result["mesh"][0]
    if not mesh_result.success:
        raise RuntimeError("SAM 3D returned an empty mesh")
    export_mesh(mesh_result, args.output)
    print(json.dumps({"output": str(args.output), "vertices": len(mesh_result.vertices), "faces": len(mesh_result.faces)}))


if __name__ == "__main__":
    main()
