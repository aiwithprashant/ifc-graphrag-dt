"""
triposr_runner.py
Layer 3 — TripoSR 3D Reconstruction Runner

Takes multi-view images from SDConditioner and reconstructs
a 3D mesh using TripoSR (VAST-AI-Research/TripoSR).

TripoSR uses a transformer-based triplane encoding to produce
a 3D mesh from a single or multi-view image input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TripoSRRunner:
    """
    Runs TripoSR 3D reconstruction from generated images.

    Parameters
    ----------
    config : dict
        generator section from pipeline_config.yaml
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("triposr_model", "stabilityai/TripoSR")
        self.device = config.get("device", "cpu")
        self.resolution = config.get("mesh_resolution", 256)
        self.output_dir = Path(config.get("output_dir", "outputs/meshes"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reconstruct(
        self,
        image_paths: list[Path],
        prompt_id: str = "",
    ) -> Path:
        """
        Reconstruct a 3D mesh from one or more view images.

        Parameters
        ----------
        image_paths : list[Path]  Paths to input images (from SDConditioner)
        prompt_id   : str         DTAH-Bench prompt ID

        Returns
        -------
        Path  Path to the output .obj mesh file
        """
        self._load_model()

        from PIL import Image  # type: ignore
        import torch

        # Use the first image (isometric/front) as primary input
        primary_image = Image.open(image_paths[0]).convert("RGB")

        logger.info("Running TripoSR reconstruction for [%s]...", prompt_id)

        with torch.no_grad():
            scene_codes = self._model([primary_image], device=self.device)

        meshes = self._model.extract_mesh(
            scene_codes,
            resolution=self.resolution,
        )

        fname = f"{prompt_id}.obj" if prompt_id else "output.obj"
        out_path = self.output_dir / fname
        meshes[0].export(str(out_path))
        logger.info("Mesh saved: %s", out_path)
        return out_path

    def reconstruct_batch(
        self,
        image_path_lists: list[list[Path]],
        prompt_ids: Optional[list[str]] = None,
    ) -> list[Path]:
        """Reconstruct meshes for a batch of image sets."""
        ids = prompt_ids or [""] * len(image_path_lists)
        return [
            self.reconstruct(paths, pid)
            for paths, pid in zip(image_path_lists, ids)
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from tsr.system import TSR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "TripoSR not installed. Run: pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
            ) from e

        logger.info("Loading TripoSR model: %s on %s", self.model_name, self.device)
        self._model = TSR.from_pretrained(
            self.model_name,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model = self._model.to(self.device)
        self._model.renderer.set_chunk_size(8192)
        logger.info("TripoSR model loaded.")
