"""
mesh_postprocess.py
Layer 3 — Mesh Post-Processor

Cleans and validates generated meshes before evaluation.
Applies geometric corrections to maximise KCS-DT Stage B scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MeshPostProcessor:
    """
    Post-processes a raw TripoSR mesh for DT deployment readiness.

    Operations
    ----------
    1. Remove degenerate faces (zero area, self-intersecting)
    2. Fill small holes (open surfaces fail simulation import)
    3. Normalise scale to IFC reference dimensions
    4. Validate watertightness (closed mesh required for BIM export)
    """

    def __init__(self, config: dict):
        pc = config.get("postprocess", {})
        self.remove_degenerate = pc.get("remove_degenerate_faces", True)
        self.fill_holes = pc.get("fill_holes", True)
        self.normalize_scale = pc.get("normalize_scale", True)
        self.output_dir = Path(config.get("output_dir", "outputs/meshes"))

    def process(
        self,
        mesh_path: Path,
        target_dimensions: Optional[dict] = None,
        prompt_id: str = "",
    ) -> dict:
        """
        Post-process a mesh file in-place and return a quality report.

        Parameters
        ----------
        mesh_path         : Path  Input .obj mesh file
        target_dimensions : dict  Expected dimensions from scene spec {"x": m, "y": m, "z": m}
        prompt_id         : str   DTAH-Bench prompt ID

        Returns
        -------
        dict  Quality report with mesh statistics and pass/fail flags
        """
        try:
            import trimesh  # type: ignore
        except ImportError as e:
            raise ImportError("trimesh required: pip install trimesh") from e

        logger.info("Post-processing mesh: %s", mesh_path)
        mesh = trimesh.load(str(mesh_path), force="mesh")

        report = {
            "prompt_id": prompt_id,
            "input_faces": len(mesh.faces),
            "input_vertices": len(mesh.vertices),
            "is_watertight_before": mesh.is_watertight,
            "is_volume_before": mesh.is_volume,
        }

        if self.remove_degenerate:
            mesh = self._remove_degenerate_faces(mesh)

        if self.fill_holes:
            mesh = self._fill_holes(mesh)

        if self.normalize_scale and target_dimensions:
            mesh = self._normalize_scale(mesh, target_dimensions)

        # Save processed mesh
        out_path = mesh_path.parent / f"{mesh_path.stem}_processed.obj"
        mesh.export(str(out_path))

        report.update({
            "output_faces": len(mesh.faces),
            "output_vertices": len(mesh.vertices),
            "is_watertight_after": mesh.is_watertight,
            "is_volume_after": mesh.is_volume,
            "output_path": str(out_path),
            "usable": mesh.is_watertight and len(mesh.faces) > 0,
        })

        logger.info(
            "[%s] Mesh: %d->%d faces, watertight=%s",
            prompt_id, report["input_faces"],
            report["output_faces"], report["is_watertight_after"]
        )
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_degenerate_faces(mesh):
        import trimesh
        mask = mesh.area_faces > 1e-10
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
        return mesh

    @staticmethod
    def _fill_holes(mesh):
        try:
            import trimesh
            trimesh.repair.fill_holes(mesh)
        except Exception as e:
            logger.warning("Hole filling failed: %s", e)
        return mesh

    @staticmethod
    def _normalize_scale(mesh, target_dims: dict):
        current_extents = mesh.bounding_box.extents
        target = np.array([
            target_dims.get("x", 1.0),
            target_dims.get("y", 1.0),
            target_dims.get("z", 1.0),
        ])
        if np.any(current_extents == 0):
            return mesh
        scale = target / current_extents
        mesh.apply_scale(scale)
        return mesh
