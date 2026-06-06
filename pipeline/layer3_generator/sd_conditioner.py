"""
sd_conditioner.py
Layer 3 — Stable Diffusion Conditioner

Takes a scene specification and generates a conditioned image
using Stable Diffusion XL. The image is then passed to TripoSR
for 3D mesh reconstruction.

Conditioning strategy:
  - Constructs a detailed text prompt from the scene spec
  - Applies negative prompting to avoid common failure modes
  - Generates multiple views (front, side, isometric) for TripoSR
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, illustration, painting, sketch, "
    "deformed, disconnected components, floating parts, unrealistic scale, "
    "missing connections, incomplete, watermark, text"
)

VIEW_PROMPTS = {
    "front":      "front view, elevation",
    "side":       "side view, elevation",
    "isometric":  "isometric view, 3/4 perspective",
    "top":        "top view, plan view",
}


class SDConditioner:
    """
    Generates conditioned images from a scene specification using SDXL.

    Parameters
    ----------
    config : dict
        generator section from pipeline_config.yaml
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("sd_model", "stabilityai/stable-diffusion-xl-base-1.0")
        self.device = config.get("device", "cpu")
        self.num_steps = config.get("num_inference_steps", 50)
        self.guidance = config.get("guidance_scale", 7.5)
        self.num_views = config.get("num_views", 4)
        self.output_dir = Path(config.get("output_dir", "outputs/meshes"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_views(
        self,
        spec: dict,
        prompt_id: str = "",
        seed: int = 42,
    ) -> list[Path]:
        """
        Generate multi-view images for a scene specification.

        Parameters
        ----------
        spec      : dict   Scene specification from Layer 2
        prompt_id : str    DTAH-Bench prompt ID (for output naming)
        seed      : int    Random seed for reproducibility

        Returns
        -------
        list[Path]  Paths to generated image files (one per view)
        """
        self._load_pipeline()
        generation_prompt = spec.get("generation_prompt", "")
        enriched_prompt = self._enrich_prompt(generation_prompt, spec)

        view_keys = list(VIEW_PROMPTS.keys())[:self.num_views]
        image_paths = []

        import torch
        generator = torch.Generator(device=self.device).manual_seed(seed)

        for view_name in view_keys:
            view_suffix = VIEW_PROMPTS[view_name]
            full_prompt = f"{enriched_prompt}, {view_suffix}"

            logger.info("Generating %s view for [%s]...", view_name, prompt_id)
            image = self._pipe(
                prompt=full_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance,
                generator=generator,
            ).images[0]

            fname = f"{prompt_id}_{view_name}.png" if prompt_id else f"{view_name}.png"
            out_path = self.output_dir / fname
            image.save(str(out_path))
            image_paths.append(out_path)
            logger.info("Saved %s view: %s", view_name, out_path)

        return image_paths

    def generate_single(
        self,
        spec: dict,
        prompt_id: str = "",
        view: str = "isometric",
        seed: int = 42,
    ) -> Path:
        """Generate a single view image (faster, for dry runs)."""
        self._load_pipeline()
        generation_prompt = spec.get("generation_prompt", "")
        enriched_prompt = self._enrich_prompt(generation_prompt, spec)
        full_prompt = f"{enriched_prompt}, {VIEW_PROMPTS.get(view, '')}"

        import torch
        generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self._pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance,
            generator=generator,
        ).images[0]

        fname = f"{prompt_id}_{view}.png" if prompt_id else f"{view}.png"
        out_path = self.output_dir / fname
        image.save(str(out_path))
        return out_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enrich_prompt(self, base_prompt: str, spec: dict) -> str:
        """Add technical detail from spec to the SD prompt."""
        entity_types = list({
            e.get("ifc_type", "").replace("Ifc", "")
            for e in spec.get("entities", [])
        })
        materials = list({
            a.get("material", "")
            for a in spec.get("attributes", {}).values()
            if a.get("material")
        })

        enrichment_parts = []
        if entity_types:
            enrichment_parts.append(f"containing {', '.join(entity_types[:5])}")
        if materials:
            enrichment_parts.append(f"made of {', '.join(materials[:3])}")

        enrichment = ", ".join(enrichment_parts)
        style = "photorealistic, technical 3D render, white background, studio lighting, high detail"

        if enrichment:
            return f"{base_prompt}, {enrichment}, {style}"
        return f"{base_prompt}, {style}"

    def _load_pipeline(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline  # type: ignore
        except ImportError as e:
            raise ImportError(
                "diffusers and torch required. Run colab_setup.sh first."
            ) from e

        logger.info("Loading SDXL pipeline: %s on %s", self.model_name, self.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(self.device)
        if self.device == "cuda":
            self._pipe.enable_model_cpu_offload()
        logger.info("SDXL pipeline loaded.")
