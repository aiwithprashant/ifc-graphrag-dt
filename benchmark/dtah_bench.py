"""
dtah_bench.py
DTAH-Bench DataLoader

Loads DTAH-Bench prompts and ground-truth annotations by tier.
Supports both pilot mode (DTAH-Bench-50) and full mode (150 prompts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

TIER_FILES = {
    1: "tier1_asset.json",
    2: "tier2_assembly.json",
    3: "tier3_system.json",
}

# DTAH-Bench-50 pilot distribution
PILOT_COUNTS = {1: 15, 2: 15, 3: 20}


class DTAHBench:
    """
    DTAH-Bench DataLoader.

    Parameters
    ----------
    prompts_dir      : str | Path  Path to benchmark/prompts/
    ground_truth_dir : str | Path  Path to benchmark/ground_truth/
    pilot_mode       : bool        If True, load only DTAH-Bench-50 pilot subset
    """

    def __init__(
        self,
        prompts_dir: str | Path = "benchmark/prompts",
        ground_truth_dir: str | Path = "benchmark/ground_truth",
        pilot_mode: bool = False,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.pilot_mode = pilot_mode
        self._cache: dict[int, list[dict]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def load_tier(self, tier: int) -> list[dict]:
        """Load all prompts for a given tier (1, 2, or 3)."""
        if tier not in TIER_FILES:
            raise ValueError(f"Tier must be 1, 2, or 3 (got {tier})")
        if tier in self._cache:
            return self._cache[tier]

        path = self.prompts_dir / TIER_FILES[tier]
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        prompts = data["prompts"]

        if self.pilot_mode:
            n = PILOT_COUNTS[tier]
            prompts = prompts[:n]
            logger.info("Pilot mode: loaded %d/%d Tier %d prompts", n, len(data["prompts"]), tier)
        else:
            logger.info("Loaded %d Tier %d prompts", len(prompts), tier)

        # Attach tier metadata to each prompt
        for p in prompts:
            p["tier"] = tier
            p["ifc_relations"] = data.get("ifc_relations", [])

        self._cache[tier] = prompts
        return prompts

    def load_all(self) -> list[dict]:
        """Load all prompts across all tiers."""
        all_prompts = []
        for tier in [1, 2, 3]:
            all_prompts.extend(self.load_tier(tier))
        logger.info("Total prompts loaded: %d", len(all_prompts))
        return all_prompts

    def load_pilot(self) -> list[dict]:
        """Load DTAH-Bench-50 pilot set (15+15+20)."""
        original_mode = self.pilot_mode
        self.pilot_mode = True
        self._cache.clear()
        prompts = self.load_all()
        self.pilot_mode = original_mode
        return prompts

    def load_ground_truth(self, prompt_id: str) -> Optional[dict]:
        """Load ground-truth annotation for a given prompt ID."""
        tier = self._tier_from_id(prompt_id)
        if tier is None:
            logger.warning("Cannot determine tier from prompt ID: %s", prompt_id)
            return None

        paths_to_try = [
            self.ground_truth_dir / "ifc_subgraphs" / f"{prompt_id}_subgraph.json",
            self.ground_truth_dir / "scene_specs"   / f"{prompt_id}_gt_spec.json",
            self.ground_truth_dir / "annotations"   / f"{prompt_id}_annotation.json",
        ]
        for path in paths_to_try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)

        logger.warning("No ground truth found for: %s", prompt_id)
        return None

    def iter_with_ground_truth(self, tiers: Optional[list[int]] = None) -> Iterator[tuple[dict, Optional[dict]]]:
        """Iterate over (prompt, ground_truth) pairs."""
        tiers = tiers or [1, 2, 3]
        for tier in tiers:
            for prompt in self.load_tier(tier):
                gt = self.load_ground_truth(prompt["id"])
                yield prompt, gt

    def stats(self) -> dict:
        """Return benchmark statistics."""
        total = sum(len(self.load_tier(t)) for t in [1, 2, 3])
        return {
            "pilot_mode":    self.pilot_mode,
            "tier1_count":   len(self.load_tier(1)),
            "tier2_count":   len(self.load_tier(2)),
            "tier3_count":   len(self.load_tier(3)),
            "total":         total,
            "domains":       ["MEP", "structural", "HVAC"],
        }

    @staticmethod
    def _tier_from_id(prompt_id: str) -> Optional[int]:
        if prompt_id.startswith("T1-"):
            return 1
        if prompt_id.startswith("T2-"):
            return 2
        if prompt_id.startswith("T3-"):
            return 3
        return None
