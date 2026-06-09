"""Download and validate IFC reference assets used by the pipeline."""

from __future__ import annotations

from pathlib import Path


DUPLEX_URLS = [
    (
        "https://github.com/buildingsmart-community/"
        "Community-Sample-Test-Files/raw/refs/heads/main/"
        "IFC%202.3.0.1%20%28IFC%202x3%29/Duplex%20Apartment/"
        "Duplex_A_20110907.ifc"
    ),
    (
        "https://raw.githubusercontent.com/stijngoedertier/"
        "georeference-ifc/master/Duplex_A_20110907.ifc"
    ),
]


def is_valid_ifc(path: str | Path, min_bytes: int = 100_000) -> bool:
    """Return whether a path contains a plausible IFC STEP file."""
    path = Path(path)
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as file:
        return b"ISO-10303-21" in file.read(256)


def ensure_duplex_ifc(
    path: str | Path = "benchmark/ifc_reference_models/duplex.ifc",
) -> Path:
    """Download the Duplex Apartment reference model when it is unavailable."""
    import requests

    path = Path(path)
    if is_valid_ifc(path):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in DUPLEX_URLS:
        try:
            response = requests.get(
                url,
                timeout=120,
                allow_redirects=True,
                headers={"User-Agent": "ifc-graphrag-dt"},
            )
            response.raise_for_status()
            if b"ISO-10303-21" not in response.content[:256]:
                raise ValueError("response is not an IFC STEP file")

            temp_path = path.with_suffix(".tmp")
            temp_path.write_bytes(response.content)
            temp_path.replace(path)
            return path
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError("Unable to download Duplex IFC:\n" + "\n".join(errors))
