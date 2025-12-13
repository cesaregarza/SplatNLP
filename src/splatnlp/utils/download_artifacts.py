from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


DEFAULT_BASE_URL = "https://splat-nlp.nyc3.cdn.digitaloceanspaces.com"
DEFAULT_DATASET_DIR = "dataset_v2"


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    required: bool = True


CORE_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("model.pth"),
    ArtifactSpec("model_info.json"),
    ArtifactSpec("model_params.json"),
    ArtifactSpec("vocab.json"),
    ArtifactSpec("weapon_vocab.json"),
)

ULTRA_ARTIFACTS: tuple[ArtifactSpec, ...] = (ArtifactSpec("model_ultra.pth"),)

ULTRA_SAE_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("sae_config_ultra.json"),
    ArtifactSpec("sae_model_ultra.pth"),
)

ULTRA_LABELS_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("feature_labels_ultra.json", required=False),
    ArtifactSpec("consolidated_ultra.json", required=False),
)

OPTIONAL_DATA_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("tokenized_data.csv", required=False),
)


def _join_url(*parts: str) -> str:
    stripped = [p.strip("/") for p in parts if p]
    return "/".join(stripped)


def _resolve_defaults(args: argparse.Namespace) -> None:
    if args.base_url is None:
        args.base_url = os.getenv("DO_SPACES_ML_ENDPOINT") or DEFAULT_BASE_URL
    if args.dataset_dir is None:
        args.dataset_dir = os.getenv("DO_SPACES_ML_DIR") or DEFAULT_DATASET_DIR
    if args.out_dir is None:
        args.out_dir = str(Path("saved_models") / args.dataset_dir)


def _iter_selected_artifacts(args: argparse.Namespace) -> Iterable[ArtifactSpec]:
    selected: list[ArtifactSpec] = list(CORE_ARTIFACTS)

    if args.include_ultra or args.include_ultra_sae:
        selected.extend(ULTRA_ARTIFACTS)

    if args.include_ultra_sae:
        selected.extend(ULTRA_SAE_ARTIFACTS)

    if args.include_ultra_labels or args.include_ultra_sae:
        selected.extend(ULTRA_LABELS_ARTIFACTS)

    if args.include_tokenized_data:
        selected.extend(OPTIONAL_DATA_ARTIFACTS)

    if args.files:
        selected = [ArtifactSpec(name, required=True) for name in args.files]

    return selected


def _download_one(
    *,
    url: str,
    dest: Path,
    force: bool,
    timeout_s: int,
    quiet: bool,
    chunk_size: int = 1024 * 1024,
) -> None:
    if dest.exists() and not force:
        if not quiet:
            print(f"skip  {dest} (exists)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    if not quiet:
        print(f"get   {url}")

    response = requests.get(url, stream=True, timeout=timeout_s)
    try:
        response.raise_for_status()
        total_raw = response.headers.get("Content-Length")
        total = int(total_raw) if total_raw and total_raw.isdigit() else None

        bar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            disable=quiet,
        )
        try:
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()
    finally:
        response.close()

    tmp_path.replace(dest)


def download_artifacts(
    *,
    base_url: str,
    dataset_dir: str,
    out_dir: Path,
    artifacts: Iterable[ArtifactSpec],
    force: bool,
    timeout_s: int,
    quiet: bool,
    dry_run: bool,
) -> None:
    base_path = _join_url(base_url, dataset_dir)

    missing_required: list[str] = []
    for spec in artifacts:
        url = _join_url(base_path, spec.filename)
        dest = out_dir / spec.filename
        if dry_run:
            print(f"would get {url} -> {dest}")
            continue

        try:
            _download_one(
                url=url,
                dest=dest,
                force=force,
                timeout_s=timeout_s,
                quiet=quiet,
            )
        except requests.HTTPError:
            if spec.required:
                missing_required.append(spec.filename)
            elif not quiet:
                print(f"warn  missing optional {spec.filename}")

    if missing_required:
        joined = ", ".join(missing_required)
        raise SystemExit(f"missing required artifacts: {joined}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download pretrained SplatNLP artifacts from DO Spaces.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL (default: $DO_SPACES_ML_ENDPOINT or "
            f"{DEFAULT_BASE_URL})"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Dataset directory under base URL (default: $DO_SPACES_ML_DIR or "
            f"{DEFAULT_DATASET_DIR})"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: saved_models/<dataset-dir>)",
    )
    parser.add_argument(
        "--include-ultra",
        action="store_true",
        help="Also download model_ultra.pth",
    )
    parser.add_argument(
        "--include-ultra-sae",
        action="store_true",
        help="Also download ultra SAE artifacts (includes ultra model).",
    )
    parser.add_argument(
        "--include-tokenized-data",
        action="store_true",
        help="Also download tokenized_data.csv (optional).",
    )
    parser.add_argument(
        "--include-ultra-labels",
        action="store_true",
        help=(
            "Also download Ultra feature labels if available "
            "(feature_labels_ultra.json / consolidated_ultra.json)."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Download only these filenames (overrides other selections).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output (progress bars disabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without downloading.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _resolve_defaults(args)

    artifacts = list(_iter_selected_artifacts(args))
    out_dir = Path(args.out_dir)

    download_artifacts(
        base_url=str(args.base_url),
        dataset_dir=str(args.dataset_dir),
        out_dir=out_dir,
        artifacts=artifacts,
        force=bool(args.force),
        timeout_s=int(args.timeout_s),
        quiet=bool(args.quiet),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
