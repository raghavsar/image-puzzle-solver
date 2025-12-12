import argparse
from pathlib import Path
from typing import Iterable

import cv2

from mosaic_engine.animation import emit_animation
from mosaic_engine.ingest import load_frame
from mosaic_engine.pipeline import solve_and_render
from mosaic_engine.segmentation import extract_fragments


def _iter_inputs(names: Iterable[str], root: Path) -> Iterable[Path]:
    for name in names:
        provided = Path(name)
        candidate = provided

        if not provided.is_absolute():
            if not provided.exists():
                candidate = root / provided

        if not candidate.exists():
            raise FileNotFoundError(f"Input not found: {candidate}")
        yield candidate


def run_batch(
    inputs: Iterable[str],
    input_root: str | Path = "inputs",
    output_root: str | Path = "outputs",
    frames: int = 30,
) -> None:
    in_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(_iter_inputs(inputs, in_root)):
        print(f"Processing {path.name} ...")
        frame = load_frame(path)

        fragments = extract_fragments(frame)
        print(f"  captured {len(fragments)} pieces")

        assembled, canvas = solve_and_render(fragments, frame)

        still_name = f"{path.stem}_solution_{index}.png"
        still_path = out_root / still_name
        cv2.imwrite(str(still_path), canvas)

        video_name = f"{path.stem}_solution_{index}.mp4"
        video_path = out_root / video_name
        emit_animation(assembled, frame, frame_count=frames, output_filename=str(video_path))

        print(f"  saved grid to {still_path}")
        print(f"  animation saved to {video_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble image puzzle pieces and emit an MP4 animation.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="File names or paths to process (if relative, resolved under --input-root).",
    )
    parser.add_argument("--input-root", default="inputs", help="Folder containing the inputs when using relative names.")
    parser.add_argument("--output-root", default="outputs", help="Directory where MP4 files will be written.")
    parser.add_argument("--frames", type=int, default=30, help="How many frames to render in the output video.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_batch(args.inputs, input_root=args.input_root, output_root=args.output_root, frames=args.frames)
