# Image Puzzle Solver Toolkit

This codebase ingests a fragmented image, extracts individual tiles, lines them up with a greedy edge matcher, and renders both a static solution and an optional animation.

## Layout
| Path | Role |
| --- | --- |
| `puzzle_runner.py` | Batch entry point that loads requested inputs, runs the solver, and writes both a solved grid image and an animation to disk. |
| `mosaic_engine/ingest.py` | Reads PNG or raw RGB inputs and returns a BGR frame. |
| `mosaic_engine/segmentation.py` | Locates quadrilateral pieces, rectifies their perspective, and captures edge pixels. |
| `mosaic_engine/matching.py` | Extracts edge vectors and scores complementary edges with a mean-squared-error similarity. |
| `mosaic_engine/placement.py` | Greedy placement logic that chooses a seed pair, positions the rest, and computes pixel anchors for animation. |
| `mosaic_engine/render.py` | Builds a simple canvas showing the solved grid. |
| `mosaic_engine/animation.py` | Interpolates motion from detected positions to the solved layout and emits an MP4. |
| `mosaic_engine/pipeline.py` | Glue that wires matching, placement, and rendering together. |

## Ownership
| Contributor | Responsibility |
| --- | --- |
| `Raghav` |  Image ingestion, piece segmentation, and grid assembly pipeline. |
| `Chinmay` | Edge matching, placement heuristics, rendering, and animation output. |

## Running it
1. Place inputs in `inputs/` (or point `--input-root` elsewhere).
2. Choose the files you want processed: `python puzzle_runner.py test_regular_rotate.png test_regular_translate.png`
   - Use absolute paths or names relative to `--input-root`.
   - Control frame count with `--frames`.
3. Outputs only: solved PNG and MP4 are written to `outputs/` (or your `--output-root`); no UI windows open.
