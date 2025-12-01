#!/usr/bin/env python3
"""
Image Puzzle Solver - Main Entry Point

A modular Python puzzle solver handling three transformation types:
- (A) Rotation only
- (B) Translation only  
- (C) Rotation + Translation

Uses MST-based reconstruction with MP4 animation output.
"""

import argparse
import sys
from pathlib import Path

from src.utils.io import load_image, save_image
from src.utils.piece import Piece
from src.segmentation.extractor import extract_pieces
from src.solver.grid_solver import solve_grid, compute_grid_dimensions
from src.visualization.animator import create_animation, render_solved_image, AnimationStyle


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve image puzzles with rotation and/or translation transformations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 16 pieces in 4x4 grid (auto-computed)
  python main.py resources/samples/mona_lisa_translate.rgb --n 16

  # 12 pieces in 3x4 grid (explicit)
  python main.py input.png --n 12 --grid-rows 3 --grid-cols 4

  # 9 pieces in 3x3 grid
  python main.py input.rgb --n 9
        """
    )
    
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input puzzle image (.png, .jpg, or .rgb)"
    )
    
    parser.add_argument(
        "--n", "-n",
        type=int,
        required=True,
        help="Expected number of puzzle pieces"
    )
    
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=None,
        help="Number of rows in the grid (auto-computed if not specified)"
    )
    
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=None,
        help="Number of columns in the grid (auto-computed if not specified)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="solution.mp4",
        help="Output animation file path (default: solution.mp4)"
    )
    
    parser.add_argument(
        "--save-solved",
        type=str,
        default=None,
        help="Save the solved puzzle as a static image"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Animation duration in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for animation (default: 30)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Image width for .rgb files (default: 800)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Image height for .rgb files (default: 800)"
    )
    
    parser.add_argument(
        "--animation-style",
        type=str,
        choices=["simultaneous", "sequential", "wave"],
        default="simultaneous",
        help="Animation style: simultaneous (all at once), sequential (one by one), wave (staggered)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the puzzle solver."""
    args = parse_args()
    
    # Validate input file
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine grid dimensions
    if args.grid_rows is not None and args.grid_cols is not None:
        grid_rows = args.grid_rows
        grid_cols = args.grid_cols
        if grid_rows * grid_cols != args.n:
            print(
                f"Error: Grid dimensions ({grid_rows}x{grid_cols}={grid_rows*grid_cols}) "
                f"don't match piece count ({args.n})",
                file=sys.stderr
            )
            sys.exit(1)
    else:
        # Auto-compute grid dimensions
        grid_rows, grid_cols = compute_grid_dimensions(args.n)
        if args.verbose:
            print(f"Auto-computed grid dimensions: {grid_rows}x{grid_cols}")
    
    # Pipeline
    try:
        # Step 1: Load image
        if args.verbose:
            print(f"Loading image: {input_path}")
        image = load_image(input_path, width=args.width, height=args.height)
        canvas_size = (image.shape[1], image.shape[0])
        
        # Step 2: Extract pieces
        if args.verbose:
            print(f"Extracting {args.n} pieces...")
        pieces = extract_pieces(image, expected_count=args.n)
        if args.verbose:
            print(f"Extracted {len(pieces)} pieces")
            for p in pieces:
                print(f"  {p}")
        
        # Step 3: Solve puzzle
        if args.verbose:
            print(f"Solving puzzle ({grid_rows}x{grid_cols} grid)...")
        solved_pieces = solve_grid(pieces, grid_rows, grid_cols, verbose=args.verbose)
        if args.verbose:
            print("\nPuzzle solved!")
            for p in solved_pieces:
                print(f"  Piece {p.id}: {p.initial_center} -> {p.solved_center}, "
                      f"rot: {p.initial_rotation} -> {p.solved_rotation}")
        
        # Step 4: Create animation
        if args.verbose:
            print(f"Creating animation: {args.output}")
        
        # Map animation style string to enum
        style_map = {
            "simultaneous": AnimationStyle.SIMULTANEOUS,
            "sequential": AnimationStyle.SEQUENTIAL,
            "wave": AnimationStyle.WAVE
        }
        animation_style = style_map.get(args.animation_style, AnimationStyle.SIMULTANEOUS)
        
        output_path = create_animation(
            solved_pieces,
            output_path=args.output,
            canvas_size=canvas_size,
            fps=args.fps,
            duration=args.duration,
            style=animation_style
        )
        print(f"Animation saved: {output_path}")
        
        # Step 5: Save solved image if requested
        if args.save_solved:
            if args.verbose:
                print(f"Rendering solved image: {args.save_solved}")
            solved_image = render_solved_image(solved_pieces, canvas_size)
            save_image(solved_image, args.save_solved)
            print(f"Solved image saved: {args.save_solved}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
