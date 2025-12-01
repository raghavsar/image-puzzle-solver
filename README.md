# CSCI 576 - Image Puzzle Solver

A computational image puzzle solver that reconstructs scrambled images composed of N pieces that have undergone rigid transformations (translation and/or rotation).

## Team Members

- [Student Name 1]
- [Student Name 2]  
- [Student Name 3]

**Demonstration Dates:** December 10-12, 2025

## Overview

This solver implements a five-phase pipeline to solve image puzzles:

1. **Image Analysis & Preprocessing** - Segment individual pieces from the input canvas
2. **Feature Extraction** - Extract color histograms, gradient histograms, and texture features
3. **Piece Matching & Search** - MST-based search to find optimal piece adjacencies  
4. **Assembly & Transformation** - Compute final positions and rotations
5. **Visualization & Animation** - Generate MP4 animation showing the solution process

### Supported Input Formats
- `.rgb` - Raw RGB files (800x800 default, configurable)
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` - Standard image formats

### Transformation Types Handled
- **(A) Rotation only** - Pieces rotated but not translated
- **(B) Translation only** - Pieces translated but not rotated
- **(C) Rotation + Translation** - Both transformations applied

## Installation

```bash
# Clone the repository
git clone https://github.com/raghavsar/image-puzzle-solver.git
cd image-puzzle-solver

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- SciPy

## Usage

### Basic Usage

```bash
# Solve a 16-piece puzzle (4x4 grid, auto-computed)
python main.py resources/samples/mona_lisa_translate.rgb --n 16

# Solve with explicit grid dimensions
python main.py resources/samples/mona_lisa_rotate.rgb --n 12 --grid-rows 3 --grid-cols 4

# Save both animation and solved image
python main.py input.rgb --n 9 --output solution.mp4 --save-solved solved.png
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_image` | Path to input puzzle image (.rgb, .png, etc.) | Required |
| `--n`, `-n` | Expected number of puzzle pieces | Required |
| `--grid-rows` | Number of rows in the grid | Auto-computed |
| `--grid-cols` | Number of columns in the grid | Auto-computed |
| `--output`, `-o` | Output animation file path | `solution.mp4` |
| `--save-solved` | Save solved puzzle as static image | None |
| `--duration` | Animation duration in seconds | 5.0 |
| `--fps` | Frames per second | 30 |
| `--width` | Image width for .rgb files | 800 |
| `--height` | Image height for .rgb files | 800 |
| `--verbose`, `-v` | Enable verbose output | False |

### Examples

```bash
# Translate-only puzzle (Example 1)
python main.py resources/samples/mona_lisa_translate.rgb --n 16 -v

# Rotate + Translate puzzle (Example 2)
python main.py resources/samples/mona_lisa_rotate.rgb --n 16 -v

# Starry Night samples
python main.py resources/samples/starry_night_translate.rgb --n 16
python main.py resources/samples/starry_night_rotate.rgb --n 16
```

## Algorithm Details

### Phase 1: Image Analysis & Preprocessing
- **Piece Segmentation**: Uses thresholding and contour detection (`cv2.findContours`) to isolate pieces from black background
- **Rotation Detection**: Estimates piece orientation using minimum area rectangle (`cv2.minAreaRect`), snapping to 0°, 90°, 180°, or 270°

### Phase 2: Feature Extraction
The solver extracts multiple feature types for robust edge matching:

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| **Color Histogram** | 96 (32 bins × 3 channels) | Per-channel BGR histogram for edge pixels |
| **Gradient Histogram** | 16 bins | Orientation histogram weighted by gradient magnitude |
| **Texture Features** | 9 | Mean, std, and gradient statistics per channel |

### Phase 3: Piece Matching & Search (MST-Based)
- **Compatibility Score**: Chi-squared distance between feature vectors
- **Search Strategy**: Build complete graph → compute Minimum Spanning Tree (MST)
- **Rotation Handling**: Tests all 16 rotation combinations (4 × 4) per edge pair

### Phase 4: Assembly & Transformation
- **Grid Assignment**: BFS traversal from MST root to assign (row, col) positions
- **Transformation Computation**: Each piece receives `solved_center` and `solved_rotation`

### Phase 5: Visualization
- **Animation**: Smooth ease-in-out interpolation from initial to solved positions
- **Output Formats**: MP4 (with FFmpeg) or GIF fallback

## Algorithmic Complexity

| Phase | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Piece Extraction | O(W × H) | O(N × P²) |
| Feature Extraction | O(N × P) | O(N × F) |
| Pairwise Matching | O(N² × 16 × P) | O(N²) |
| MST Construction | O(N² log N) | O(N²) |
| Grid Assignment | O(N) | O(N) |
| Animation | O(F × N) | O(N × P²) |

Where:
- N = number of pieces
- P = piece dimension (e.g., 100 pixels)
- F = feature vector length (~121)
- W, H = canvas dimensions

**Overall Complexity**: O(N² × P) dominated by pairwise matching

## Project Structure

```
image-puzzle-solver/
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── src/
    ├── segmentation/
    │   └── extractor.py      # Piece extraction from images
    ├── solver/
    │   ├── edge_matcher.py   # Edge matching (SSD + histogram)
    │   ├── features.py       # Feature extraction module
    │   └── grid_solver.py    # MST-based grid reconstruction
    ├── visualization/
    │   └── animator.py       # MP4 animation generation
    └── utils/
        ├── piece.py          # Piece dataclass
        └── io.py             # RGB/PNG image loading
```

## Output

The solver produces:
1. **Animated Solution** (`solution.mp4`): Shows pieces moving from scattered positions to solved grid
2. **Solved Image** (optional): Static reconstruction of the completed puzzle

## License

See [LICENSE](LICENSE) file.