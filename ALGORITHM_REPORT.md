# 🧩 Image Puzzle Solver - Algorithm Phase Report

## Executive Summary

The solver implements a **5-phase pipeline** to reconstruct scrambled images composed of N puzzle pieces that have undergone rigid transformations (translation and/or rotation).

---

## Phase 1: Image Loading & Preprocessing

**File:** `src/utils/io.py`

### What it does:
- Loads input images in multiple formats (`.rgb`, `.png`, `.jpg`, `.bmp`)
- For `.rgb` files: reads raw binary data and reshapes to (H, W, 3)
- Converts RGB to BGR for OpenCV compatibility

### Algorithm:
```
1. Detect file extension
2. If standard format → cv2.imread()
3. If .rgb format:
   a. Read raw bytes: np.fromfile()
   b. Reshape to (height, width, 3)
   c. Convert RGB → BGR
```

### Complexity:
- **Time:** O(W × H)
- **Space:** O(W × H × 3)

---

## Phase 2: Piece Segmentation & Extraction

**File:** `src/segmentation/extractor.py`

### What it does:
- Identifies individual puzzle pieces from the black background
- Extracts piece images with masks
- Estimates initial rotation of each piece

### Algorithm:
```
1. Convert image to grayscale
2. Binary threshold (pixel > 10 → foreground)
3. Find external contours using cv2.findContours()
4. Filter contours by minimum area (> 100 pixels)
5. For each valid contour:
   a. Get bounding box (x, y, w, h)
   b. Extract piece image from bounding box
   c. Create mask from contour
   d. Apply mask (set background to black)
   e. Estimate rotation using minAreaRect()
   f. Snap rotation to nearest 90° (0, 90, 180, 270)
6. Sort pieces by position (top-left to bottom-right)
```

### Rotation Estimation:
```python
rect = cv2.minAreaRect(contour)
angle = rect[2]  # Range: [-90, 0)
if angle < -45:
    angle += 90
angle = round(angle / 90) * 90 % 360
```

### Complexity:
- **Time:** O(W × H) for thresholding + O(N × P²) for extraction
- **Space:** O(N × P²) where P = piece dimension

---

## Phase 3: Feature Extraction

**File:** `src/solver/features.py`

### What it does:
Extracts robust features from piece edges for matching. Implements the **histogram-based approach** required by the project specification.

### Feature Types:

#### 3.1 Color Histogram (96 dimensions)
```
- 32 bins per channel (B, G, R)
- L1-normalized per channel
- Captures color distribution along edge
```

#### 3.2 Gradient Histogram (16 dimensions)
```
- Sobel operators for dx, dy gradients
- Compute magnitude and orientation
- Weighted histogram (by magnitude)
- Captures texture/edge patterns
```

#### 3.3 Texture Features (9 dimensions)
```
- Mean per channel (3 values)
- Std deviation per channel (3 values)
- Gradient mean per channel (3 values)
```

### Total Feature Vector: **121 dimensions**

### Algorithm:
```
1. Apply rotation to piece image if needed
2. Extract edge region (1-pixel strip for color, 3-pixel for gradient)
3. Compute color histogram for edge pixels
4. Compute gradient histogram using Sobel
5. Compute texture statistics
6. Concatenate all features
```

### Complexity:
- **Time:** O(P) per edge, O(4P) per piece
- **Space:** O(121) per edge feature vector

---

## Phase 4: Piece Matching & Search (MST-Based)

**Files:** `src/solver/edge_matcher.py`, `src/solver/grid_solver.py`

### What it does:
Finds optimal piece adjacencies by computing compatibility scores between all edge pairs, then uses MST to find the best reconstruction.

### Matching Methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| **SSD** | Sum of Squared Differences on raw pixels | Fast, large puzzles (>50 pieces) |
| **HISTOGRAM** | Chi-squared distance on feature vectors | Accurate, small puzzles (≤50 pieces) |
| **COMBINED** | Weighted blend (30% SSD + 70% Histogram) | Balanced |

### Algorithm:
```
1. For each pair of pieces (i, j):
   a. For each rotation combination (16 total = 4 × 4):
      - Test horizontal adjacency (i.right ↔ j.left)
      - Test vertical adjacency (i.bottom ↔ j.top)
      - Compute feature distance (chi-squared)
   b. Store minimum cost and best rotations
   
2. Build cost matrix (N × N)

3. Compute Minimum Spanning Tree (MST):
   - Using scipy.sparse.csgraph.minimum_spanning_tree()
   - Kruskal's algorithm internally
   
4. MST contains N-1 edges representing best adjacencies
```

### Chi-Squared Distance:
```python
distance = Σ (h1[i] - h2[i])² / (h1[i] + h2[i] + ε)
```

### Complexity:
- **Pairwise Matching:** O(N² × 16 × P) where 16 = rotation combinations
- **MST Construction:** O(N² log N)
- **Total:** O(N² × P) dominated by matching

---

## Phase 5: Grid Assembly & Transformation

**File:** `src/solver/grid_solver.py`

### What it does:
Assigns grid positions to pieces using BFS traversal of the MST, then computes final transformations.

### Algorithm:
```
1. Start BFS from piece 0, place at grid center
2. For each piece in BFS queue:
   a. Get adjacency info from MST edge
   b. Determine relative position:
      - horizontal → neighbor to right/left
      - vertical → neighbor above/below
   c. Find nearest free cell if position occupied
   d. Assign rotation from matching result
   
3. Normalize positions (shift to start from 0,0)

4. Compute final transformations:
   - solved_center = (col × piece_width + piece_width/2,
                      row × piece_height + piece_height/2)
   - solved_rotation = rotation from matching
```

### Complexity:
- **Time:** O(N) for BFS
- **Space:** O(N) for grid storage

---

## Phase 6: Visualization & Animation

**File:** `src/visualization/animator.py`

### What it does:
Generates MP4/GIF animation showing pieces moving from scattered positions to solved grid.

### Animation Styles:

| Style | Description |
|-------|-------------|
| **SIMULTANEOUS** | All pieces move at once |
| **SEQUENTIAL** | Pieces move one at a time |
| **WAVE** | Staggered start with overlap |

### Algorithm:
```
1. Create matplotlib figure with canvas
2. For each frame (fps × duration):
   a. Calculate global progress t ∈ [0, 1]
   b. For each piece:
      - Interpolate position (ease-in-out cubic)
      - Interpolate rotation (shortest path)
      - Apply rotation to image
      - Update display extent
3. Save animation using FFMpegWriter or PillowWriter
```

### Interpolation:
```python
# Ease-in-out cubic
t_smooth = t² × (3 - 2t)
position = start + (end - start) × t_smooth
```

### Complexity:
- **Time:** O(F × N × P²) where F = frame count
- **Space:** O(N × P²) for piece images

---

## Overall Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (800×800)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Load Image                                        │
│  • Support .rgb, .png, .jpg formats                         │
│  • O(W×H)                                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Segment Pieces                                    │
│  • Threshold → Contours → Extract                           │
│  • Estimate initial rotations                               │
│  • O(W×H + N×P²)                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Extract Features                                  │
│  • Color histograms (96 dims)                               │
│  • Gradient histograms (16 dims)                            │
│  • Texture features (9 dims)                                │
│  • O(N×P)                                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Match Pieces (MST)                                │
│  • Pairwise compatibility (N² × 16 comparisons)             │
│  • Chi-squared distance on features                         │
│  • Minimum Spanning Tree                                    │
│  • O(N²×P)                                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Assemble Grid                                     │
│  • BFS on MST to assign positions                           │
│  • Compute final transformations                            │
│  • O(N)                                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 6: Animate & Render                                  │
│  • Interpolate positions/rotations                          │
│  • Generate MP4/GIF                                         │
│  • O(F×N×P²)                                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: solution.mp4 + solved.png              │
└─────────────────────────────────────────────────────────────┘
```

---

## Test Results

| Sample | Pieces | Grid | Matching | Time | Status |
|--------|--------|------|----------|------|--------|
| `mona_lisa_translate.png` | 16 | 4×4 | Histogram | ~2s | ✅ Pass |
| `sample1_translate.png` | 20 | 4×5 | Histogram | ~3s | ✅ Pass |
| `mona_lisa_translate.rgb` | 144 | 12×12 | SSD | ~30s | ✅ Pass |

---

## Key Algorithms Used

1. **Contour Detection** - OpenCV's findContours with RETR_EXTERNAL
2. **Rotation Estimation** - minAreaRect for rectangular piece orientation
3. **Histogram Matching** - Chi-squared distance for robust comparison
4. **MST Reconstruction** - Kruskal's algorithm via scipy
5. **BFS Grid Assignment** - Breadth-first traversal for position assignment
6. **Cubic Interpolation** - Smooth ease-in-out animation curves

---

## Project Structure

```
image-puzzle-solver/
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
├── README.md                    # User documentation
├── ALGORITHM_REPORT.md          # This file
└── src/
    ├── segmentation/
    │   └── extractor.py         # Phase 2: Piece extraction
    ├── solver/
    │   ├── features.py          # Phase 3: Feature extraction
    │   ├── edge_matcher.py      # Phase 4: Edge matching
    │   └── grid_solver.py       # Phase 4-5: MST + Grid assembly
    ├── visualization/
    │   └── animator.py          # Phase 6: Animation
    └── utils/
        ├── piece.py             # Piece dataclass
        └── io.py                # Phase 1: Image I/O
```

---

## References

- OpenCV documentation for contour detection and image processing
- SciPy sparse graph algorithms for MST computation
- Matplotlib animation module for video generation

---

*Report generated: November 30, 2025*
