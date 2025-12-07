# Computational Image Puzzle Solver

## File Structure

|File|Responsibility|
| ---- | ---- |
|`main.py`|Image I/O, Tile contour detection, and calls to the solver/animator.|
|`animation_generator.py`|(Yuxin) Contains logic for simulating the puzzle solution and generating the final solution GIF animation using rigid transformations (translation + rotation).|
|`solver.py`|(Kevin) Contains logic for solving the puzzle using Greedy Algorithm and edge analysis for optimal matching |
|`samples`|Input directory.|
|`outputs`|Output directory.|
