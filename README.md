##### Simple datamining project, kmeans algorithm visualization using:
 - pygame
 - PIL
 - numpy

# Syntax

Window takes 4 arguments such as:
```python
img_path: str,
img_height: int,
img_width: int,
K: int
```    

Width and height are target sizes.
Graphs will draw **2 * width * height + 2 * K** circles, keep in mind it's pygame.

# Controls

 - A / D - turn graphs around y axis
 - S - shows original and comressed image using PIL module
 - SPACE - updates KMeans Algorighm


