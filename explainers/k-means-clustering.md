# K-Means Clustering

How to partition a set of points into k groups by finding the natural centres —
and why it's the right primitive for colour quantisation.

---

## The core idea

Given N points and a target number of groups k, k-means finds k **centroids**
(one per group) such that every point is assigned to its nearest centroid and
the total squared distance from each point to its centroid is minimised.

No labels needed. No training data. The structure comes from the geometry of the
points themselves.

---

## The algorithm

Four steps, repeated until stable:

```
1. Initialise  — pick k starting centroid positions
2. Assign      — for each point, find its nearest centroid → that's its cluster
3. Update      — recompute each centroid as the mean of its assigned points
4. Repeat      — go back to step 2 until assignments stop changing (or max iters)
```

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300)
km.fit(pixels)          # pixels: (N, 3) array of RGB values
labels   = km.labels_   # which cluster each pixel belongs to
centres  = km.cluster_centers_  # the two centroid colours
```

The loop terminates when no point changes cluster between iterations. In practice
it converges quickly — usually under 50 iterations for small problems.

---

## Visual intuition

Imagine 20 points scattered on a 2D plane. You want to find 2 clusters.

**Step 0 — random init:**
```
  . . .     C1       . .
      .   . .          .
  . .           C2 . .
```
C1 and C2 are placed randomly — not where the clusters actually are.

**Step 1 — assign to nearest centroid:**
```
  a a a     C1       b b
      a   a a          b
  a a           C2 b b
```
Every point gets label a or b based on distance to C1 or C2.

**Step 2 — recompute centroids:**
```
  a a a       C1'    b b
      a   a a          b
  a a       C2'  b b
```
C1' is the mean of all a-points. C2' is the mean of all b-points.
The centroids have moved toward the actual cluster centres.

**Repeat** until the centroids stop moving. Two or three iterations usually gets
you close; convergence is reached when no point switches cluster.

---

## Initialisation matters

Random init can place two centroids in the same cluster — one cluster gets split,
another goes uncovered. The result is a local minimum, not the global one.

**k-means++** fixes this: it picks the first centroid at random, then picks each
subsequent centroid with probability proportional to its squared distance from
the nearest already-chosen centroid. This spreads the initial centroids out and
makes bad starts rare.

```python
KMeans(init='k-means++')  # default in scikit-learn — use this
```

For colour quantisation with k=2, the difference is small (only two centroids),
but it's good practice and costs nothing.

---

## Choosing k

When k is not fixed, two methods help:

**Elbow method** — plot total within-cluster variance (inertia) against k.
The curve bends sharply at the "natural" number of clusters. Beyond that point,
adding more clusters gives diminishing returns.

**Silhouette score** — measures how well each point fits its own cluster vs. the
next nearest cluster. Score near +1 = tight, well-separated clusters.
Score near 0 = overlapping. Score negative = likely mis-assigned.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(points, km.labels_)
```

For this project, k=2 is fixed by the ZX Spectrum hardware constraint (each 8×8
cell can only hold two colours). No search needed.

---

> **Coming from C/JS/TS**
>
> K-means is "find natural groups in data" — the unsupervised equivalent of sorting
> by category when you don't know the categories ahead of time.
>
> In C: think of it as computing k representative values from a set, then bucketing
> every element into the nearest bucket. The centroid update is just `sum / count`.
>
> In JS/TS: it's like `Array.reduce` to compute group means, then `Array.map` to
> re-assign each item to the nearest mean. The distance function is the only
> domain-specific piece — Euclidean distance for spatial data, but you can swap
> it for anything (Hamming distance for strings, cosine similarity for vectors).
>
> You've likely used this thinking already: grouping log entries by error pattern,
> bucketing users into behaviour cohorts, finding representative cache keys. K-means
> formalises that intuition into an algorithm that finds the buckets automatically.

---

> **In practice**
>
> - **Customer segmentation** — cluster customers by spend, frequency, recency.
>   The centroids describe the "typical" customer in each segment.
> - **Anomaly detection** — a point far from all centroids is an outlier.
>   K-means gives you a distance-to-nearest-centroid score for free.
> - **Colour quantisation** — reduce an image to k representative colours.
>   Exactly what this project does.
> - **Feature grouping** — cluster high-dimensional features before dimensionality
>   reduction. The cluster assignments become a compact representation.
> - **Initialising other algorithms** — k-means centroids are a good warm start
>   for Gaussian Mixture Models or hierarchical clustering.

---

## Connection to this POC (Glyph / ZX Spectrum colour fitting)

The ZX Spectrum screen is divided into 768 cells of 8×8 pixels. Each cell stores
exactly two colours from a fixed 15-colour palette. To convert an arbitrary RGB
image to Spectrum format, each cell needs to be reduced to its two best-fit colours.

That's a clustering problem: given 64 RGB pixels (the cell), find 2 centroids
that minimise the total distance from each pixel to its nearest centroid.

```
For each 8×8 cell:
  pixels = cell.reshape(64, 3)       # 64 RGB points in 3D colour space
  km = KMeans(n_clusters=2)
  km.fit(pixels)
  colour_a, colour_b = km.cluster_centers_  # two best-fit RGB values
  # → snap each to the nearest Spectrum palette entry
```

768 cells = 768 independent k-means runs. Each one is tiny (64 points, k=2)
so the total cost is low. The quality of the result depends entirely on how well
the two centroids capture the dominant colours in that cell — which is exactly
what k-means minimises.

The Spectrum palette snap (nearest palette colour by Euclidean RGB distance) is
not part of k-means — it's a post-processing step that maps the floating-point
centroid to the closest discrete palette entry.
