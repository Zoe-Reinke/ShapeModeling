**Title:** `MeshUtils::findReferenceMesh` returns a mesh outside the random subset when there are more than 30 shapes

### Summary

For more than 30 meshes, `MeshUtils::findReferenceMesh` (and so `find_reference_image_index` / `find_reference_mesh_index` in Python) does not return the medoid. It returns the lowest-index mesh that is **not** in the random subset, which is a mesh that was never compared to any other.

Affected: v6.7.0 and current `master` (`Libs/Mesh/MeshUtils.cpp`).

### Cause

With the default `random_subset_size = -1`, only a random subset of 30 meshes goes through the pairwise ICP. The mean-distance vector is still allocated for **all** meshes and initialised to 0:

```cpp
std::vector<double> means(meshes.size(), 0);
...
auto smallest = std::min_element(means.begin(), means.end());
return std::distance(means.begin(), smallest);
```

Meshes outside the subset keep `means[i] == 0`. Distances are always ≥ 0, so one of those meshes always wins the argmin. With N ≫ 30, the result is almost always index 0, or the first index that did not land in the subset.

The normalisation `count = meshes.size() - 1` should also be `num_meshes - 1`. This does not change the argmin, but the mean values are wrong.

### Impact

Grooming workflows that pick the alignment reference with `find_reference_image_index` get an arbitrary shape as reference once the cohort has more than 30 shapes. This includes the recommended segmentation grooming, which rigidly ICP-aligns every shape to that reference. If index 0 happens to be an unusual shape, the pre-alignment of the whole cohort, and with it the particle initialisation, suffers without any warning.

### Reproduction

`repro_find_reference_mesh.py` (attached, needs only numpy and scipy from the ShapeWorks env) builds 100 concentric spheres with radii 20.0 … 10.1 mm, where index 0 is the largest. After centroid-matched rigid ICP the pairwise distance is about |r_i − r_j|, so the medoid is the median-radius sphere (index about 50).

```
expected medoid (median r): index ~49/50 (r ≈ 15 mm)
findReferenceMesh(default): index <small> (r ≈ 20 mm)   <- extreme sphere, never compared
findReferenceMesh(..., 0) : index ~50                   <- full pairwise is correct
```

(Paste your actual output here.)

### Proposed fix

Take the argmin only over meshes that were compared, and normalise by `num_meshes - 1`. A patch is attached (`fix_findReferenceMesh.patch`).

### Minor, possibly worth a look while touching this

- Each unordered pair gets one directional distance (i→j only), and that value is credited to both meshes. A symmetric distance would be more robust for shapes that differ a lot.
- The pairwise ICP uses only 10 iterations.
