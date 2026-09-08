# Sparse Annotations: Fitting From Every K-th Frame (Synthetic Lab Rig)

Epic #9791, child #9802. The synthetic three-view detections were turned into manual sets that keep every k-th frame (all joints clicked, 1 px noise), then reconstructed and fitted with the golfer through the normal path (`rig reconstruct --observations observations_manual_kK`, `rig fit-model`). Model landmark RMS is against the truth and against the dense (k = 1) fit. Regenerate with `python3 -m scripts.motion_capture.sparse_fit_evidence`.

| stride k | frames with clicks | reconstruction RMS px | model vs truth mm | model vs dense fit mm | velocity violations | rejected |
| -------- | ------------------ | --------------------- | ----------------- | --------------------- | ------------------- | -------- |
| 1        | 48/48              | 1.01                  | 68.7              | 0.0                   | 0                   | 96       |
| 3        | 16/48              | 1.00                  | 68.3              | 5.8                   | 0                   | 32       |
| 5        | 10/48              | 0.94                  | 68.5              | 8.1                   | 0                   | 20       |
| 10       | 5/48               | 0.90                  | 89.1              | 57.8                  | 0                   | 10       |

Frames without clicks are still produced by the continuity prior; the reconstruction places them by the segment priors alone, so their accuracy is the model fit's, not the triangulation's.
