# Camera Subsets: What Fewer Cameras Cost (Synthetic Lab Rig)

Epic #9790, child #9796. Synthetic three-view session (`reconstruct synth`, 48 frames, 1.0 px noise, 2% outliers, seed 0), matched with every camera subset: the full match (reference), the three pairs (triangulated) and the three singles (image-space fit with the full match's cameras). Reprojection RMS is per view, held-out views marked; 3-D RMS and joint-angle RMS are against the reference over the frames both have. Regenerate with `python3 -m scripts.motion_capture.camera_subsets_evidence`.

| variant          | views                      | source      | down_line px     | face_on px       | overhead px      | 3-D vs ref mm | angles vs ref deg |
| ---------------- | -------------------------- | ----------- | ---------------- | ---------------- | ---------------- | ------------- | ----------------- |
| (default)        | face_on,down_line,overhead | triangulate | 12.4             | 16.6             | 13.8             | 0.0           | 0.0               |
| pair_do          | down_line,overhead         | triangulate | 13.6             | 18.9 (held out)  | 13.5             | 32.2          | 8.5               |
| pair_fd          | face_on,down_line          | triangulate | 13.0             | 23.5             | 19.9 (held out)  | 44.0          | 8.3               |
| pair_fo          | face_on,overhead           | triangulate | 26.0 (held out)  | 21.3             | 15.5             | 71.4          | 25.0              |
| single_down_line | down_line                  | image_space | 12.4             | 786.1 (held out) | 693.3 (held out) | n/a           | 42.3              |
| single_face_on   | face_on                    | image_space | 514.2 (held out) | 21.3             | 509.7 (held out) | n/a           | 55.0              |
| single_overhead  | overhead                   | image_space | 367.5 (held out) | 373.3 (held out) | 12.0             | n/a           | 48.6              |

Reading it: a pair's held-out view is the honest test of that pair; a single view's reprojection is small by construction and its angle error against the reference shows what one camera cannot see (depth-direction motion).
