# MotionBERT Monocular 3-D Lifting: Evaluation and Decision

Issue #9683 (epic #9677). Date: 2026-09-07. Harness:
`scripts/research/motionbert_eval.py`; raw numbers:
`docs/motion_capture/evidence/motionbert_eval_results.json`.

## Question

Single-camera sessions stop at 2-D events, tempo and image-plane angles.
Would lifting the 2-D track to 3-D with MotionBERT (Zhu et al., 2023) give a
useful 3-D result for one camera, and should it become a `lift` stage behind
the estimator registry?

## Decision

**Not adopted as a pipeline stage.** Three independent reasons, any one of
which would be enough:

1. **Accuracy is an order of magnitude short of the multi-camera fit.** On
   the synthetic bundle with 3-D truth, the best view reaches 103 mm mean
   per-joint error after per-frame similarity alignment (rotation, scale
   and translation all given away), with shoulder and hip widths wrong by
   36 to 46 % in the face-on view. The joint fit on the same bundle holds
   bone lengths within 3 % and reprojects within 2 px.
2. **The released weights are not licence-clean for this project.** The
   code is Apache-2.0, but every 3-D pose checkpoint is fine-tuned on
   Human3.6M, whose licence is "limited to academic use only" with
   commercial use by separate agreement. UpstreamDrift is MIT and the rig
   is meant for coaching use; shipping or depending on those weights would
   put the whole pipeline under an academic-only condition.
3. **Dependency weight.** CPU torch adds 464 MB (655 MB venv), the
   checkpoint 64 MB, for a stage whose output the fit cannot trust.

What stays: single-camera sessions keep the honest 2-D analysis. If a
monocular 3-D estimate is ever wanted for _visualisation only_, the route is
a licence-clean model (training data not under an academic-only licence),
exported to ONNX, evaluated with this harness first.

## Setup

| Item       | Value                                                                                                                                                                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code       | github.com/Walter0807/MotionBERT, Apache-2.0                                                                                                                                                                                                             |
| Checkpoint | `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`, 64,099,897 B                                                                                                                                                                                         |
| SHA-256    | `9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f` (matches the Hugging Face LFS record)                                                                                                                                                 |
| Model      | DSTformer, 16.0 M parameters, 243-frame clips, flip-test averaging                                                                                                                                                                                       |
| Runtime    | torch 2.14.0+cpu, Python 3.12, isolated `uv` environment; the checkpoint is loaded with `weights_only=True`                                                                                                                                              |
| Input      | 2-D keypoints mapped to H36M-17: Spine = mid-point of hip and neck, Head = nose + 0.5 (nose − neck); centred and divided by min(w, h)/2 as in `dataset_wild.read_input`; confidence as the third channel; missing joints at the centre with confidence 0 |
| Output     | (T, 17, 3) in normalised image units, root depth zeroed at clip start (global_lite convention)                                                                                                                                                           |

Training-data fit: Human3.6M is 50 fps, indoor, four fixed cameras at roughly
eye height, actors walking, sitting, and so on. Golf swings at 60 to 120 fps
from a bay camera, and an overhead view, are outside that distribution; the
harness resampled the real take to 60 fps.

## Synthetic Bundle With Truth

`reconstruct synth`: 3 views, 240 frames at 60 fps, 1 px noise, 5 %
occlusion, 2 % outliers, 1920×1200. Errors against the true 3-D joints over
the 15 joints the fit uses.

| View      | MPJPE, per-frame similarity aligned | p95    | MPJPE, one global scale + per-frame rotation | scale CV | ms/frame (CPU) |
| --------- | ----------------------------------- | ------ | -------------------------------------------- | -------- | -------------- |
| face_on   | 102.5 mm                            | 143 mm | 103.4 mm                                     | 0.033    | 6.6            |
| down_line | 110.8 mm                            | 194 mm | 111.5 mm                                     | 0.037    | 6.4            |
| overhead  | 164.6 mm                            | 222 mm | 167.2 mm                                     | 0.066    | 7.0            |

Bone-length error after one global scale (median relative error):

| Segment                 | face_on | down_line | overhead |
| ----------------------- | ------- | --------- | -------- |
| hip → knee (left)       | 0.9 %   | 2.2 %     | 12.1 %   |
| knee → ankle (left)     | 1.4 %   | 1.8 %     | 12.5 %   |
| shoulder → elbow (left) | 6.7 %   | 6.7 %     | 3.5 %    |
| elbow → wrist (left)    | 4.1 %   | 5.2 %     | 7.4 %    |
| hip → neck              | 0.9 %   | 2.3 %     | 6.3 %    |
| shoulder width          | 46.3 %  | 9.9 %     | 16.4 %   |
| hip width               | 35.7 %  | 16.8 %    | 3.6 %    |

Reading: limb lengths along the image plane are recovered well; the
_depth-dominated_ widths (shoulders and hips seen face-on) are not. That is the
monocular ambiguity itself, not noise: the network has to guess how far the
far shoulder is behind the near one. Frame-to-frame bone-length variation
(CV) of the lifted output reaches 0.30 for shoulder width in the face-on
view, so the result also fails the "rigid segments" requirement that the
multi-camera fit enforces by construction.

Caveat in MotionBERT's favour: the synthetic golfer is a rigid skeleton
driven by a smooth rotation, not a human motion capture; real swings may sit
closer to the training prior. The real-take figures below say otherwise for
the arms.

## Real Take (Single Camera)

Take 2, cam_b, 1280×720 at 119.7 fps, MediaPipe 2-D, resampled to 59.9 fps
(720 frames). Only 45 frames had all 15 joints at confidence ≥ 0.5 (the
golfer is out of frame for much of the take); consistency is measured on
those.

| Metric                                           | Value              |
| ------------------------------------------------ | ------------------ |
| Inference                                        | 7.5 ms/frame (CPU) |
| Bone-length CV, thigh / shank (left)             | 0.007 / 0.013      |
| Bone-length CV, upper arm / forearm (left)       | 0.093 / 0.273      |
| Bone-length CV, shoulder width / hip width       | 0.137 / 0.044      |
| Left–right asymmetry, thigh / upper arm (median) | 9.5 % / 9.3 %      |
| Knee flexion, 5th–95th percentile                | 9° to 27°          |
| Lead-elbow flexion, 5th–95th percentile          | 31° to 68°         |

Legs are stable; the arms, which carry the swing, change length by up to
27 % between frames. No 3-D truth exists for this take (the three-view
recording is still owner-side), so this is a consistency statement, not an
accuracy one.

## Alternatives Checked

- **VideoPose3D**: CC-BY-NC, unusable.
- **ONNX exports of MotionBERT** exist on Hugging Face (third-party); they
  would remove torch but not the Human3.6M licence condition or the accuracy
  gap.
- **Multi-camera reconstruction** (the existing pipeline) is the answer to
  "3-D from this rig"; two cameras already remove the depth ambiguity that
  produces every large error above.

## Reproduce

```bash
# once, outside the project venv
uv venv --python 3.12 mb-venv && uv pip install --python mb-venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu numpy pyyaml easydict scipy
git clone --depth 1 https://github.com/Walter0807/MotionBERT.git
curl -L -o MB_ft_h36m_global_lite.bin https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
sha256sum MB_ft_h36m_global_lite.bin   # 9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f
python3 -m src.motion_capture.reconstruct synth --out synth --frames 240 --fps 60
mb-venv/bin/python scripts/research/motionbert_eval.py synth [observations/<view>.json]
```
