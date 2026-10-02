# Implementation Handoff — Exploratory GS3DX Simscape Model (#10950)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-gs3dx-v2`
- Branch: `feat/simscape-gs3dx-exploratory`
- Baseline commit: `2d5830d18` (origin/main)
- Implementation commit: `SELF`
- Pull request: #11179 (draft, open); #10963 was closed by the 2026-09-29 history scrub
- Governing issue/epic: #10950 (children #10951–#10959, #10985, #10986, #11011; in progress #10979); quality issues #11156–#11160; forward-dynamics goal #11173
- Lease session: `claude-gs3dx-shape-20260928` (#10979); expired, not renewed at turnover: the next agent posts its own lease
- Development log: `DL-#10950`

## Turnover (2026-10-01): Start Here

Work was paused so another agent can continue. Do not start a second MATLAB
while `%USERPROFILE%/gs3dx_matlab.lock` names a live process. A stale lock
whose owner is dead is cleared by the runner itself. The two MATLAB processes
started 2026-09-25 and 2026-09-29 (`modes.m`, `probe_signs.m`) belong to the
owner: never kill them. Scratch batch scripts live under `%TEMP%`
(`gs3dx_shift_leg_sim.m`, `gs3dx_shift_leg_probe2.m`); they are not in git.

**Where things live**

- Branch `feat/simscape-gs3dx-exploratory`, draft PR #11179 (replaces #10963,
  closed by the history scrub). The branch tip is local only; it has not been pushed.
- Playbook for the forward-dynamics goal (#11173): `exploratory_gs3dx/docs/FORWARD_DYNAMICS.md`
  (ladder, lessons with evidence, convergence ideas, roadmap with acceptance
  per step, and a "Next step for the next agent" line). Read it first.
- Range of motion: `docs/ROM.md`. Fit, balance, human model: `docs/FIT.md`,
  `docs/HUMAN.md`. This file's "Next Steps" keeps the full history.
- Owner-swing epic #11161 (capture-O on five engines) lives on a separate
  branch, `feat/capture-registry-11161` (draft PR #11172, worktree
  `UpstreamDrift-worktrees/claude-epic-11161`) with its own HANDOFF.

**How to run MATLAB here**

- Only through `exploratory_gs3dx/tools/run_matlab_locked.ps1` (one MATLAB at a
  time; GS3DX runs share the Simulink cache). Put the script in a scratch
  folder, end it with `fprintf("GS3DX_BATCH_DONE\n"); quit(0, "force");`, run the
  runner in the background, read `EXIT n` at the end of the log.
- Capture data is private: set `$env:CAPTURE_DATA_DIR` to
  `[IO.Path]::Combine($env:USERPROFILE,'Private_Backups','Capture_Data_Private')`
  in the same PowerShell before the runner. Never write that absolute path,
  the capture's vendor or any third-party name into a committed file, issue
  or PR (use `capture-A`, `capture-B`, `capture-O`).
- In a script: `addpath(<exploratory_gs3dx>); gs3dx_setup();` (never
  `setup_matlab_environment`: it calls `savepath`). Never load-and-save the
  original models; `gs3dx_save_model` guards the writes. Compiled blocks <= 975.
- Tests: `matlab.unittest.TestRunner.withTextOutput` over one test file; the
  IK tests take 5-30 min, Human simulations about 30 min.

**Open work, in order (each with its first action)**

1. **#11156 lead arm straight (blocked on the owner, test RED).** Added
   `gs3dx_golf_rom` (lead elbow flexion band <= 20 deg), the IK option `rom=`
   and `tests/test_gs3dx_lead_arm.m`. `the_golf_band_narrows_only_the_lead_elbow`
   passes. Free-elbow calibration still fails the IK test: held 18.5-41.7 deg
   (limit 21), RMS 43.2 mm against 23.3 mm free. The 2026-09-30 banded-offset
   probe also fails, so do not switch the test to band calibration and do not
   re-run that probe. Both fits below used offsets calibrated with
   `rom_weight=3` and `rom=gs3dx_golf_rom()` on `calibration_frames=1:15:f0-90`,
   then every 10th frame to impact (regularization 0.01/0.02/false/0.5). The
   free column is not the old free-elbow calibration.

   | Fit                           | Lead elbow (deg) | Mean RMS (mm) |
   | ----------------------------- | ---------------- | ------------- |
   | Free, band-calibrated offsets | 57.7 to 120.1    | 37.0          |
   | Held, same offsets, weight 3  | 2.6 to 65.8      | 76.8          |

   Held max 65.8 deg exceeds 21. Held RMS is 39.8 mm above free (cap is free +
   15 mm = 52.0 mm). Wall 2778 s, batch exit 0. Scratch mat
   `lead_arm_probe_11156.mat` under the session temp probe folder, not in git.
   Next action is an owner decision on #11156. Do not loosen 20 deg or 15 mm.

2. **#11160 club motion matches (acceptance red).** `gs3dx_club_match` is unchanged
   (10 mm head RMS, 20 mm head max, 2 deg shaft max, 2 % impact speed).
   `ik.points` is in place. `gs3dx_club_prediction` reports head, grip and a
   face normal; `gs3dx_hand_centres` is the shared hand-sphere solve.
   `frame_one_has_finite_club_geometry` and `residual_matches_points_to_capture`
   passed. `the_club_matches_the_capture` failed on every 10th frame to impact
   (free-elbow offset calibration, regularization 0.01/0.02/false/0.5):

   | head RMS mm | head max mm | grip RMS mm | shaft RMS deg | face RMS deg | impact speed err % |
   | ----------- | ----------- | ----------- | ------------- | ------------ | ------------------ |
   | 20.012      | 41.987      | 28.344      | 1.609         | 54.206       | -2.009             |

   `M.ok` failed head RMS, head max, shaft max (the max was not logged; RMS is
   under 2 deg) and speed. Grip is not part of `M.ok`. Head, grip and speed are
   in the address frame and are the model gap. The face normal now stays in
   that frame (`Rk * e`, fast tests EXIT 0). A fresh match gives face RMS
   12.0 deg and shaft max 3.11 deg. The head error is not a constant length
   bias: along-shaft mean 1.0 mm, along-shaft RMS 19.5 mm, perpendicular
   RMS 4.3 mm. The simulated grip-to-head length is rigid at 787.1 mm. The
   worst frame is capture 441, 97 ms before impact (frame 476), 41.7 mm
   along the shaft and 4.6 mm perpendicular. Do not add a constant shaft
   Wrist maximums at that weight are 53.5 mm lead and 68.6 mm trail, past the
   35 mm maximum in `test_gs3dx_fit`. Grip error is sideways: 24.8 mm RMS and
   50.7 mm max, which is the 3.70 deg shaft angle on the 787 mm shaft. The
   printed "worst frame 1" is not usable (the index was broadcast). Weight 10
   is rejected. Equal weights already miss the shaft at 3.11 deg, so do not
   try another head weight. Next action is an owner decision: the 10 mm head
   bar and the 35 mm wrist bar were not met together, and the 2 deg shaft bar
   was missed at both weights. Do not loosen those numbers.

3. **#11173 roadmap step 4** (balance-consistent leg reference) before step 2.
   `gs3dx_foot_split` is the closed-form two-foot split (5/5 synthetic tests,
   EXIT 0). `gs3dx_capture_foot_residual` reports the split on a kinematic
   GRF. The scratch figure moment RMS 857 N·m is not evidence: `k.com` is
   rotated into the target axes but keeps the capture origin, while the
   ankle centres use the address waist origin. With `com_offset = S.' * origin`
   the address COM is 0.149 m from the ankle midpoint, and the scratch is
   force RMS 0, moment RMS 152.5 N·m (0.193 body-weights at a 1 m lever),
   impact vertical GRF 1.41 body weights (EXIT 0). The 857 N·m figure is
   void. The vertical part is 100.2 N·m and the horizontal part is 109.0 N·m.
   `gs3dx_hull_moment` on the heel-and-toe hull (foot length from
   `gs3dx_leg_table`, 3/3 tests, EXIT 0) is 0 N·m RMS on the capture: the
   address-frame centre of mass stays inside that hull, so the 100.2 N·m
   figure is the two-ankle line, not a centre of mass outside the feet.
   `gs3dx_support_residual` (5/5 tests, EXIT 0) puts that shear shift on the
   same hull: force residual 0, tip moment 29.4 N·m RMS (0.037 body-weights
   at a 1 m lever) and 89.7 N·m at impact. `gs3dx_com_shift` (3/3 tests,
   EXIT 0) is the smallest horizontal centre-of-mass move that lands that
   centre of pressure on the hull: 35.7 mm RMS, 183.9 mm max, 80.6 mm at
   impact. The corrected moment is 0 only because the ground force is held
   fixed. Do not apply that shift. The 184 mm sample is frame 539, 0.175 s
   after impact, inside the club-head gap (0.12–0.21 s after impact), so it
   is not a pelvis correction. To impact the zero-moment shift is 21.4 mm
   RMS. Point-mass angular momentum does not remove the residual: to impact
   it rises from 24.8 N·m to 31.8 N·m RMS (0.040 body-weights at a 1 m
   lever), and at impact from 89.7 N·m to 143.6 N·m, with |hdot| 55.8 N·m
   (7/7 support tests, 2/2 momentum tests, 3/3 kinematic GRF tests, EXIT 0).
   Segment spin is not in that rate. The capture toe is shorter than the
   model toe (141 mm versus 205 mm ahead of the ankle). At impact the
   zero-moment centre of pressure is 163 mm ahead of the right ankle: 43 mm
   inside the model toe and 22 mm past the capture toe. A capture-toe hull
   raises the to-impact residual from 24.8 N·m to 42.3 N·m. Do not use the
   capture toe. `gs3dx_foot_contacts` spaces heel and toe by the stance foot
   width (tests EXIT 0). At impact the zero-moment centre of pressure is
   587 mm across the right-foot axis (half-width 55 mm). The eight-point
   hull only trims the to-impact residual from 24.8 N·m to 21.9 N·m, and
   with point-mass angular momentum from 31.8 N·m to 29.3 N·m (0.037
   body-weights at a 1 m lever). Impact stays 81.4 N·m, or 135.4 N·m with
   that rate. At impact the centre of pressure is 73 mm past the left toe
   (along the foot 276 mm, toe at 205 mm), with the vertical force at 1.41
   body weights. Thirty of the 476 frames through impact are outside the
   hull; the largest gap is 105 mm, 19 ms before impact. A pelvis nudge
   that covers 73 mm inside the downswing adds a horizontal force large
   enough to move the centre of pressure again, so do not apply
   `gs3dx_com_shift` to the pelvis. Trunk spin (`gs3dx_trunk_spin`, tests
   EXIT 0) does not close it: the to-impact residual goes from 21.9 N·m to
   29.3 N·m with point-mass angular momentum and to 32.5 N·m with trunk spin
   (0.041 body-weights at a 1 m lever). Impact goes from 81.4 N·m to 135.4
   N·m and then 130.3 N·m. Keep the foot-width residual without those rates:
   21.9 N·m through impact, 81.4 N·m at impact. `gs3dx_vertical_share`
   (3/3 tests, EXIT 0) places that vertical force on the clipped centre of
   pressure. `gs3dx_contact_wrench` compares the tip moment only (4/4 tests,
   EXIT 0). On the capture foot-width hull it does not error, the force
   residual is 0, and the moment is 21.863 N·m through impact and 81.427
   N·m at impact. That is the one contact solve. It is 0.028 body-weights
   at a 1 m lever. `tests/test_gs3dx_contact_solve.m` pins those three
   numbers (1/1, EXIT 0). A finish-phase `gs3dx_contact_check` of
   `GS3DX_Human` to 1.809 s threw at 0.389 s (nonlinear step at hmin,
   EXIT 1, 2026-09-30). The stored rates were then replaced by the
   derivative of those angles (SAVED 1). Lead-elbow track row 1 at 1.458 s
   stepped from 150 deg to -5 deg. That sample was held at 150 deg
   (LE_STEP 0, SAVED 1, EXIT 0). The range table was not changed. The
   0.389 s solver stop is still earlier than this sample. A later run to
   1.809 s returned status success (EXIT 0) with a vertical ground
   reaction of 2466 body weights at 0.386 s, and 12.5 body weights within
   0.05 s of impact, against a kinematic peak of 1.85 body weights. The
   2466 body-weight sample is the same instant the previous run died, so
   the 12.5 figure is not the impact-support measurement. Between 0.30 s
   and 0.45 s the largest reference step is 1.15 deg (lead forearm, at
   0.447 s). Arc-only wrists were then shifted by whole turns (SAVED 1):
   lead flexion median -1095 deg to -15.4 deg, lead radial -376.5 to
   -16.5, trail flexion -1406 to 34.3. Spans stayed 50 and 150 deg. The
   rerun to 1.809 s finished (EXIT 0) with 365 body weights at 0.449 s and
   0 body weights within 0.05 s of capture impact. The kinematic peak is
   1.85 body weights. The 365 body-weight sample is not the impact-support
   measurement, and the near-impact zero is after that spike. The 5 % bar
   is not claimed. At 0.45 s the lead and trail ankle references sit
   0.23 mm and 0.27 mm below the address floor. The deepest reference
   penetration is 26.6 mm on the lead foot at 1.814 s. The 0.45 s
   clearance does not explain the 365 body-weight spike. At that sample
   the ankle track values are about -12, -9, 13 and -4 deg. Trail ankle
   dorsiflexion track reaches 30.5 deg; with sign -1 that is about 14 deg
   of dorsiflexion, inside the -50 to 20 deg table.
   `human_references_stay_in_the_human_range` passed again (EXIT 0).
   A run to 0.55 s puts the whole 365 body-weight spike at 0.449 s on
   the right lesser-toe sphere, which is the only contact loaded then.
   A kinematics pose at that sample puts the sphere origin 13 mm above
   the address sole. Zeroing the trail-inversion target left that height
   unchanged, so this pose does not show the roll burying the sphere.
   At 0.45 s the simulated trail inversion is 14.5 deg against a 13.5 deg
   reference, but dorsiflexion is 42.5 deg against -4.3 deg and the
   midfoot is 22.4 deg from 0. That plantarflexion, not the reference
   pose, loads the lesser toe. Trail-ankle Kp is 50 and Kd is 1 on both
   axes. Midfoot stiffness is 2000 N\*m/rad. One run to 0.55 s with
   feedback on (EXIT 0, 2026-10-01) splits row 12 at 0.45 s into servo
   -3755.6 N\*m and balance +2130.0 N\*m (sum -1625.5 N\*m). A second run
   to 0.55 s (EXIT 0) samples the same axis before that instant. At 0.35 s
   and 0.40 s the vertical ground reaction is 0, the track is +32.3 deg
   and +35.2 deg against references -5.1 deg and -4.7 deg, and the balance
   term (+1964.9 N\*m, then +2039.6 N\*m) cancels the servo
   (-1946.1 N\*m, then -2019.0 N\*m). The ankle is already about 37 deg
   off the reference with both feet in the air. Contact is present by
   0.42 s (18479 N, 9935 N of it on the right lesser toe). The 365
   body-weight peak remains the 0.449 s sample between these prints.
   The gains stay as they are. A run to 0.36 s (EXIT 0) splits the
   +1964.9 N\*m balance term at 0.35 s into a centre-of-mass shift of
   451.2 N\*m, a foot term of 503.8 N\*m, and a rate term of 1009.9 N\*m.
   The shift is -8.7 mm, +5.1 mm, and +99.5 mm. Support over that run
   stays 0. At the first sample the servo torque is 0, so the servo
   angle matches the reference; the joint log's -340 deg reading is a
   different branch. An unsaved run to 0.55 s with BalanceGain row 12
   zeroed (EXIT 0) drops that balance torque to 0 and the servo torque
   at 0.35 s to -7.2 N\*m. The joint log reads -285 deg there, which
   would take about 14000 N\*m of servo torque, so that reading is not
   the servo angle. The peak support in this run is 22.6 body weights
   at 0.418 s (17821 N, 9797 N on the right lesser toe), against 365
   body weights at 0.449 s with the saved gain. The same unsaved
   override run to 1.809 s (EXIT 0) repeats that peak: 22.6 body
   weights at 0.418 s, and the only loaded contacts are the right big
   toe (8024 N) and the right lesser toes (9797 N). Within 0.05 s of
   capture impact the vertical support is 0, against a kinematic peak
   of 1.85 body weights. The 0.2 body-weight bar is not met. The saved
   gains are unchanged. A further unsaved run to 0.55 s (EXIT 0) repeats
   the 22.6 body-weight peak. At 0.35 s the right midfoot is 1.35 deg
   and the row-12 servo is -7.2 N\*m. At the 0.42 s sample the servo is
   +153.3 N\*m and the midfoot log reads -312.4 deg. The documented
   positive midfoot angle bends the toes down, so this single sample
   does not establish the spring angle. The 0.01 s series (EXIT 0)
   keeps the joint in radians: 0.0236 rad (1.35 deg) at 0.35 s, falling
   to 0.0128 rad (0.73 deg) at 0.41 s, then -5.452 rad at 0.42 s. The
   22.6 body-weight peak at 0.418 s sits in that one step. The spring
   is still nearly straight before the strike. From 0.35 s through
   0.41 s the row-12 servo stays between -7.2 and -4.1 N\*m and every
   contact force is 0. At 0.42 s the servo is +499 N\*m and the only
   vertical forces are the right big toe (2294 N) and the right lesser
   toes (4708 N). The strike is the first contact, and it is those two
   spheres. The kinematics solve for those contact points returned
   status -1, so those positions are not a pose. The simulated ankle
   at 0.41 s is at world z -0.705 m. The address sole is at z -1.000 m,
   so the ankle is 0.295 m above that plane eight milliseconds before
   the strike. The midfoot log steps from 0.013 rad to -5.45 rad in
   that same interval. The saved maximum step is 0.001 s. An unsaved
   run with the maximum step set to 0.0001 s (EXIT 0) still has the
   midfoot at 0.0128 rad at 0.41 s and -5.323 rad at 0.42 s, and the
   support peak is 23.1 body weights at 0.418 s. The joint-centre
   kinematic vertical ground reaction at the 360 Hz sample nearest
   that time is 1.018 body weights. At capture impact it is 1.410
   body weights, and its peak is 1.853 body weights, 0.0556 s before
   impact. The model peak is not that kinematic peak, and the model
   support at impact is 0. Over that strike the simulated right
   ankle falls from z -0.045 m at 0.35 s to z -0.705 m at 0.41 s,
   0.661 m in 0.06 s, while the vertical force is still 0. The step
   from 0.40 s to 0.41 s is 0.116 m. After the toes load, the ankle
   stops near z -0.82 m. The impact drive has no pelvis position
   time series. Its hip signals are polynomial torque coefficients
   (`HipInputZA`, `HipInputZB`, `HipInputZC`), not a prescribed height.
   With row 12 zeroed, the pelvis falls from z 0 at the start to
   z -0.424 m at 0.30 s, while the right ankle rises from z -0.883 m
   to z 0.354 m. The ankle is then 0.778 m above the pelvis, and the
   vertical force is still 0. Both are falling by 0.41 s, and the
   toes load at 0.418 s. At 0.30 s the right-leg servo and the
   balance loop cancel: hip Y is -3378 N\*m against +3350 N\*m, hip Z
   is +636 N\*m against -633 N\*m, and the knee is -6869 N\*m against
   +7030 N\*m. The vertical force is 0. With BalanceGain rows 7
   through 12 zeroed (EXIT 0) the knee balance torque is 0 and the
   knee servo is -2.1 N\*m, so the knee is on the reference, and the
   ankle is still at z 0.411 m against a pelvis at z -0.433 m. The
   support peak falls from 22.6 to 16.4 body weights, at 0.421 s.
   The saved right-leg reference at that time is hip -11.7, -36.0,
   -4.0 deg, knee -23.7 deg, and ankle 13.1, -5.5 deg. Those angles
   are a small bend. At that same sample the pelvis frame in world,
   columns then rows, is [0.545 0.765 -0.344; 0.421 -0.604 -0.676;
   -0.725 0.223 -0.651]. Its axis nearest world up is 77 deg from
   vertical. The capture pelvis up-axis is 18.8 deg from vertical at
   that time, 18.7 deg at address, and 7.9 deg at impact. The capture
   right ankle at 0.30 s is 0.919 m below the pelvis. With rows 7
   through 12 zeroed, the pelvis tilt from vertical is 22.5 deg at
   the start, 50.9 deg at 0.10 s, 102 deg at 0.20 s, and 77.1 deg at
   0.30 s. The vertical force is 0 through all four samples, and by
   0.20 s the ankle is already above the pelvis. The ground origin
   is at z -1.00027 m and AnkleHeight is 0.0702 m. A sole that far
   below the starting right ankle (z -0.883 m) would sit at z -0.953 m,
   47 mm above the ground. Raising the ground by 47.3 mm, still
   with rows 7 through 12 zeroed (EXIT 0), puts 3180 N on the feet
   at the first sample. By 0.10 s the force is 0 again. The tilt is
   22.5 deg, then 40.8, 85.4, and 110 deg at 0.30 s. The pelvis
   start rate is -13.1, -12.0, and 10.8 deg/s, a magnitude of 20.8
   deg/s. The translation start rate is -0.031, 0.005, and 0.0002 m/s.
   The tilt angle changes by about 80 deg in the first 0.20 s. The Hip
   Torque Output chart commands
   JointTorqueHip = HexPolyInputFunction(HipInput*A..G, Time)*Killswitch.
   At 0, 0.10, and 0.20 s that torque is (-174, -112, 6.6), (-107, 815,
   502), and (1199, -293, -14) N·m. These are chart outputs, not a measured
   joint reaction. Holding KillswitchInitialValue and KillswitchFinalValue
   at 0, with rows 7 through 12 still zeroed, reproduces the same four
   tilt samples (22.5, 50.9, 101.9, and 77.1 deg) and the same pelvis and
   ankle heights. The impact drive sets ModelingMode 0, which already
   zeros the original joint actuators, so that chart is not the moment in
   these runs. With rows 7 through 12 still zeroed, a 0.35 s feedback
   run (MODE 0) has leg servos at 0 at the start and Spine X feedback
   of 6545 N·m. The left balance torque at that sample is 199, 318,
   -207, -457, -205, and -304 N·m. By 0.30 s the left knee servo and
   balance are -7124 and 7219 N·m, while the right hip Y servo is
   -133 N·m and the right knee servo is -2.1 N·m. The largest upper
   torque is then RScap X at -1169 N·m. The saved spine start is not
   that sample. SpineTrackAngle begins at 80 and 22.4 deg,
   SpineStartPosition is 14.2 and 11.8 deg, and the start rates are
   101 and 63.0 deg/s against reference rates 0 and 289 deg/s. With
   Kp 74.4 N·m/deg that start pose is 4498 and 1685 N·m, not the
   logged 6545 N·m. The impact drive sets SpineStartPositionX to 0.
   The logged joint at the first sample is angle 0 and 0 deg and rate
   -150 and 300 deg/s, against the reference 80 and 22.4 deg and rate
   0 and 289 deg/s. That error is the logged 6545 N·m (position term
   5953, rate term 592). Across the 21 upper-body axes the largest
   initial torques are Spine X 6545, LS Z -2488, RS X 1853, and Spine Y
   1625 N·m. Setting those 42 start angles and rates to the first
   reference frame, still with rows 7 through 12 zeroed, leaves the
   tilt at 22.5, 19.4, 46.9, and 69.2 deg. At 0.30 s the ankle is at
   z -0.060 m and the pelvis at z -0.544 m. The largest upper-body
   torque on that run is the right wrist Y: -1210 N·m at the start
   and -2730 N·m at 0.30 s. The right-wrist joint start parameters
   are RWStartPositionX/Y and RWStartVelocityX/Y. The chart's tracking
   branch sets the wrist torque from gs3dx_track_torque. The impact
   drive's own wrist start is -38.8 and -72.0 deg, at 909 and
   -478 deg/s. With the 42 starts matched, the logged spine is
   80 and 22.4 deg at the set rates, and its feedback is 0. The
   logged right wrist is 28.2 and 194.0 deg, at 69.2 and 46.7 deg/s,
   against the set -50.0 and 34.3 deg. That gap is the -608 and
   -1210 N·m. The right-wrist position and velocity target priorities
   are None. The grip loop is Club/Weld Joint plus
   Grip/LeftHandOnClubForce and Grip/RightHandOnClubForce. Setting
   both wrists' position and velocity priorities to High, without
   saving, puts the right wrist at -48.9 and 34.5 deg against the
   targets -50.0 and 34.3 deg. The run then fails at 0 s in the right
   shoulder gimbal acceleration loop. The saved priorities stay None.
   With the wrist tracking gains at zero, the same matched starts,
   and rows 7 through 12 still zeroed, the tilt is 22.5, 20.5, 46.0,
   and 72.9 deg. The vertical force is 0 at all four samples. At
   0.30 s the ankle is at z -0.073 m and the pelvis at z -0.559 m.
   The largest upper-body torque is then the lead scapula Y at 723
   N·m at the start and the trail shoulder Z at 222 N·m. Zeroing
   the other six balance rows as well leaves the tilt at 22.5, 23.2,
   46.8, and 82.5 deg, still with no vertical force. At 0.30 s the
   ankle is at z -0.024 m and the pelvis at z -0.574 m. The leg
   servos are then at most 92 N·m, and the balance torque is 0.
   Setting the 21 upper-body start rates to zero, positions still
   matched, leaves the tilt at 22.5, 17.8, 42.6, and 58.2 deg, with
   no vertical force. At 0.30 s the ankle is at z -0.147 m and the
   pelvis at z -0.498 m. Setting the 12 upper-body tracking gains
   to zero as well leaves the tilt at 22.5, 20.0, 64.9, and 74.1 deg,
   still with no vertical force. At 0.30 s the ankle is at z 0.304 m
   and the pelvis at z -0.520 m. The same quiet state with
   GroundOffset z raised by 47.3 mm, to -0.95297 m, puts 3180 N on
   the feet at the first sample, 0 at 0.10 s, 722 N at 0.20 s, and
   794 N at 0.30 s. The tilt is 22.5, 26.5, 36.6, and 51.0 deg. The
   pelvis stays near z 0.09 m and the ankle near z -0.86 m.
   Through 0.55 s that load is the right heel at the first sample
   (3180 N), nothing at 0.10 s, the right ball at 0.20 s (386 N of
   722 N), and the right lesser toes alone at 0.30 s (794 N) and
   0.40 s (505 N). At 0.50 s the tilt is back to 19.6 deg, the
   vertical force is 0, the pelvis is at z 0.012 m, and the ankle
   is at z -0.771 m. By 1.0 s the tilt is 109 deg, the pelvis
   is at z -1.87 m, the ankle is at z -0.989 m, and the vertical
   force is 8667 N, of which the left heel carries 5232 N. The
   feet are unloaded from 0.50 s through 0.70 s. At address the
   capture centre of mass projects to 0.509 on the ankle segment,
   and it stands 0.149 m forward of that segment. The table foot
   length is 0.274 m, so the toe line is 0.205 m forward of the
   ankle and the weight line is inside the sole. The contact wrench
   at that first frame has force residual about 0 and moment 0.
   The swing residual stays the pinned 21.863 N·m RMS, and 81.427
   N·m at impact. The unapplied centre-of-mass shift that would
   put the centre of pressure on the foot hull is 0 at address,
   1.7 cm RMS through impact, 7.3 cm at impact, and 10.5 cm at
   1.303 s. `capture_com_shift_matches_the_foot_width_lever` pins
   those values (2 passed, EXIT 0). It is not applied. An IK
   on every 10th frame, scored against its own pelvis target,
   gives 11.1 mm free and 6.6 mm with the shift subtracted, a
   difference of -4.5 mm. That score is not the error against
   the capture pelvis. Scored against the original capture, the
   free fit is 11.1 mm and the shifted fit is 6.8 mm, a difference
   of -4.3 mm. The shifted pelvis point is 18.9 mm RMS from the
   capture pelvis. The 15 mm stop is not triggered. The shift is
   not written into the model. The 5 % bar is not claimed. The
   first attempt to feed that IK to `gs3dx_leg_reference` failed:
   `unique([1:10:impact impact])` is 49 frames, stride 10, and the
   last frame is 476, so the series is not even
   (`gs3dx:legref`, EXIT 0). The even grid from frame 1 at stride 10
   ends at 471. Human `LegReferenceAngle` is 12 by 654 and
   `LegReferenceTime` has 654 samples (probe EXIT 0). The unsaved
   1.0 s run (`%TEMP%/gs3dx_shift_leg_sim.m`, EXIT 0, STATUS success)
   built that even-grid reference and linearly resampled `ref.q` onto
   the 654-sample time base. It is not a size-matched override.
   Tilt, pelvis z, right-ankle z, and vertical force, every 0.10 s:
   22.5 deg, z 0, ankle -0.883 m, 0 N; 50.9 deg, -0.118 m, -0.557 m, 0 N;
   99.9 deg, -0.321 m, 0.334 m, 0 N; 69.3 deg, -0.420 m, 0.347 m, 0 N;
   64.3 deg, -0.680 m, -0.520 m, 0 N; 62.5 deg, -1.131 m, -0.760 m, 0 N;
   88.0 deg, -1.563 m, -1.000 m, 5818 N; 110.3 deg, -1.736 m, -0.891 m,
   399 N; 109.3 deg, -1.893 m, -0.916 m, 1539 N; 100.3 deg, -1.944 m,
   -1.054 m, 3587 N; 101.3 deg, -1.768 m, -1.025 m, 113 N. The vertical
   force is 0 through 0.50 s while the pelvis falls. At 1.0 s the tilt
   is 101 deg and the pelvis is at z -1.77 m. Capture impact is at
   1.319 s, so this run does not measure impact support. The 0.2
   body-weight bar is not met. Do not repeat this resample by itself.
   The same resampled reference with the quiet start (21 positions
   matched, 21 rates zero, both wrist track gains 0, BalanceGain rows
   1-12 zero) and GroundOffset z -0.95297 finished (48 frames, EXIT 0).
   The right heel takes 3180 N at the first sample, the feet unload by
   0.10 s, and the right lesser toes carry 621 N, 666 N, and 326 N at
   0.20 s, 0.30 s, and 0.40 s. At 0.50 s the tilt is 15.9 deg, the
   pelvis is at z -0.045 m, the right ankle is at z -0.758 m, and the
   vertical force is 0. It stays unloaded through 0.80 s. At 1.0 s the
   tilt is 108 deg, the pelvis is at z -1.93 m, and the vertical force
   is 3591 N, of which the left heel carries 1991 N. Do not repeat this
   combination. It does not reach capture impact at 1.319 s. The 0.2
   body-weight bar is not met. The address hold (LegReferenceAngle
   frozen on its first sample, 654 columns, LegReferenceRate 0) with
   that same quiet start and GroundOffset z -0.95297 finished
   (EXIT 0). The right heel takes 3180 N at the first sample. The
   right foot then carries 719 N, 808 N, and 502 N at 0.20 s, 0.30 s,
   and 0.40 s. At 0.50 s the tilt is 14.5 deg, the pelvis is at
   z 0.008 m, the right ankle is at z -0.767 m, and the vertical force
   is 0. At 1.0 s the tilt is 114 deg, the pelvis is at z -1.85 m, and
   the vertical force is 6717 N, of which the left heel carries
   4716 N. Holding the address angles does not keep a stance. Do not
   repeat this hold. The upper-body TrackAngle series were still the
   swing in that run. Holding every upper-body TrackAngle on its
   first column (12 variables) and every TrackRate at 0, with the
   frozen leg reference, the quiet start, and GroundOffset z -0.95297,
   finished (EXIT 0). The right heel takes 3180 N at the first sample.
   The right foot then carries 736 N, 794 N, and 480 N at 0.20 s,
   0.30 s, and 0.40 s. At 0.50 s the tilt is 13.2 deg, the pelvis is
   at z -0.007 m, the right ankle is at z -0.767 m, and the vertical
   force is 0. At 1.0 s the tilt is 124 deg, the pelvis is at
   z -1.88 m, and the vertical force is 2786 N, of which the left heel
   carries 2429 N. Freezing the upper-body targets does not keep a
   stance. Do not repeat this freeze. The same first sample is a
   3180 N right-heel strike, then the force is 0 at 0.10 s. Multiplying
   FootContactDamping by 10, from 1000 to 10000, with that same freeze
   finished (EXIT 0). The right heel takes 15513 N at the first sample
   and 286 N at 0.10 s. The right foot then carries 357 N, 876 N,
   482 N, and 109 N at 0.20 s through 0.50 s. At 0.50 s the tilt is
   15.2 deg and the pelvis is at z 0.010 m. The force is 0 at 0.60 s.
   At 1.0 s the tilt is 117 deg, the pelvis is at z -1.83 m, and the
   vertical force is 1814 N on the left heel. The larger damping does
   not keep a stance, and the first strike is larger. Do not repeat
   this damping change. At the first sample of the still freeze, with
   the saved damping, the left ankle is at z -0.865495 m and the right
   ankle is at z -0.882753 m, so the left ankle is 17.3 mm higher. The
   pelvis up vector is [0, -0.382683, 0.92388]. The ten contact forces
   are 0 except the right heel at 3180 N. Do not raise the ground into
   that heel. The pelvis up axis is world up rotated 22.5 deg about
   the facing axis. The saved hip starts are X -30.3159 deg at
   -13.1026 deg/s, Y 41.0041 deg at -12.0027 deg/s, and Z 131.077 deg
   at 10.7556 deg/s. None is 22.5 deg, so the level check did not
   simulate (EXIT 0). The 22.5 deg tilt is the composed pelvis up
   axis, not one start parameter. Do not zero one of these angles.
   Do not change FootContactStiffness. Do not save. Do not write the
   shift into the model. Do not claim the 5 % bar. Do not rerun
   1.809 s.
4. **#11157 grip/wrists**, **#11159 less spine motion**: not started; both are
   bands like #11156 (add rows to `gs3dx_golf_rom`, one acceptance test each).
5. **#11158 follow-up**: `human_references_stay_in_the_human_range` passed
   (EXIT 0, 2026-09-30) after `GS3DX_Human` references were rewritten.
   The range-limited IK (rom weight 3, every 10th frame to impact) is
   37.7 mm RMS versus 23.3 mm free. Bounded rows were at most 0.71°
   outside and were projected onto the anatomical limit. Wrist arcs,
   which that penalty does not bound, were scaled to the anatomical span
   after interpolation. Lead and trail ankles were clamped onto their
   existing limits (lead dorsiflexion had been 2.29° over). The ROM table
   was not loosened. Block count stayed within 975.
6. Finish-phase trail-foot lift (see #10979 notes below). The law is
   `gs3dx_trail_load` (3/3, EXIT 0, 2026-09-30): after 1.35 s,
   `gs3dx_trail_preload` integrates `ki * max(preload - fz, 0)` and
   `gs3dx_trail_foot_reference` subtracts that height from row 6 only.
   Defaults are 1.35 s, 0.1*80*9.81 N, and 1e-4 m/(N·s). `GS3DX_Human`
   compiles to 965 blocks. The 975 cap leaves the rest for the in-memory
   sensors in `gs3dx_contact_check`, so this loop is not added to the saved
   model. `BalanceFootRef` is unchanged.

## Objective and Status

- Objective: build agent-editable `GS3DX_` clones of `GolfSwing3D_Kinetic.slx` in
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/`,
  move shoulders and hip to quaternion joints, and grow to a full-body model under
  the 1,000 non-virtual-block Home-license limit, without touching the originals.
- Status: GS3DX-1 (#10951) scaffold, GS3DX-2 (#10952) regression harness, GS3DX-3
  (#10953) block budget, GS3DX-4 (#10954) `GS3DX_Slim`, GS3DX-5 (#10955) `GS3DX_Quat`
  quaternion shoulders, GS3DX-6 (#10956) quaternion hip, GS3DX-7 (#10957) lower body and
  GS3DX-8 (#10958) weld stance + integration run and GS3DX-9 (#10959) visual QA done;
  follow-up GS3DX-10 (#10979) drives the legs and refits the pelvis inputs; balance through the leg servo holds the body over its feet (section 7 of `docs/FIT.md`). GS3DX-11
  (#10985) data audit and GS3DX-12 (#10986) ground contact done (`GS3DX_FullBodyContact`).
  #11011 typical segment masses done (`GS3DX_Golfer`, 80 kg de Leva) plus the force-plate-free
  total GRF from the capture (`gs3dx_kinematic_grf`); owner: no force plates will be captured.
  #10979 in progress (owner: anthropometry unknown, match it to the data): `GS3DX_Fit` segment
  lengths from the capture, a least-squares whole-body IK, and the hand-on-grip geometry from the
  capture (`gs3dx_fit_grip`; out-of-sample RMS max 32 -> 17 mm).
- Completed:
  1. `gs3dx_setup` (session path, cache redirect, no `savepath`), `gs3dx_names`,
     `gs3dx_assert_no_shadowing`, `gs3dx_save_model` (write guard),
     `gs3dx_original_manifest`, `gs3dx_clone_baseline`.
  2. `models/GS3DX_Baseline.slx` + `GS3DX_KD_{Gimbal,Revolute,Universal}.slx` generated
     headlessly in R2025b; 12 `ReferencedSubsystem` blocks re-pointed.
  3. `tests/test_gs3dx_safety.m` 5/5 pass.
  4. #10952: `gs3dx_simulate`, `gs3dx_flatten_bus` (1 kHz grid), `gs3dx_compare`
     (`max_abs <= atol + rtol*range`, atol 1e-6, rtol 1e-3) and
     `gs3dx_capture_original_baseline`. Baselines per drive (`gs3dx_drive`,
     `gs3dx_baseline_file`): `original_GolfSwing3D_Kinetic_impact_0p3S.mat` (regression
     drive, 344 steps) and `..._persisted_0p3S.mat` (model's saved inputs, 4,236 steps).
     GS3DX_Baseline matches both on all 413 signals with identical step counts.
  5. #10953 (agy draft, reviewed and corrected): `gs3dx_block_budget`,
     `gs3dx_license_limit_probe`, `docs/BLOCK_BUDGET_FINDINGS.md`. Limit verified exactly:
     1,000 non-virtual blocks simulate, 1,001 fail at compile.
  6. #10954: `gs3dx_build_slim` copies Baseline/KD to `GS3DX_Slim`/`GS3DX_KDS_*` and
     `gs3dx_direct_torque_drive` rewires each torque axis converter -> joint InputTorque
     (drops Ideal Torque Source, Rotational Multibody Interface, Reference). 672 -> 609
     non-virtual blocks (63 saved). Slim matches the original on the impact drive
     (413/413 signals, 344 steps, clubhead max diff 6e-14 m).
  7. Found the persisted drive is ill-conditioned (clubhead > 4 km/s by 0.3 s; a 1e-9
     RelTol change moves the clubhead 2.4 m), so it cannot prove any restructuring;
     the impact drive is the regression drive. `docs/SENSITIVITY_FINDINGS.md`.
  8. #10955: `gs3dx_build_quat` copies Slim/KDS_Gimbal to `GS3DX_Quat`/`GS3DX_KDS_Spherical`;
     `gs3dx_gimbal_to_spherical` swaps the Gimbal Joint for a Spherical Joint keeping the
     subsystem interface (ports, mask, 21 bus elements) via `gs3dx_xyz_map` (Euler X-Y-Z <->
     quaternion; `XYZ Torque` and `XYZ Kinematics` MATLAB Function blocks, `Angle Reference`
     integrator for the 360-degree branch). 609 -> 599 blocks. Isolated rig
     (`gs3dx_joint_rig`): all 21 signals match the Gimbal to <= 2e-7 of peak at RelTol 1e-8.
     Full model: the Low-priority right shoulder assembles differently in a closed chain, so
     `gs3dx_pinned_drive` pins its start state; then Quat converges to Slim as RelTol tightens
     (clubhead gap 13.9 mm / 0.54 mm / 0.039 mm at 1e-3/1e-5/1e-7), is closer than Slim to the
     converged answer at every tolerance, and takes 7-9% fewer steps.
     `docs/QUATERNION_SHOULDERS.md`.
  9. Refactors: `gs3dx_physical_graph` + `gs3dx_redraw_nets` (shared net surgery),
     `gs3dx_copy_models` (copy guard), `gs3dx_simulate` `block_parameters` option,
     `gs3dx_repoint_references` skips subsystem-reference internals.
  10. #10956: `gs3dx_quaternion_swap` is the shared swap core (port map per joint type);
      `gs3dx_gimbal_to_spherical` and `gs3dx_bushing_to_6dof` are thin specs over it.
      The hip Bushing Joint becomes a 6-DOF Joint (prismatics copied, rotation quaternion);
      `GS3DX_Quat` is 594 blocks (15 below Slim). `gs3dx_hip_rig` (+ `gs3dx_rig_scaffold`,
      `gs3dx_rig_simulate`, shared with the refactored joint rig): all 30 HipLogs signals match
      the Bushing to <= 1.9e-6 of peak. Convergence with the hip: 14.8 / 0.61 mm at 1e-3 / 1e-5.
  11. #10957/#10958: `gs3dx_build_lower_body` copies Quat to `GS3DX_FullBody` and adds a
      `Lower Body` subsystem from `gs3dx_leg_table` (de Leva anthropometry; Spherical hip,
      Revolute knee, Universal ankle KDS). `gs3dx_stance_frames` measures pelvis/shoulder frames
      on Quat at t0 (unsaved) to lay out a leg frame; feet are framed rigidly to World (weld).
      751 blocks (cap 900). Starts in Quat's exact state, knees assemble at -28.14 deg, 0.3 s
      impact window completes (348 steps); passive legs make the pelvis diverge and the right
      knee hyperextend (+67 deg). `gs3dx_contact_trial`: contact = +2 blocks, 1.8x wall time.
      `docs/FULL_BODY.md`. `gs3dx_layout_qa` renders diagrams to `docs/screenshots/` (0 overlaps).
  12. #10985 data audit: `gs3dx_capture_stance` reads `data/C3D_TA_Driver.c3d` via pyenv ezc3d
      (654 frames, 360 Hz, **no force plates / analog: no GRF data**); stance 0.65 m ankle width,
      foot yaw -2.15/+8.15 deg -> `gs3dx_leg_table` `.stance`. Mass double-count (upper body 77.6 kg,
      FullBody 109.4 kg), foot/thigh proxy conflicts, and the impact drive's pelvis path being out
      of reach of planted feet: `docs/DATA_AUDIT.md` (owner decision on trunk mass pending).
  13. #10986 ground contact: `gs3dx_leg_fk` (matches Simscape to 1e-9) and `gs3dx_leg_ik` (damped
      Gauss-Newton); `gs3dx_build_contact` copies FullBody to `GS3DX_FullBodyContact`: welds removed,
      3 sole spheres per foot on one Infinite Plane, pelvis joint unactuated (NoTorque, 4 converters
      deleted), stance-hold leg servo (matrix Gain on [q; qd] + existing Constant), IK start angles and
      rates (High). `gs3dx_contact_check`: Newton momentum balance, foot slip/lift, GRF. From rest it
      stands (slip 2 mm, no lift, Newton 0.71/3.22 N*s); the impact drive tips it over the right foot
      (131 N*s start momentum, gain-independent) -> needs the #10979 refit. `docs/GROUND_CONTACT.md`.
  14. **Block budget correction:** the Home license counts the _compiled_ model; FullBody is 751
      uncompiled but 945 compiled (Quat 594 -> 740). `gs3dx_block_budget(mdl, compiled=true)` adds
      `.compiled_total`; FullBodyContact compiles to 967 with a 25-block validation reserve.
  15. #11011 typical masses: `gs3dx_anthropometry` (one de Leva table: masses + COM fractions;
      `gs3dx_leg_table` now reads its leg masses from it, values unchanged); `gs3dx_build_golfer`
      copies FullBodyContact to `GS3DX_Golfer`, sets 15 upper-body solids to `Golfer*` kg variables
      (block count unchanged, 967 compiled). Sensed mass 80.393 kg = 80 + 0.393 equipment (was 109.4).
      From rest it stands (Newton 0.46/2.37 N\*s, slip 1.4 mm, no lift); the impact drive still tips it
      (start momentum, and the drive torques were fitted to the 77.6 kg upper body).
      `gs3dx_capture_markers` (C3D loader, club clusters, impact frame, address target frame; shared
      with `gs3dx_capture_stance`); `gs3dx_kinematic_grf`: total GRF = sum m a - (M+m_club) g from
      the capture COM: address 1.001 BW, pre-impact peak 1.24-1.33 BW ~60 ms before impact (6-15 Hz),
      club must be included (+0.57 BW). Plan without force plates: `docs/ANTHROPOMETRY.md`.
  16. #10979 anthropometry from the data (`docs/FIT.md`): `gs3dx_capture_joint_centres` (joint
      centres + segment lengths from the markers; RShoulderTop rebuilt from RShoulderBack),
      `gs3dx_fit_lengths` + `gs3dx_build_fit` -> `GS3DX_Fit` (11 upper-body cylinders re-pointed at
      `Fit*` vars the drive file cannot overwrite, ThighLength/ShankLength set; block count unchanged,
      967 compiled; KinematicsSolver FK: forearm 0.280, pelvis-hub 0.490 vs data 0.278/0.488).
      `gs3dx_whole_body_ik`: KinematicsSolver refuses over-determined targets (status -3), so
      lsqnonlin LM over 33 independent coordinates with the solver as FK (5 ms, right arm closed by
      the grip weld loop); per-body marker offsets calibrated by alternation; forward+backward
      tracking; gap-filled samples (`jc.gap`: club head 445, 519-551, 653-654) are dropped - fitting
      them lost the whole follow-through (0.5 m). Whole trial (63 min): RMS 7.1 mm median, 20.3 mm
      max. Out of sample (offsets from the backswing, fit the last 0.25 s): pelvis/legs/shoulders
      within 1-3 cm, club head 40 mm, wrists 52/88 mm -> the model's hand-on-grip geometry (literal
      grip cylinder lengths, hand sphere radius 2 in, standoffs) is the next thing to fit. Frame
      `Rotation` outputs = intrinsic XYZ (verified 3e-16).
  17. #10979 hand-on-grip geometry (`docs/FIT.md` section 4): `gs3dx_fit_grip` (Kabsch club pose
      from the six club markers, sphere-fit functional wrist centres of WristTop in the club frame,
      shaft axis from the IK model's hand-sphere centres). The original grip put both wrists 1 in to
      the same side; the data has them 77 mm apart along the grip and 72-74 mm apart across it, on
      opposite sides. The split across the shaft is not identifiable (the club-head marker offset
      absorbs a sideways shaft shift), so it is split equally. `gs3dx_build_fit` re-points the 6 grip
      solids at `FitButtToLeadHand` 3.08, `FitHandSpacing` 3.03, `FitGripToShaft` 4.39 (butt to
      shaft stays 10.5), `FitLeft/RightWristStandoff` 1.46 in, and flips the lead standoff by swapping
      its frames' end features (parameter-only, 967 compiled). Out of sample: wrists 88/52 -> 18/15 mm
      median, club head 40 -> 29, RMS max 32 -> 17 mm, wrist marker offsets 6-7 -> 2.8 cm; the grip
      estimate is a fixed point within 0.07 in. `ik.model` now records the fitted model. Whole trial on
      the fitted grip (43 min): RMS 7.0 median / 16.1 p95 / 19.1 mm max, offsets <= 47 mm (scratch
      `ik_full_grip.mat`, not committed; regenerate with `gs3dx_whole_body_ik(jc, verbose=true)`).
  18. #10979 leg servo references (`docs/FIT.md` section 5): `gs3dx_leg_reference(ik, jc, cap)` -> 12
      servo angles/rates from the IK pelvis path (joint j1 follower = 'Lower Torso') and the measured
      foot path (ankle centre + ankle/ToeIn/ToeOut triad rotation; not planted: trail heel +36 mm at
      impact), feet levelled +-9.3 mm, 10 Hz Butterworth, a constant torsion per foot (+24.0/-17.9 deg,
      the universal ankle has no axial DOF), exact `gs3dx_leg_ik`, reach clamp <= 1.5 mm (post impact).
      Knees vs capture to impact 19/20 mm median, 48/55 mm at impact (the missing ankle axial DOF).
      `gs3dx_build_fit_legs` -> `GS3DX_FitLegs`: Leg Torque Commands Constant -> From Workspace
      (one for one, 967 compiled), start state + `LegReferenceStart` struct (pass it after the drive:
      the drive file sets PlaneTilt 22.5 vs the saved 30 deg, which tilted the start pelvis 7.5 deg),
      ground under the start feet (capture frame = World). Standing from rest: slip 1.8/1.5 mm, lift 0,
      Newton pass. The impact drive is PASSIVE (ModelingMode 0 zeroes every upper-body torque).
  19. #10979 upper-body tracking (`docs/FIT.md` sections 3 and 6). Regularized whole-body IK options
      (`posture_weight`, `smooth_weight`, `backward`, `gap_weight`; defaults off, test_gs3dx_fit 7/7):
      the unregularized trunk jumped branches (torso 464 deg range); with 0.01/0.02/false/0.5 the whole
      trial fits 6.3 mm RMS median (25.2 p95), pelvis step <= 1.3 mm to impact (scratch
      `ik_full_reg.mat`, ~1 h). `GS3DX_FitLegs` REBUILT from it (torsion +23.7/-16.8, clamp 1.6 mm,
      knees 27/32 mm median). `gs3dx_upper_body_reference` -> 12 chart references (12 Hz).
      `gs3dx_build_fit_track` -> `GS3DX_FitTrack`: each upper-body chart gets feedforward + PD
      (`gs3dx_track_torque`, `UpperBodyTracking`, ModelingMode stays 0 = free pelvis), inertia-scaled
      gains (`gs3dx_track_gains`, 6 Hz, zeta 1), and REWIRED inputs: the original LW chart reads the
      LEFT SCAPULA angles and LE/LF/RF/RW read tags no Goto writes (wrists ran away 14,800 deg). No
      block added (967). `gs3dx_track_learn` (ILC from the Simscape log; signal logging passes the
      license limit): full swing to impact angle RMS 0.96 -> 0.25 deg in one update, but PD RMS
      22.1/10.3/14.7/20.7 N*m grows from iteration 3 (Q-filtering all of F did not help; torso keeps
      34 N*m at 0.19 deg); the model holds the iteration-2 feedforward. Whole body to impact: pelvis
      206 mm RMS / 502 mm at impact off the capture (steady X drift, feet slide 8 cm, lift 9/20 cm:
      the body TIPS OVER ITS FEET - joint tracking alone does not balance); vertical GRF 0.20 BW RMS
      from the capture, peak 1.43 BW is the landing at 0.05 s. 17 min per full-swing simulation.
  20. #10979 balance (`docs/FIT.md` section 7): `gs3dx_build_fit_balance` -> `GS3DX_FitBalance`
      (973 compiled; contact-check sensors +10 = 983). Whole-mechanism Inertia Sensor -> Goto ->
      State-Space rate; `Leg Torque Commands` -> MATLAB Function `gs3dx_balance_command`: servo
      angle + G(t) shift (shift = -Kp e - Kd de, 3-axis COM error, limit 0.1 m) + per-leg G(t) kf
      e_foot (ankle GlobalPosition via Bus Selectors vs `ref.feet`). `gs3dx_balance_gain` = foot-fixed
      damped inverse leg Jacobian (12 x 3 x frames). To impact: pelvis off the capture at impact 503 -> 62-81 mm (FIT.md
      table; 3-axis + foot gain 1: pelvis 44 RMS / 81 mm at impact, COM 38.5/52.0 mm). Gains raised
      to Kp 3 / Kd 0.4 (builder default): COM 17.9/25.7 mm, pelvis 27/73 mm. Support peak 2.4-2.7 BW
      just before impact (capture 1.27) is DEMANDED BY THE COM REFERENCE (pelvis ref + balance-off
      offset implies 2.31 BW; 40 mm vertical range vs capture 19 mm; model COM within 6 mm of it) -
      not a foot defect: per-contact forces (`gs3dx_contact_check` `.contacts`) show a broad hump
      1.23-1.31 s, trail foot on its inner toe sphere. Rejected: ankle servo 10 N\*m/deg (falls, pelvis
      906 mm at impact); foot-tilt feedback via ankle `Rotation Transform` (worse COM 49/75 mm; patch
      kept in the session scratchpad only). The model is built with axes 3, gains [3 0.4], foot 1.
      Earlier "zero logged samples" errors were very likely a FULL C: DRIVE (Simulink turns off
      recording under low disk space), not only concurrency.
  21. #10979 references and inertia (2026-09-28): `com_ref` option of `gs3dx_build_fit_balance`
      (World path, exclusive with `com_offset`) + `gs3dx_capture_com_reference` (capture COM from
      `gs3dx_kinematic_grf`, translated to the model's at address). Trial, not saved: support peak
      1.44 BW (was 2.63; capture 1.27), slip 44/53 mm (58/88), pelvis 42/72 mm (27/73) - the spike
      is the reference, confirmed. `gs3dx_inertia_audit` + `gs3dx_segment_inertia` + de Leva
      `.gyration`/`.length` in `gs3dx_anthropometry`; `docs/INERTIA.md`: limb longitudinal inertia
      0.43-0.65 of de Leva, lower trunk 0.53, foot close (1.08). Agents (agy, uncommitted worktrees
      `agy-gs3dx-{shape,render}`): GS3DX_Shape (ellipsoid solids with custom de Leva inertia/COM)
      and `gs3dx_render` (headless KinematicsSolver + MATLAB graphics) in progress.
  22. #10979 shape and rendering (2026-09-28): `gs3dx_render` (headless stills/video from a
      KinematicsSolver pose of every Solid; `docs/RENDERING.md`; stills in
      `docs/screenshots/GS3DX_{FitBalance,Shape}_*.png`). `gs3dx_build_shape` -> `GS3DX_Shape`:
      custom de Leva moments on limbs and head (audit 1.00), de Leva centres of mass on thighs,
      shanks, upper arms and forearm halves (joints measured on each solid's z axis, proximal +z),
      ellipsoids for thighs/shanks/hands/head via `PortConnectivity` rewiring; 973 compiled (one
      extra visual solid costs 7). `gs3dx_balance_reference` shared with FitBalance. Finding: the
      Shape COM reference is within 1 mm RMS of FitBalance's (12.4 vs 12.9 mm from the capture's)
      and balance runs repeat it (2.64 BW offset, 1.44 BW capture COM): segment inertia does not
      cause the impact spike (`docs/SHAPE.md`). The earlier lower-trunk 0.53 ratio was an audit
      length artifact (`docs/INERTIA.md`).
  23. #10979 render views (2026-09-28): `gs3dx_render`'s "face-on" drew down the line and
      "down-the-line" drew face-on from behind; now face-on = camera on +X (az 90), down-the-line =
      camera on -Y (az 0), unknown views error, `out.view` reported and tested. All
      `docs/screenshots/GS3DX_*` stills re-rendered.
  24. #10979 impact spike resolved (2026-09-28): per-segment vertical COM, model (regularised IK,
      `gs3dx_render` poses) vs capture, agrees to 1 mm for legs/arms/club; the 12.9 mm RMS is the
      trunk (10.3) and head (3.5). The trunk part is the capture's C7-skin proxy: with the
      shoulder-centre -> hip-centre trunk (`gs3dx_kinematic_grf(trunk="joint_centres")`) the
      capture matches the model to 4.0 mm RMS. As balance reference it gives 1.77 BW peak (vs
      2.64 offset / 1.44 C7), pelvis 22 mm RMS, COM vert 2.9 mm, slip 38/16 mm: best run, and
      `GS3DX_Shape` is saved on it (`docs/SHAPE.md`). The head is rigid with the upper trunk
      (no neck joint, no head IK target) and rises ~50 mm in the downswing; a driven neck needs
      ~10 compiled blocks and Shape has 2 under the 975 cap.
  25. #10979 neck (2026-09-28): `GS3DX_Neck` (`gs3dx_build_neck`, `docs/NECK.md`) = Shape + a
      Universal Joint between Rigid Transform5 and the Neck's "Bottom of Neck" frame (the neck
      base), both axes input motion from `NeckReference` (capture head frame relative to the model
      upper trunk; axial turn dropped). 6 blocks, paid by deleting the 4 massless elbow/shoulder
      spheres (1 compiled block each, not ~7): 975. Head vertical travel to impact 213 -> 113 mm
      (capture 55), vertical error 52 -> 37 mm RMS; facing error is the trunk's. Adding a joint
      renumbers KinematicsSolver IDs (block-path order), so `gs3dx_render` now matches joints by
      `gs3dx_joint_keys` (block path + primitive); IK/fit helpers still hard-code j15/j18/j19 and
      are only used on neck-free variants. agy's `gs3dx_capture_head_frame` +
      `gs3dx_capture_address_transform` ported (reviewed).
  26. #10979 human shape (2026-09-28/29): `GS3DX_Human` (`gs3dx_build_human`, `docs/HUMAN.md`,
      965 compiled; current state in Next Steps item 2) = Neck with: the unused "Inertia Sensor" subsystem removed (-75); trunk,
      neck, shoulder, arm, forearm cylinders and the misaligned `ZeroMassHipReference` bar hidden
      (inertia kept) and massless ellipsoids drawn on each parent's exposed reference frame (a
      cylinder with custom frames shows only its end frames); head radii 90x90x105 mm;
      `NeckAddress` (-24.2, -7.0 deg) aims the neck at the capture head (address error 126 ->
      40 mm); `FaceSquareRoll` -20.73 deg squares the face at address (driver-head visual,
      10.5 deg loft; still 14 deg open at impact from the posed hands); sprung revolute
      midfoot (MTP) joints with a 0.25 kg forefoot carrying the toe contact spheres, foot COM
      preserved. `gs3dx_reference_pose` renders with the servo leg reference (IK feet were
      reversed: no toe target). Tests: human 5/5, neck 5/5, render 4/4.

## Files and Decisions

- The originals are only `copyfile`d, never loaded or saved, so R2025b cannot re-save them
  (the original .slx was last saved in R2025a).
- The exploratory folder sits outside `matlab/src/`, so `setup_matlab_environment`'s
  `genpath(src)` never adds it to the path.
- Shadowing risk is real: during the probe, a same-named scratch copy won `which()`.
- The baseline is stored in double precision: float32 rounding alone (1.9e-6 on the
  constant GolferMass) exceeded atol; the tolerance was not widened.
- The model needs `matlab/src/functions` on the path (`HexPolyInputFunction`), so
  `gs3dx_setup` adds it as `dependency_dirs`.
- Per-joint cost: KD Gimbal 37 non-virtual blocks (21 converters), Universal 29 (17),
  Revolute 21 (13). Each torque axis uses Simulink-PS -> Ideal Torque Source ->
  Rotational Multibody Interface + Reference (4 blocks) although the joint axis is
  already `InputTorque`; a direct PS torque input saves 3 blocks per axis (63 total).
- Simulink physical lines: a branched net lists its segments only on the trunk
  (`LineChildren`, which includes the trunk itself); branch segments report no parent or
  end ports. `gs3dx_direct_torque_drive` finds nets by union-find over all segments.
- Count blocks on a freshly loaded model: after a simulation in the same session
  `find_system` counts differ (+101 on Slim).
- The KDS Bus Creator is named `Bus
Creator` (newline); find it by BlockType.
- A MATLAB Function block is direct feedthrough, and the sensed acceleration depends on the
  torque, so torque and kinematics mapping are two blocks (one would form an algebraic loop).
  The damping Constant needs SampleTime 0: constant-time blocks may not touch the shoulder
  input-function loop.
- A Spherical Joint has one target per quantity: the mask init asserts equal Rx/Ry/Rz
  priorities, so set all six priorities in one `set_param` call.
- Impact-input mode sweep: mode 3 ill-conditioned, mode 2 torqued but 1e8 N\*m, mode 1 fails
  early. Not gimbal lock: Quat (quaternion hip) fails at 19.9 ms with hip Y at 49° while the
  mode-1 hip torque command reaches ~2e7 N\*m. The drive blows up, not the joint.
- `gs3dx_physical_graph` on a subsystem: `find_system` lists the subsystem itself; exclude it.
- Rig constant signals are logged once; `gs3dx_rig_simulate` expands them to the output grid.
- The hip rig's default torques keep the Bushing's Y angle within -72..62°; 6x larger
  torques reach 84° (near singular, 1.8e4 deg/s).
- Welded legs close a loop through the pelvis joint: Simscape ignores targets when every joint
  in a loop has one, so the leg hips have no target (knee/ankle Low pick the branch).
- A freshly added Subsystem Reference does not list its contents to `find_system`; resolve
  ports from the referenced file (`gs3dx_port_source`).
- Infinite Plane ports: frame on the left, geometry on the right; a Brick with exported
  geometry is the other way round.
- `*.png` is git-ignored outside the root `docs/`; the exploratory screenshots are force-added.
- A name on a branched signal line is drawn on every branch; name signals on the outputs of
  a virtual Demux instead (`Actuator Torque` in the hip/shoulder subsystems).
- The hip bus element is `AngularPosition Z` (space) in the bus; logged Datasets expose it as
  `AngularPosition_Z`.
- Deleting one line of a physical net deletes the whole net: removing the foot weld also cut
  ankle Distal -> Foot COM, which `gs3dx_build_contact` redraws and asserts.
- `gs3dx_stance_frames` closes the model it simulates: read model-workspace values first.
- The impact drive file sets `LowerTorsoMass`, `UpperTorsoMass`, `*ShoulderMass`, `*ArmMass` (unused
  by any solid); the new masses use `Golfer*` names so a drive can never overwrite them.
- A scratch `poly.m` in the MATLAB run folder shadowed the built-in (broke `butter`); renamed.
- `gs3dx_build_lower_body`/FullBody tests still cap the _uncompiled_ count at 900; FullBody's
  compiled count (945) would fail that cap. Left as is (existing test); flagged in the PR.

## Validation

- From an empty cwd: `matlab.exe -batch "addpath('<exploratory_gs3dx>'); info=gs3dx_setup(); runtests(fullfile(info.root,'tests'))"`
  → 85 passed / 0 failed (2026-09-27, with test_gs3dx_fit 7/7 incl. the fitted-grip out-of-sample
  IK and the grip fixed point; the IK test takes ~5 min). Earlier 83 (test_gs3dx_fit 5).
  Plus `runtests('test_gs3dx_fit_legs')` → 4 passed / 0 failed (2026-09-27; setup runs a
  whole-body IK, 11–25 min; clamp 0.0 mm, knee median 25/31 mm, p95 58/61 mm; standing slip
  1.8/1.5 mm, lift 0).
  Plus (2026-09-28, one MATLAB process) `test_gs3dx_fit_balance` 6/6 (leak max 0.975 / median
  0.303, bounds 0.98 / 0.31; 0.3 s COM error 14.6 RMS / 24.9 max mm, bound 26 mm),
  `test_gs3dx_fit_track` 5/5, `test_gs3dx_leg_kinematics` 4/4, `test_gs3dx_contact` 7/7.
  Plus `runtests('test_gs3dx_fit_track')` -> 5 passed / 0 failed (2026-09-27; 0.3 s replay 0.283 deg,
  worst joint 0.491 deg, PD 16.7 N\*m) and, after the FitLegs rebuild, `test_gs3dx_fit_legs` 3/4 + the
  standing test rerun alone 1/1 (slip 1.4/0.9 mm, lift 0, pelvis 28 mm). Run alongside another
  MATLAB simulation the standing test errored (zero logged samples in `gs3dx_stance_frames`):
  do not run GS3DX simulations in parallel processes (shared cache). test_gs3dx_fit 7/7 after
  the IK options. The full suite was not rerun.
  Earlier 78 (after #11011: golfer 6, capture +2 kinematic GRF; earlier 70 after #10985/#10986: capture 4, leg kinematics 4, contact 7
  new; the FK test failed once on a closed-model handle, fixed and rerun 4/4). The capture tests
  need MATLAB pyenv with ezc3d (they are skipped otherwise).
- `gs3dx_layout_qa` on the rebuilt diagrams: 0 overlapping blocks.
- The original model simulates headlessly (0.3 s of swing takes about 223 s cold); the model workspace is embedded (676 vars).

## Blockers and Risks

- MATLAB tests need a licensed R2025b; they are not part of the Python CI.

## Next Steps

1. Owner review of draft PR #10963; mark it ready once reviewed (a GUI open-check via
   computer use needs the owner to grant app access interactively).
2. #10979 human, in progress (2026-09-29, owner reviews of the renders). State:

   - **Done, tested (human 9/9, render 5/5):** five
     contacts per foot; fixed "Neck Address" transform; neck pivot at C7 (`NeckLength`
     7.717 in; lift is `-lift*Q0(:,3)`); driver head = massless File Solid of
     `models/gs3dx_driver_head.stl` (Tools `rate_of_closure` Driver 10.5,
     `models/README_DRIVER_HEAD.md`; File Solid needs `UnitType Custom`); neck shape now
     runs head centre -> 50 mm past C7 and a "Trapezius" ellipsoid on UpperTorsoTop joins
     it to the shoulders (owner: neck gap; test `neck_joins_the_head_and_the_trunk`).
     965 compiled / 788 uncompiled. Renderer returns `out.focus` (bug found by the mesh
     test; fixed).
   - **Start kick FIXED:** bisected with scratch `human/bisect/bisect_human.m` (Neck copy +
     builder stages). Not the feet/midfoot/foot inertia: the neck address moved the address
     COM (-4.7, -0.9, +5.2) mm while `BalanceCOMRef` was anchored at Neck's COM, so the
     balance loop kicked both feet off the ground at t = 0. `gs3dx_build_human` now
     re-anchors `BalanceCOMRef` after saving (1 ms balance-off `gs3dx_contact_check`;
     `report.com_shift`; test `balance_reference_is_anchored_at_this_body`). Support now
     0.46-1.97 BW (was 0-10.3).
   - **Balance drift FIXED (2026-09-29):** the foot layout, not the midfoot or the head. A
     Neck-feet variant with the same head + anchor drifts 18.0 mm; the first five-contact
     layout 42.9 mm even locked (159 mm at 100 N\*m/rad). COP (scratch `human/cop.m`): both
     models stand on the front contacts, lead foot on its inside front corner, and the
     Human's front support stopped short of the toe tip. Layout sweep (scratch
     `human/layout1.m`, `layout2.m`; table in `docs/HUMAN.md#why-the-human-drifted`) ->
     built layout: one heel on the axis, balls unchanged, Big Toe under the tip 50 mm
     inside, Lesser Toes at 0.945 of the foot length 50 mm outside, both on the forefoot.
     Built model (scratch `human/test9.m`): pelvis 20.2 mm RMS locked, 21.9 mm at 2,000
     N\*m/rad (COM 14.0, support 0.52-2.02 BW); `midfoot_stiffness` default now 2,000
     (100/800 fold). Human 9/9 + render 5/5 pass on the rebuilt model; human 10/10 with the simulated-render test (2026-09-29).
   - **Trap:** a failed `gs3dx_build_human` leaves `GS3DX_Human.slx` as a copy of
     `GS3DX_Neck`; check size/mtime before trusting a run. `gs3dx_render` and
     `gs3dx_contact_check` close the models they load: tests reload after them.
   - **Finish (2026-09-29, scratch `human/fin1.m`, `fin2.m`):** all workspace references
     span 654 frames (1.814 s), so `gs3dx_contact_check(..., stop_time=1.809)` runs through
     the finish. Trail heel up onto Big Toe then Lesser Toes (ankle rise 142 mm vs capture
     129), lead foot onto its outside edge: works. OPEN: at 1.55-1.6 s the trail toe carries 0.73-0.83 BW with the lead foot
     at 0, and the unloaded lead foot slides 234 mm (capture 36 mm, turns 30 deg in place);
     pelvis 56.6 mm RMS after impact. NOT the COM: along trail ankle (0) -> lead ankle (1)
     the model COM ends at 0.60, its reference 0.58, the capture's 0.54 (scratch
     `human/fin5.m`; TRAP: `gs3dx_kinematic_grf` .com is in target axes with the LAB
     origin: subtract `tf.S.' * tf.origin` (`gs3dx_capture_address_transform`) before
     comparing with joint centres, which use the address-waist origin). Neck and Shape do
     the same and worse (support 0 near 1.4 s, trail slide 0.55 m; scratch `fin4.m`). The
     trail foot runs 60-105 mm ahead of its reference from 1.45 s. REJECTED hypothesis
     (scratch `human/fin6.m`): the stiff midfoot is not the cause. Trail overshoot 113-123
     mm and ankle rise 142 mm at MidfootStiffness 300 and 1e4 alike; K only trades balance
     (pelvis pre/post impact 37.8/72.8 mm at 300, 20.5/59.7 at 1e4). Next suspects: foot
     feedback gains, the servo references through the pivot, contact friction.
   - **Finish diagnosis continued (2026-09-29, scratch `human/fin8.m`-`fin13.m`):**
     `BalanceFootKp` sweep 0/0.5/1/2: pelvis post-impact RMS 54.2/52.7/56.6/68.5 mm, lead
     slide 78/114/236/87 mm, trail overshoot 160/100/100/190 mm, support min
     0.05/0.23/0.25/0 BW. kf 0.5 is best; the foot term causes the lead slide but not the
     trail overshoot. In EVERY run the trail foot is unloaded at 1.40-1.45 s and lands
     ahead. Contact trace (`fin10.m`): from 1.27 s the trail foot stands on the Big Toe
     sphere alone and the midfoot stays within 2 deg of flat. Toe-drive prototype
     (`fin11.m`, in memory only, 970 compiled blocks): trail reference re-pitched so the
     ball line reaches the floor (up on the toes past 1.4 s) + midfoot torque input K
     theta_ref; the toe angle tracks, the lead foot improves (slip 57 mm), but the trail
     foot still lifts at 1.40 s and slides 340 mm -> not the foot. Joint trace (`fin12.m`):
     the trail KNEE flexes 5-9 deg past its reference at 1.40-1.43 s; with LegServoKp 100
     N\*m/deg that is the balance command, not lag. Hypothesis: the vertical axis of the
     COM loop (BalanceGain 12x3) commands a pelvis drop that the loaded lead leg resists,
     so the light trail leg folds and lifts its foot. REJECTED by `fin13.m`
     (BalanceGain(:, 1:2, :), horizontal only): kf 1 19.3/65.8 mm, kf 0.5 18.0/63.1 mm,
     lead slip 40 mm but trail overshoot about 180 mm and the trail ankle still 18-24 mm
     above its reference at 1.40-1.45 s. `fin14.m` found the likely cause: the REFERENCE
     foot poses penetrate the floor (floor = mean address sole). Lead heel sphere -21 mm at
     1.35 s, -30 at 1.40 s, -37 at 1.70 s; lead BallOut -10 to -13 mm; trail Big Toe -2 to
     -12 mm from 1.2 s and Lesser Toes -22 mm by 1.7 s. The stiff servos drive the lead
     heel into the ground, the reaction lifts the body and the light trail foot leaves the
     floor. REJECTED by `fin15.m`: each reference foot raised so its lowest sphere sits on
     the floor (lead up to 37.7 mm, 14.4 mm at impact; trail up to 25.3 mm), leg IK, rates
     and BalanceGain recomputed. kf 1: pelvis 19.7/116.4 mm; kf 0.5: 18.8/110.0 mm. In both
     the trail foot is unloaded from 1.45 s to the end and dragged 1.1 m. The small trail
     penetration was the trail foot's only preload; the joint servos track angles relative
     to the pelvis, so a lightly loaded trail foot follows every pelvis error. The captures
     have no force plates, so a per-foot load target cannot come from data. REJECTED by
     `fin16.m` (committed reference, kf 0.5, trail-leg LegServoKp/Kd rows 7-12 scaled):
     x0.5 gives 28.9/78.1 mm (the trail foot relands and carries 0.3-0.4 BW through the
     finish but still lifts at 1.45 s); x0.25 collapses (121.9/595.1 mm). Best remains the
     committed model at kf 0.5 (20.4/52.7 mm). Next idea: a trail-foot vertical load loop
     (contact-force sensor -> integral on the trail reference height, minimum preload
     ~0.1 BW after 1.35 s), within the 975-block reserve.
   - **Source anonymization (2026-09-29, owner request):** a read-only audit found the
     capture and its vendor highly traceable on public main (raw files, vendor metadata,
     derived TRC, docs). Resolved 2026-09-29/30: history scrubbed, private data moved to
     a private data repo resolved through `CAPTURE_DATA_DIR`; pushes resumed. Still: no
     vendor/source names in new commits, PR text or docs.
   - **Simulated-motion render (2026-09-29):** `gs3dx_contact_check(..., joints=dt)` logs
     Simscape (decimation 10) and returns `.joints` via new `tools/gs3dx_simlog_joints.m`
     (a spherical joint's logged quaternion `S.Q` becomes `S.ax_x`/`ax_y`/`ax_z`/`q`),
     which `gs3dx_render` draws. Test `simulated_joints_render_the_simulated_body` checks
     the drawn pelvis stays fixed in the simulated pelvis frame. Finish videos: scratch
     `human/fin7.m` -> `human/sim_finish_<view>.mp4`; copies on the owner's Desktop in
     `GS3DX Swing Progress 2026-09-29`.

   - **Range of motion (#11158, 2026-09-30):** `gs3dx_joint_rom` (table), `gs3dx_rom_check`,
     `gs3dx_rom_from_ik`, `gs3dx_rom_reference`, `docs/ROM.md`. `gs3dx_whole_body_ik(...,
rom_weight=W)` applies the range penalty by continuation (free chain for the warm
     starts, then a penalized polish per frame). Every 10th frame to impact: 23.3 mm free,
     37.7 mm with weight 3 and every bounded range within 0.71 deg.
     `test_gs3dx_joint_rom`: `human_references_stay_in_the_human_range` passed
     (EXIT 0, 2026-09-30) after the reference rewrite. See item 5 above.
   - **Forward-dynamics goal (#11173, 2026-09-30):** `docs/FORWARD_DYNAMICS.md` is the
     playbook (ladder L0-L4, every lesson with evidence, convergence ideas, roadmap).
     Step 1 tooling: `gs3dx_feedback_torque` (pure; `test_gs3dx_feedback_torque` 5/5
     pass), `gs3dx_contact_check(..., feedback=true)`, `gs3dx_track_state` and
     `gs3dx_track_blocks` (shared with `gs3dx_track_learn`). The Human test
     `feedback_torque_is_measured_on_every_driven_axis` passes (2026-09-30, fifth run).
     The fourth run exposed that Simscape numbers joints in block-path order, so
     GS3DX_Human's neck renumbers every joint after it: `gs3dx_upper_body_joints` now
     carries each joint's `.block` path and `gs3dx_track_blocks` matches on it, not on
     GS3DX_Fit's ids. First 0.06 s RMS feedback: upper 30.5, legs 135.5, balance
     172.5 N\*m; the full-swing table goes into docs/FORWARD_DYNAMICS.md.
   - **Batch MATLAB:** scripts end with `fprintf("GS3DX_BATCH_DONE\n"); quit(0, "force")`;
     the scratch lock runner kills its own MATLAB 90 s after the marker (exit hangs held
     the lock 10-15 min).

3. #10979 learning drift: per-joint PD torque over more iterations (`out.joint_pd`), then a
   forgetting factor or PD-only loop joints.
4. Optional owner inputs: the golfer's height/mass (mass is not identifiable from markers).
5. Reproducibility: three inputs exist only in model workspaces (FitBalance `com_offset`,
   Shape `com0`, Neck `NeckReference`); add a tool per input and save the values beside
   the models so `docs/DESIGN_REPORT.md#reproducing-the-model` runs end to end.
