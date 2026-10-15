# Primary Data Extraction for the Segmental Spine Epic

Issue: D-sorganization/UpstreamDrift#12737 (SPN-0b), part of epic #12735.
Extraction date: 2026-10-15.

## Scope and Rules

- A table carries `verified_from_primary: true` only when the number was read
  in the primary publication text (full text or abstract of the paper itself).
  Numbers known only from reviews, citing papers or abstracts of other works
  are `false`.
- Nothing here is estimated or interpolated. Missing items are listed under
  Owner Action: Primary PDFs Needed.
- Access channel: Wilke 1999 was read from the primary PDF text (extracted
  with `pdftotext`). Fujii 2007 and Bruno 2015 were read from the PMC HTML
  full text through a web-fetch tool that returns model-assisted extractions;
  the Fujii results text was extracted twice with identical numbers. Re-check
  those two against the PDF when convenient (see Owner Action).
- Each JSON block has the keys `source_id`, `table_id`,
  `verified_from_primary`, `rows`.

## Source Index

| source_id | Citation | DOI |
|---|---|---|
| bruno2015 | Bruno AG, Bouxsein ML, Anderson DE. Development and validation of a musculoskeletal model of the fully articulated thoracolumbar spine and rib cage. J Biomech Eng 137(8):081003 (2015). PMC5101035 | 10.1115/1.4030408 |
| pearcy1984 | Pearcy MJ, Tibrewal SB. Axial rotation and lateral bending in the normal lumbar spine measured by three-dimensional radiography. Spine 9(6):582-587 (1984) | not retrieved |
| fujii2007 | Fujii R, Sakaura H, Mukai Y, et al. Kinematics of the lumbar spine in trunk rotation: in vivo three-dimensional analysis using magnetic resonance imaging. Eur Spine J 16(11):1867-1874 (2007). PMC2223353 | not retrieved |
| wilke1999 | Wilke HJ, Neef P, Caimi M, Hoogland T, Claes LE. New in vivo measurements of pressures in the intervertebral disc in daily life. Spine 24(8):755-762 (1999) | not retrieved |
| hosea1990 | Hosea TM, Gatt CJ, Galli KM. Biomechanical analysis of the golfer's back. In: Cochran AJ, ed. Science and Golf. London: E and FN Spon; 1990 | none |
| lim2012 | Lim YT, Chow JW, Chae WS. Lumbar spinal loads and muscle activity during a golf swing. Sports Biomech 11(2):197-211 (2012) | 10.1080/14763141.2012.670662 |
| sugaya1999 | Sugaya H, Tsuchiya A, Moriya H. Low back injury in elite and professional golfers: an epidemiologic and radiographic study. In: Farrally MR, Cochran AJ, eds. Science and Golf III. Human Kinetics; 1999 | none |
| wu2002 | Wu G, Siegler S, Allard P, et al. ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion, part I: ankle, hip, and spine. J Biomech 35(4):543-548 (2002) | not retrieved |

Author lists for fujii2007 and wu2002 beyond the first author are from
memory and must be confirmed against the PDFs; the titles, venues, volumes and
pages were confirmed from search metadata or text.

Correction to the brief: "Sutcliffe et al. 2008" crunch factor could not be
found. The crunch factor is attributed in the literature to Sugaya et al.
(1997/1999) and was tested by Cole and Grimshaw (2014, Spine J). Wu "2005"
Part II is the upper limb; the spine recommendation is the 2002 Part I paper.

## Bruno 2015

Source key: `bruno2015`.

### Table B1: Regional Motion Split (Main Text, Section 2.1)

Share of the spine contribution to total trunk motion, lumbar versus thoracic.
Cited by the paper to White and Panjabi, Wong, Fujimori 2013, Rozumalski
(lateral bending) and Fujii 2007, Fujimori 2012 (axial rotation).

| Axis | Lumbar (%) | Thoracic (%) |
|---|---|---|
| Flexion/extension | 51 | 49 |
| Lateral bending | 27 | 73 |
| Axial rotation | 17 | 83 |

Lumbopelvic ratio (lumbar flexion over pelvic flexion) falls from about 2.5 to
1.0 across trunk flexion (Supplemental Table 1; source Tafazzol et al.).

verified_from_primary: true (main text only)

```json
{"source_id":"bruno2015","table_id":"B1_regional_split","verified_from_primary":true,"rows":[
{"axis":"flexion_extension","lumbar_pct":51,"thoracic_pct":49},
{"axis":"lateral_bending","lumbar_pct":27,"thoracic_pct":73},
{"axis":"axial_rotation","lumbar_pct":17,"thoracic_pct":83},
{"axis":"lumbopelvic_ratio_flexion","start":2.5,"end":1.0}]}
```

### Table B2: Per-Joint Coupling Ratios T1-L5 (Supplemental Table 2)

Not obtained. The main text states that Supplemental Table 2 gives the
distribution of movement at each intervertebral joint, but the supplement
(ASME Supplementary_Material_BIO-14-1531.zip) is behind a bot check and the
per-level values are not in the main text. No numbers are reported.

verified_from_primary: false

```json
{"source_id":"bruno2015","table_id":"B2_per_joint_coupling_supplemental_table_2","verified_from_primary":false,"rows":[]}
```

### Table B3: Model Conventions Stated in the Main Text

| Item | Value |
|---|---|
| Intervertebral joints | Ball joints, L5/S1 to T1/T2, centre at disc geometric centre |
| Head and neck | One lumped body on T1 by a ball joint |
| Hip | Pin joint, one DOF (pelvic tilt) |
| Costovertebral joints | Pin joints |
| T1-T12 Cobb angle | 50 deg (kyphosis) |
| L1-L5 Cobb angle | -43 deg (lordosis) |
| Muscle strength | 100 N/cm^2 (shoulder groups 140 N/cm^2) |

The paper gives no explicit axis or sign definitions beyond these. verified_from_primary: true

```json
{"source_id":"bruno2015","table_id":"B3_model_conventions","verified_from_primary":true,"rows":[
{"item":"t1_t12_cobb_deg","value":50},
{"item":"l1_l5_cobb_deg","value":-43},
{"item":"muscle_specific_tension_N_per_cm2","value":100},
{"item":"shoulder_muscle_specific_tension_N_per_cm2","value":140}]}
```

### Table B4: Validation Task Definitions (Bruno Table 3)

| Study | Subjects | Measure | Activities |
|---|---|---|---|
| Andersson | 3 F | L3/L4 IDP, T4 EMG | Lumbar flexion 10-50 deg, 10.2 kg per hand |
| Polga | 4 M, 2 F | T6-T7 to T10-T11 IDP | Standing, flexion, extension, rotation, lateral bending, weighted arm positions |
| Rohlmann | 2 M | L1 implant load | Standing, flexion, extension, rotation, lateral bending, arm elevation and abduction |
| Sato | 8 M | L4/L5 IDP | Standing, 30 deg flexion, 15 deg extension |
| Schultz | 3 F, 1 M | L3/L4 IDP, L1-L5 EMG | Standing, arm positions, 8 kg, 30 deg flexion |
| Takahashi | 3 M | L4/L5 IDP, L3 EMG | 0-30 deg flexion, with and without 5 kg per hand |
| Wilke | 1 M | L4/L5 IDP | Standing, 20 kg loads, 30 and 90 deg flexion, extension, rotation, lateral bending |

verified_from_primary: true (sex split of Polga and Schultz cohorts: the
extraction printed "W" for women; treat as F).

```json
{"source_id":"bruno2015","table_id":"B4_validation_tasks","verified_from_primary":true,"rows":[
{"study":"Andersson","n":3,"measure":["IDP_L3L4","EMG_T4"],"task":"lumbar flexion 10-50 deg, 10.2 kg per hand"},
{"study":"Polga","n":6,"measure":["IDP_T6T7_to_T10T11"],"task":"standing, flexion, extension, rotation, lateral bending, weighted arms"},
{"study":"Rohlmann","n":2,"measure":["L1_implant_load"],"task":"standing, flexion, extension, rotation, lateral bending, arm elevation/abduction"},
{"study":"Sato","n":8,"measure":["IDP_L4L5"],"task":"standing, 30 deg flexion, 15 deg extension"},
{"study":"Schultz","n":4,"measure":["IDP_L3L4","EMG_L1_L5"],"task":"standing, arm positions, 8 kg, 30 deg flexion"},
{"study":"Takahashi","n":3,"measure":["IDP_L4L5","EMG_L3"],"task":"0-30 deg flexion, 0 or 5 kg per hand"},
{"study":"Wilke","n":1,"measure":["IDP_L4L5"],"task":"standing, 20 kg, 30 and 90 deg flexion, extension, rotation, lateral bending"}]}
```

### Table B5: Reported Model Outputs

| Output | Metric | Value |
|---|---|---|
| IDP lumbar (Fig. 4) | r | 0.91 |
| IDP lumbar | slope 95% CI | 0.767 to 1.096 |
| IDP lumbar | intercept 95% CI | -0.205 to 0.147 |
| IDP thoracic | r | 0.87 |
| IDP thoracic | slope 95% CI | 0.989 to 1.966 |
| IDP thoracic | intercept 95% CI | -1.178 to 0.178 |
| L1 vertebral load (Fig. 5) | r | 0.88 (0.97 excluding extension) |
| L1 extension | model vs measured, percent of standing | 114 vs 42 |
| L1 extension sensitivity | 5 deg / 30 deg | 94% / 137% |
| Erector spinae EMG (Fig. 6) | r | 0.98 (T4), 0.98 (L3), 0.91 (L1-L5) |

IDP model definition (Eq. 2): IDP = F_C / CSA_vert x 0.66, with 0.66 a
lumbar-derived correction applied to thoracic levels as well. Units of IDP in
Fig. 4 were not stated in the retrieved text (MPa is the evident unit but is
not read here). verified_from_primary: true

```json
{"source_id":"bruno2015","table_id":"B5_model_outputs","verified_from_primary":true,"rows":[
{"output":"idp_lumbar","r":0.91,"slope_ci95":[0.767,1.096],"intercept_ci95":[-0.205,0.147]},
{"output":"idp_thoracic","r":0.87,"slope_ci95":[0.989,1.966],"intercept_ci95":[-1.178,0.178]},
{"output":"l1_vertebral_load","r":0.88,"r_excluding_extension":0.97,"slope_ci95":[0.302,1.044],"intercept_ci95":[-4.312,101.634]},
{"output":"l1_extension_pct_of_standing","model":114,"measured":42,"model_at_5deg":94,"model_at_30deg":137},
{"output":"emg_erector_spinae","r_T4":0.98,"r_L3":0.98,"r_L1_L5":0.91},
{"output":"idp_conversion_factor","value":0.66}]}
```

The vertebral body load comparison also appears in Bruno Table 2 (muscle CSA
and moment-arm RMSE); that table is irrelevant to the kinematic distribution
and is not transcribed.

## Pearcy and Tibrewal 1984

Source key: `pearcy1984`.

Primary text not accessible (Lippincott paywall; PubMed fetch returned only a
cookie page). The 1985 monograph (Pearcy, Acta Orthop Scand Suppl 212) is open
at actaorthop.org but is a scanned bitmap with no text layer; it is a
different publication and would need OCR, so it is not used here.

### Table P1: Per-Level Axial Rotation and Coupled Lateral Bend, L1-S1

Not transcribed. verified_from_primary: false

```json
{"source_id":"pearcy1984","table_id":"P1_axial_rotation_coupled_bend","verified_from_primary":false,"rows":[]}
```

Fujii 2007 reproduces Pearcy's in vivo ranges in its Table 1 (flexion-
extension, axial rotation, lateral bending per level), so the figures can be
read there as a citing source once an owner confirms them; they were not
transcribed here.

## Fujii 2007

Source key: `fujii2007`.

Results text, n = 10 healthy volunteers (6 men, 4 women per the extraction),
3D MRI, trunk rotated 45 deg, mean +/- SD, degrees. Axial rotation is per
side. Coupled flexion positive = flexion. Coupled lateral bending negative =
opposite to the trunk rotation, positive = same direction. Segment-level
rotation in the paper is relative to neutral for each vertebra pair.

### Table F1: Per-Segment Motion at 45 Degrees of Trunk Rotation

| Segment | Axial rotation (deg) | Coupled flexion (deg) | Coupled lateral bending (deg) |
|---|---|---|---|
| T12/L1 | 1.2 +/- 0.4 | -0.1 +/- 0.4 (within error) | +0.9 +/- 0.3 |
| L1/2 | 1.3 +/- 0.5 | 0.8 +/- 0.6 | -1.6 +/- 0.7 |
| L2/3 | 1.4 +/- 0.3 | 1.0 +/- 0.8 | -3.0 +/- 1.0 |
| L3/4 | 1.7 +/- 0.6 | 1.3 +/- 0.5 | -3.4 +/- 0.9 |
| L4/5 | 1.7 +/- 0.6 | 0.9 +/- 0.6 | -1.6 +/- 0.9 |
| L5/S1 | 1.6 +/- 0.7 | 1.9 +/- 1.6 | +0.6 +/- 0.6 |

Meaning: coupled lateral bend reverses sign across the lumbar spine (opposite
to rotation L1-L5, same direction at the thoracolumbar and lumbosacral ends),
and coupled flexion is positive from L1/2 to L5/S1. The paper's Table 1 lists
per-level ranges from several sources whose values differ from these text
values; its column definitions were not stated in the extraction, so Table 1
is not transcribed. verified_from_primary: true (PMC full text, two
independent extractions agree; PDF re-check recommended).

```json
{"source_id":"fujii2007","table_id":"F1_45deg_trunk_rotation","verified_from_primary":true,"unit":"deg","n":10,"rows":[
{"segment":"T12/L1","axial_rotation":[1.2,0.4],"coupled_flexion":[-0.1,0.4],"coupled_lateral_bend":[0.9,0.3]},
{"segment":"L1/2","axial_rotation":[1.3,0.5],"coupled_flexion":[0.8,0.6],"coupled_lateral_bend":[-1.6,0.7]},
{"segment":"L2/3","axial_rotation":[1.4,0.3],"coupled_flexion":[1.0,0.8],"coupled_lateral_bend":[-3.0,1.0]},
{"segment":"L3/4","axial_rotation":[1.7,0.6],"coupled_flexion":[1.3,0.5],"coupled_lateral_bend":[-3.4,0.9]},
{"segment":"L4/5","axial_rotation":[1.7,0.6],"coupled_flexion":[0.9,0.6],"coupled_lateral_bend":[-1.6,0.9]},
{"segment":"L5/S1","axial_rotation":[1.6,0.7],"coupled_flexion":[1.9,1.6],"coupled_lateral_bend":[0.6,0.6]}]}
```

(Each pair is [mean, SD].)

## Wilke 1999

Source key: `wilke1999`.

Single volunteer (male, 45 years, 70 kg, 1.68 m), transducer in the
non-degenerated L4-L5 nucleus, range to 3.5 MPa. Source: Table 1, page 757,
read from the primary PDF text.

### Table W1: L4-L5 Intradiscal Pressure by Posture and Task

| Position | Pressure (MPa) |
|---|---|
| Lying supine | 0.10 |
| Lying on the side | 0.12 |
| Lying prone | 0.11 |
| Lying prone, extended back, on elbows | 0.25 |
| Laughing heartily, lying laterally | 0.15 |
| Sneezing, lying laterally | 0.38 |
| Peaks by turning around | 0.70-0.80 |
| Relaxed standing | 0.50 |
| Standing, Valsalva maneuver | 0.92 |
| Standing, bent forward | 1.10 |
| Sitting relaxed, no backrest | 0.46 |
| Sitting actively straightening the back | 0.55 |
| Sitting with maximum flexion | 0.83 |
| Sitting bent forward, elbows supported | 0.43 |
| Sitting slouched into the chair | 0.27 |
| Standing up from a chair | 1.10 |
| Walking barefoot | 0.53-0.65 |
| Walking with tennis shoes | 0.53-0.65 |
| Jogging, hard street shoes | 0.35-0.95 |
| Jogging, tennis shoes | 0.35-0.85 |
| Climbing stairs, one at a time | 0.50-0.70 |
| Climbing stairs, two at a time | 0.30-1.20 |
| Walking down stairs, one at a time | 0.38-0.60 |
| Walking down stairs, two at a time | 0.30-0.90 |
| Lifting 20 kg, bent over, round back | 2.30 |
| Lifting 20 kg as taught in back school | 1.70 |
| Holding 20 kg close to the body | 1.10 |
| Holding 20 kg, 60 cm from the chest | 1.80 |
| Pressure increase during night (7 h) | 0.10-0.24 |

The abstract text also lists "sitting unsupported 0.46" and "nonchalant
sitting 0.3" (Table 1 gives 0.27 for slouched; the abstract rounds). The
30 and 90 degree flexion and rotation/lateral-bending tasks cited by Bruno
for this study are not rows of this Table 1 and were not located in the pages
read. verified_from_primary: true

```json
{"source_id":"wilke1999","table_id":"W1_intradiscal_pressure_L4L5","verified_from_primary":true,"unit":"MPa","rows":[
{"position":"lying_supine","value":0.10},
{"position":"lying_side","value":0.12},
{"position":"lying_prone","value":0.11},
{"position":"lying_prone_on_elbows","value":0.25},
{"position":"laughing_lying_lateral","value":0.15},
{"position":"sneezing_lying_lateral","value":0.38},
{"position":"turning_peaks","range":[0.70,0.80]},
{"position":"relaxed_standing","value":0.50},
{"position":"standing_valsalva","value":0.92},
{"position":"standing_bent_forward","value":1.10},
{"position":"sitting_relaxed_no_backrest","value":0.46},
{"position":"sitting_active_straight","value":0.55},
{"position":"sitting_max_flexion","value":0.83},
{"position":"sitting_forward_elbows_supported","value":0.43},
{"position":"sitting_slouched","value":0.27},
{"position":"standing_up_from_chair","value":1.10},
{"position":"walking","range":[0.53,0.65]},
{"position":"jogging_hard_shoes","range":[0.35,0.95]},
{"position":"jogging_tennis_shoes","range":[0.35,0.85]},
{"position":"stairs_up_one","range":[0.50,0.70]},
{"position":"stairs_up_two","range":[0.30,1.20]},
{"position":"stairs_down_one","range":[0.38,0.60]},
{"position":"stairs_down_two","range":[0.30,0.90]},
{"position":"lift_20kg_round_back","value":2.30},
{"position":"lift_20kg_back_school","value":1.70},
{"position":"hold_20kg_close","value":1.10},
{"position":"hold_20kg_60cm","value":1.80},
{"position":"night_increase_7h","range":[0.10,0.24]}]}
```

## Golf Lumbar Loads: Context Bands

### Table G1: Hosea 1990 L3-L4 (Secondary Citation Only)

Hosea, Gatt and Galli used a five iron, 4 professionals (mean age 37) and
4 amateurs (mean age 34). The numbers below come from a citing review
(PMC4335481) and web search snippets, not from the 1990 chapter.

| Quantity | Professional | Amateur |
|---|---|---|
| Compressive load L3-L4 (N) | 7584 | 6100 |
| Peak shear (N) | 329 | 596 |

The review equates these compressions with about 8 times body weight.
verified_from_primary: false

```json
{"source_id":"hosea1990","table_id":"G1_L3L4_loads","verified_from_primary":false,"rows":[
{"group":"professional","compression_N":7584,"peak_shear_N":329},
{"group":"amateur","compression_N":6100,"peak_shear_N":596}]}
```

### Table G2: Lim, Chow and Chae 2012 L4-L5 (Abstract)

Five male collegiate golfers, 4-camera plus 2 force plates, EMG-assisted
optimization. Values are from the abstract of the paper itself.

| Quantity | Value |
|---|---|
| Peak compression (downswing) | greater than 6 BW |
| Peak anterior shear (follow-through) | approaching 1.6 BW |
| Peak medial shear (follow-through) | approaching 0.6 BW |

Body-weight multiples are as stated; no newton values are used (a 4,300 N /
882 N / -252 N set seen in a search snippet is from an earlier, different
paper and is not used). Tables and figures of the full paper were not read.
verified_from_primary: true (abstract only)

```json
{"source_id":"lim2012","table_id":"G2_L4L5_loads_abstract","verified_from_primary":true,"unit":"BW","rows":[
{"quantity":"peak_compression","value_gt":6.0},
{"quantity":"peak_anterior_shear","value_approx":1.6},
{"quantity":"peak_medial_shear","value_approx":0.6}]}
```

## Crunch Factor and ISB Spine Recommendation

- Crunch factor (Sugaya et al.): instantaneous product of lateral trunk
  flexion angle and axial trunk rotational velocity. The definition is from
  the abstract and summary of Cole and Grimshaw (2014), a citing paper, so
  verified_from_primary is false. The unit would be deg^2/s; the sign and
  frame conventions of the original were not read.
- Wu 2002 spine JCS: not read. The repository already implements the ISB
  spine profile in `joint_conventions.py`; the original text should be
  compared against it by the owner.

```json
{"source_id":"sugaya1999","table_id":"C1_crunch_factor_definition","verified_from_primary":false,"rows":[
{"term":"crunch_factor","definition":"lateral_bend_angle * axial_rotation_velocity","as_cited_by":"Cole and Grimshaw 2014"}]}
```

## Owner Action: Primary PDFs Needed

1. Bruno 2015 Supplemental Tables 1-4 (zip
   `Supplementary_Material_BIO-14-1531.zip`; the PMC and ASME downloads
   returned a bot check, and Europe PMC's API reports the article as not open
   access). Needed: per-joint T1-L5 coupling ratios for flexion, lateral
   bending and axial rotation (Supplemental Table 2) for SPN-3. Also the
   main-text PDF to re-confirm Tables B1-B5.
2. Pearcy and Tibrewal 1984, Spine 9:582-587, full text with per-level axial
   rotation and coupled bend tables.
3. Fujii 2007 PDF to confirm Table F1 and to read the paper's Table 1; and to
   confirm the full author list and DOI.
4. Wilke 1999: no action for Table W1; the 30/90 degree flexion, rotation and
   lateral-bending values quoted by Bruno are not in Table 1, so confirm in
   the paper's figures or text if SPN-11a needs them (a DOI is also needed).
5. Hosea, Gatt and Galli 1990 chapter (Science and Golf, E and FN Spon),
   L3-L4 compressive, shear, lateral-bend and torque loads.
6. Lim, Chow and Chae 2012 full text for tabulated peak loads and timing.
7. Sugaya et al. 1999 (Science and Golf III) or the original crunch factor
   paper for the exact definition; Wu et al. 2002 for the ISB spine axes.
