# PSA Email Summary - Lessons Learned and Large-Scale Plan

## Scope

- Source: PSA-related email threads in `Email Correspondence/` (92 .msg files) plus non-email docs in the folder
- Focus: lessons learned and performance implications for a larger-scale PSA unit
- Additional docs reviewed: `PSA Testing Results.xlsx`, `PSA Pressure Trace.xlsx`, `PSA Pattern.xlsx`, `3041 - Xebec PSA Proposed Guarantee 20181205 - Detailed.xlsx`, `PSA Control Narrative.docx`, `ICR PSA Logic Summary.docx`, `PSA Status Update July 2018.docx`, `Design Basis for PSA Unit.JPG`
- Note: PDFs and vector drawings could not be text-extracted with available tools in this environment; review pending for `Xebec PSA Manual.pdf`, `30784 IOM manual.pdf`, proposals, and P&IDs

## Lessons Learned (Performance-Focused)

### 1) Purity shortfalls observed during testing

- Product gas from the first PSA test contained N2 at ~0.81%, O2 at ~786 ppm, and H2O >22 ppm after ~2 hours of operation following an N2 purge.
- Operators reported N2 and CO2 trending high, with CO and H2 sometimes low relative to expected performance.
- Impurity control appears sensitive to flow, valve speed, and upstream composition variability.
- Lab results in `PSA Testing Results.xlsx` show multiple H2 sample sets with total contaminants ranging from ~8,981 ppm (purity ~99.10%) to ~12,057 ppm (purity ~98.79%) in April 2018 samples, improving to ~106.6 ppm (purity ~99.989%) in a later May 2018 sample and ~274.6 ppm (purity ~99.973%) after air-correction.

### 2) Recovery below expected levels in field testing

- Performance spec expectations: ~99.999% purity and ~70% recovery (water saturated at 100% RH, no liquid water).
- Reported performance on one test case:
  - PSA feed: 9.66 SCFM; product: 1.33 SCFM; recovery ~44%.
  - Feed composition: H2 31.2%, CO 31.4%, CO2 17.7%, CH4 0.25%, N2 19.4%.
  - Inlet pressure 160 psig; outlet pressure 155 psig; valve speed 29.9 RPH (0.5 RPM).
- A design basis table (low-pressure case) targets 99.999% H2 purity at 70% recovery with feed 14 scfm at 160 psig, product 3 scfm at 145 psig, and exhaust 11 scfm at 2 psig; nominal feed composition listed as H2 29.6%, CO 29.6%, CO2 21.2%, N2 19.0%, H2O 0.6% (by volume).

### 3) Operating limits tie purity to valve speed and feed component flows

- Minimum rotary valve speeds rise with feed flow. Lower speeds increase product flow but reduce purity and increase bed loading.
- Component-specific max flows at each valve speed can become limiting even if total flow appears acceptable.
- Example for low flows (1-5 SCFM): minimum speed ~0.17 RPM (10 RPH). Above 5 SCFM, minimum speed increases and component limits tighten.
- Control narrative calculations explicitly set minimum rotational speed using rolling 1-minute average feed SCFM and feed composition (CO and CO2 constraints) and use RPH derived from gear tooth timing (32-tooth gear).

### 4) Turndown constraints are a major sizing risk

- Xebec PSA turndown noted at ~50-60% of design flow.
- Large units cannot be run efficiently at very low flow (e.g., 7 SCFM on a 50-100 SCFM unit).
- A smaller unit is feasible for 7 SCFM but caps flow around ~22 SCFM and limits scalability.

### 5) Feed composition variability requires modeling across a matrix, not a single normalized case

- Syngas composition can shift rapidly with feed changes (e.g., CO from ~36% to <22%, CO2 from ~20% to ~38% in an hour).
- Operators requested model runs across max-concentration bounds (e.g., 15 SCFM with CO 45%, CO2 40%, N2 25%, CH4 5%, O2 2%).
- Normalizing a single composition risks under-sizing and purity loss during off-nominal operation.

### 6) Adsorbent layering and water/CO2 management are critical

- Observed/expected layering (from internal analysis):
  - Layer 1: activated alumina or aluminosilicate desiccant (water protection)
  - Layer 2: activated carbon (CO2, residual water, CH4)
  - Layer 3: 5A zeolite (CO/CO2/CH4 removal)
  - Layer 4: CaX zeolite (N2/O2 polishing; sensitive to water/CO2)
- CO2 and H2O can “clog” zeolites, reinforcing the need for front-end removal and proper packing.
- BTX noted above 10 ppm historically; ammonia also present and only partially removed by scrubbers.
- PSA Status Update (July 2018) explicitly notes that adding activated carbon above the desiccant was a warranty modification after initial tests failed to meet purity/recovery, indicating a strong sensitivity to CO2/H2O front-end removal.

### 7) Mechanical / packing issues affect performance and reliability

- On-site bedfill risks: slight vessel tilt (1-2 degrees) can force media into the O-ring groove, preventing seal and causing rework/scrap.
- Repacking and pressure trace diagnostics indicated possible restriction in product pressurization steps (slower pressurization, flattened curves).
- Pressure hold tests showed ~2 psig loss over 5 minutes, suggesting no large internal leak but possible localized restrictions.
- `PSA Pressure Trace.xlsx` captures a 12.3-hour trace on 2018-08-07 with pressure oscillating between ~0 and ~164 psig (mean ~19.25 psig, 95th percentile ~64.27 psig), consistent with rapid cycling and equalization events.

### 8) Cycle behavior and control expectations

- PSA uses 2 rotary valves and 9 beds; each revolution includes 18 half-steps.
- Typically 2-3 columns produce, 2-3 regenerate, the rest equalize at any moment.
- Product purity control often uses a 4-20 mA H2 purity signal to adjust rotary valve speed.
- `PSA Pattern.xlsx` documents 18-step sequences for 1-, 2-, and 3-step equalization patterns across 9 columns, reinforcing the 18-step rotary cycle model referenced in email correspondence.

### 9) Pressure and compressor constraints

- Proposed/expected PSA inlet pressures ranged from ~150-225 psig; 200 psig was a common baseline for guarantees.
- Small booster compressor behavior:
  - Boosts from ~100 to 160 psig (system cycles 145-180 psig).
  - Approximate flow at 120 psig ~4 SCFM; nominal design flow ~3 SCFM.
- PSA logic ranges and alarms include 0-300 psig transmitters with HH/H/L alarms on Stage 1 bottom pressure (HH 270 / H 255 / L 225 psig), showing high-pressure protection thresholds used in controls.

### 10) CO2-removal configurations appear feasible at small scale

- Small PSA repurpose target: 8 SCFM at 150 psig to reduce CO2 from ~20% to <4%.
- CO2 is easiest to remove; activated carbon is the preferred adsorbent for CO2 removal.
- A 3-stage CO2 PSA design was discussed, positioned before H2 PSA to reduce load and improve overall performance.

## Performance Targets and Proposed Guarantees (from email guidance)

- Purity spec: 99.999% (per performance specs email)
- Expected recovery: ~70% (performance specs email)
- Proposed guarantee (200 psig, with linear interpolation between cases):
  - Case 1 (600 SCFM): H2 > 30%, CH4 < 2%, CO < 30%, CO2 < 30%, N2 < 8%, H2O saturated at 100 F
  - Case 2 (1100 SCFM): H2 > 30%, CH4 < 2%, CO < 40%, CO2 < 23%, N2 < 5%, H2O saturated at 100 F
- Guarantees assume tail gas recycle from PSA2 to compressor inlet.
- Detailed guarantee table (`3041 - Xebec PSA Proposed Guarantee 20181205 - Detailed.xlsx`) shows interpolation from Case 1 (600 scfm) to Case 2 (1100 scfm) at 200 psig with recovery rising 70% to 75% and tightening impurity limits: CO 30% to 40%, CO2 ~30% to ~23%, CH4 2% to 1%, N2 8% to 5%.

## Plan for the Larger-Scale Unit (Performance-Centric)

### A) Confirm the feed envelope and performance targets

- Lock in flow range with realistic turndown (50-60% of design flow), avoiding oversizing.
- Use a composition matrix across max and min constituent bounds (not a normalized nominal).
- Define acceptance criteria explicitly: product purity (N2/O2 ppm), H2 recovery, pressure drop, and steady-state stability after purge.
- Align test conditions with the design basis flow/pressure table (160 psig feed, 145 psig product, 2 psig exhaust at 14/3/11 scfm) and document deviations during testing.

### B) Compare two-stage architectures using consistent performance criteria

**Option 1: Two PSA units in series (Xebec approach)**

- Stage 1: ~85% purity at high recovery.
- Stage 2: polishing to high purity at lower recovery.
- Tail gas from stage 2 recycled to compressor inlet.
- Expected overall recovery ~90% (from internal model based on small-scale data).

**Option 2: Two PSA units in series (InEnTec approach)**

- Stage 1: remove CO2/CH4/H2O with activated carbon; partial CO/N2 removal (~20% assumed).
- Stage 2: remove CO and N2 using zeolites; no tail gas recycle.
- Model assumption: ~75% removal per stage yields equivalent overall recovery ~86% after a chemical recycle step.
- Stage 1 inlet example (from internal model): 1013 SCFM with CO 37.8%, CO2 19.5%, H2 32.8%, H2O 5.6%, N2 3.7%.

### C) Evaluate tail-gas recycle to process (TRC) for recovery gains

- 50% tail-gas recycle: total PSA flow ~935 SCFM; recovery improvement from ~55% to ~70% (modeled).
- 75% tail-gas recycle: total PSA flow ~1260 SCFM; modeled recovery improvement toward ~97% (near compressor limits).
- Consider energy penalty (~550,000 BTU/hr) and potential 10-12% consumption of recycled H2/CO.

### D) Design for impurity management and adsorbent protection

- Front-load water and CO2 removal to protect downstream zeolites (especially CaX).
- Validate adsorbent layering and packing procedures to avoid CO2/water poisoning and performance drift.
- Consider dedicated CO2 PSA ahead of H2 PSA if CO2/H2O are dominant performance inhibitors.

### E) Instrumentation, controls, and mechanical safeguards

- Provide product purity measurement (H2 analyzer) with 4-20 mA output for valve speed control.
- Include thermal mass or ultrasonic flow meters for product flow.
- Add a surge tank on tail-gas line to dampen high-frequency pressure fluctuations from short cycle times.
- Avoid on-site bedfill when possible; enforce strict vessel alignment if on-site fill is unavoidable.
- Implement the control narrative formulas for minimum RPH based on rolling feed flow and CO/CO2 composition to prevent under-speed operation.
- Verify pressure transmitter scaling and alarm setpoints (Stage 1 bottom pressure HH/H/L = 270/255/225 psig) in commissioning to avoid nuisance trips or overpressure risk.

### F) Performance validation test plan (pre-commission and commissioning)

- Run tests across the full feed composition matrix at minimum, nominal, and maximum flow rates.
- Verify minimum valve speeds and component max flows per operating limit table.
- Track recovery, purity, O2/N2 ppm, H2O ppm, and pressure drop vs. valve speed.
- Include steady-state holds after purges to confirm stability and repeatability.

## Key Open Questions for the Large-Scale Design

- What purity/recovery tradeoffs are acceptable at the high-flow limit if composition shifts beyond the two guarantee cases?
- Can the compressor and recycle configuration consistently support 200-225 psig at the required flow without excessive cycling?
- Is 20% CO/N2 removal in activated carbon a realistic assumption, or should it be re-estimated with vendor data?
- What is the acceptable operating window for CH4 and BTX, given observed BTX >10 ppm and limited scrubber removal?
- Can the PSA be operated at the design-basis low-pressure case (14 scfm at 160 psig) without violating minimum RPH constraints derived from the control narrative?

## Data Extract for Agents (Non-email docs)

### PSA Testing Results (H2 Samples)

Source: `PSA Testing Results.xlsx` (sheet: H2 Samples).
| Metric | 2018-04-19 | 2018-04-25 | Blank | 2018-05-25 Sample 1 | 2018-05-25 Sample 2 | 2018-05-25 Sample 1 No Air | 2018-05-25 Sample 2 No Air | Units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oxygen | 786.1 | 2092.0 | 0.23 | 78.4 | 9100 | 0 | 0 | ppm |
| H2O | 22.0 | 22.0 | 18.5 | 22 | 22 | 22 | 22 | ppm |
| THC | 21.11 | 58.8 | 0.01 | 1.51 | 50.7 | 1.51 | 50.7 | ppm |
| Nitrogen | 8100.000000000001 | 9300.000000000002 | 0.24199999999999997 | 2.647 | 21000 | 0 | 0 | ppm |
| Carbon Dioxide | 5.02 | 442.0 | 10.4 | 1 | 35 | 1 | 35 | ppm |
| Carbon Monoxide | 47.2 | 142.6 | 1.69 | 1 | 166.9 | 1 | 166.9 | ppm |
| Total Contaminants | 8981.430000000002 | 12057.400000000003 | 31.072000000000006 | 106.55700000000002 | 30374.600000000002 | 25.51 | 274.6 | ppm |
| Purity | 99.101857 | 98.79426 | 99.9968928 | 99.9893443 | 96.96253999999999 | 99.99744899999999 | 99.97254 | % purity |
| Time at Stable Operation | 1.5 | 3.5 | | | | | | hours |
| PSA Feed Flow | 9.66 | 9.4 | | 10.2 | 11 | 10.2 | 11 | SCFM |
| PSA H2 Flow | 1.33 | 2.0 | | 2 | 2.8 | 2 | 2.8 | SCFM |
| PSA Valve Speed | 30.0 | 24.9 | | 32 | 24 | 32 | 24 | RPH |

### Design Basis (Flow Case 2: Low Pressure)

Source: `Design Basis for PSA Unit.JPG`.

- Targets: H2 purity 99.999%, expected recovery 70%.
- Stream properties:

| Stream          | Fresh feed | Product | Exhaust |
| --------------- | ---------- | ------- | ------- |
| Flow (NCMH)     | 22         | 5       | 18      |
| Flow (scfm)     | 14         | 3       | 11      |
| Pressure (psig) | 160        | 145     | 2       |
| Temperature (C) | 40         | 40      | 40      |

- Feed composition (vol%): H2 29.6, CO 29.6, CO2 21.2, N2 19.0, H2O 0.6 (total 100).
- Product composition: H2 99.999, balance unspecified components.
- Exhaust composition (vol%): H2 11.2, CO 37.4, CO2 26.7, N2 23.9, H2O 0.8 (total 100).

### Proposed Guarantee (Interpolated Cases)

Source: `3041 - Xebec PSA Proposed Guarantee 20181205 - Detailed.xlsx` (pressure 200 psig).
| Flow scfm | H2 >= % | CO <= % | CO2 <= % | CH4 <= % | N2 <= % | Recovery % |
| --- | --- | --- | --- | --- | --- | --- |
| 600 | 30 | 30 | 30 | 2 | 8 | 70 |
| 700 | 30 | 32 | 28.721062618595823 | 1.8 | 7.4 | 71 |
| 800 | 30 | 34 | 27.371841155234655 | 1.6 | 6.8 | 72 |
| 900 | 30 | 36 | 25.962134251290884 | 1.4 | 6.2 | 73 |
| 1000 | 30 | 38 | 24.5 | 1.2 | 5.6 | 74 |
| 1100 | 30 | 40 | 22.99212598425197 | 1 | 5 | 75 |

### PSA Control Narrative (Logic/Calculations)

Source: `PSA Control Narrative.docx`.

- RPH calculation from gear tooth timing: RPH = (Time2 - Time1) \* TeethNo; gear teeth = 32.
- 1-minute rolling average feed flow: AveFeedFlow = (T1Flow + ... + T60Flow) / 60 (lb/hr).
- Feed flow conversion to SCFM at 0 C, 1 atm: AveFeedSCFM = AveFeedFlow _ SyngasMW _ R _ Ps _ Ts (R=1.31443 ft^3\*atm/lbmol/K, Ps=1 atm, Ts=273.15 K).
- Minimum rotational speed constraints (by flow and by component):
  - MinRPH,Flow = AveFeedSCFM \* NominalRPH / NominalFlow.
  - MinRPH,CO = AveFeedSCFM _ CO%/100 _ NominalRPH / NominalCO.
  - MinRPH,CO2 = AveFeedSCFM _ CO2%/100 _ NominalRPH / NominalCO2.

### PSA Logic Summary (Alarms/Setpoints)

Source: `ICR PSA Logic Summary.docx`.

- Pressure transmitters (0-300 psig): Stage 1 bottom, Stage 2 inlet, Stage 2 bottom.
- Stage 1 bottom pressure alarm setpoints: HH 270 psig, H 255 psig, L 225 psig.
- Stage 2 inlet/bottom setpoints listed as HH/H/L/LL = 0 (placeholders in doc).

### Pressure Trace Summary

Source: `PSA Pressure Trace.xlsx` (2018-08-07).

- Time span: 2018-08-07 00:00:00.123000 to 2018-08-07 12:18:21.401000; rows 173897.
- Pressure stats (psig): min 0.00, max 164.30, mean 19.25, P05 0.00, P95 64.27.

### PSA Cycle Pattern

Source: `PSA Pattern.xlsx`.

- 18-step cycle patterns provided for 1-, 2-, and 3-step equalization across 9 columns (S/E percentages per step).

### PSA Status Update (Operational Theory + Field Issues)

Source: `PSA Status Update July 2018.docx`.

- Field results: six processing days; samples did not meet vendor spec; warranty work in progress to add second stage.
- Adsorbent update: Xebec added activated carbon above desiccant after initial failures; later tests still failed to meet spec, prompting second-stage concept.
- Theory highlights:
  - Multi-layer adsorption; strongly adsorbed species (CO2/H2O) can dominate capacity and are hard to desorb.
  - Pressure equalization exposes upper zeolite layers to desorbed CO2/H2O, reducing N2 capacity and causing product contamination.
  - Low H2 feed (30-40% vs typical 60-80%) increases tail-gas flow and contaminant partial pressures; worsens desorption and promotes re-adsorption in upper layers.
  - Purity vs recovery tradeoff: higher valve speed improves purity but reduces recovery due to higher tail-gas fraction.
- Recommendations: two-stage approach to decouple H2O/CO2 removal; recycle second-stage tail gas to inlet to improve overall recovery (with N2 buildup caution).

### PSA Manuals and Proposals (PDF Extracts)

Source: `Xebec PSA Manual.pdf`, `30784 IOM manual.pdf`, `H16-1061 PSA upgrading proposal G4 rev 06.pdf`, `H16-1061 PSA upgrading proposal G4 rev 06 with Perez notes from 2019 01 17 9am call with Xebec - Internal Use Only.pdf`, `H17-1016 PSA upgrading proposal G2 rev 01 (002).pdf`.

- Manual titles and models:
  - Xebec Rapid-Cycle PSA IOM (Rev00, 31/03/2020), 2-stage PSA: 30" dia (G4) + 12" dia (G2), models `PSA30-90H-G4-250ASME-120VAC` and `PSA12-70H-G2-250ASME-120VAC`, Job 33090.
  - Xebec Rapid-Cycle PSA IOM (Rev0, March 2018), model `PSA3H-G2-330A-480`, Job 30784.
- Proposal highlights (G4/G2 twin stage, Arlington OR):
  - 9-bed fast-cycle PSA with triple equalization per stage; rotary valves (<1 CPM typical); pre-loaded adsorbents; adjustable cycle speed.
  - Design basis flow cases:
    - HIGH: Raw feed 1100 scfm; product 263 scfm; exhaust 837 scfm; feed pressure >200 psig; product ~180 psig; exhaust ~2 psig. Guaranteed recovery ~70.6% with product purity >99.999% H2.
    - LOW: Raw feed 600 scfm; product 147 scfm; exhaust 453 scfm; feed pressure >200 psig; product ~180 psig; exhaust ~2 psig. Guaranteed recovery ~72.9% with product purity >99.999% H2.
  - Product impurity targets (HIGH case, product stream): CO2 <1 ppmv, CO <0.2 ppmv, CH4 <2 ppmv; O2/N2 ~0.
  - Process conditions: feed water at 100% RH, zero liquid water; zero particulates >0.5 micron; ±10 C product/exhaust temp from feed.
  - Operating margin: +10% feed flow or contaminants handled by +10% valve speed with ~0.5% recovery reduction.
- Proposal highlights (G2 small unit):
  - High-pressure case: feed 22 scfm at 290 psig; product 5 scfm at 275 psig; exhaust 17 scfm at 2 psig; expected recovery 73% at 99.999% purity.
  - Low-pressure case: feed 14 scfm at 160 psig; product 3 scfm at 145 psig; exhaust 11 scfm at 2 psig; expected recovery 70% at 99.999% purity.
  - Process conditions: 100% RH, zero liquid water; zero particulates >0.5 micron; stated nominal composition tolerance ±2.5%.

### PDFs and Drawings Pending Extraction

Remaining items not yet extracted:

- `3041-PID-ALL-2024-03-21.pdf`
- `3041 PSA drawing.PDF`
