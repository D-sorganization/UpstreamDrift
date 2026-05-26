# Strokes Gained Optimizer — Project Specification

**Working title:** `sg-optimizer` (final name TBD; suggested companion repo to AffineDrift under `D-sorganization`)
**Status:** Design doc v2 — adds course-condition tuning and classic-hole library
**Target agents:** Jules / Claude Code, dev-branch-first workflow
**Owner:** Dieter

---

## 0. Mission

Build a **personalized golf strategy optimizer** that recommends optimal aim points, club selections, and shot types on a per-hole basis. The system is built on four pillars:

1. A **player-specific shot dispersion model** — a tilted bivariate distribution capturing the long-left / short-right asymmetry produced by clubface mechanics, scaled by the player's individual skill profile relative to a tour baseline.
2. A **course conditions model** — independently tunable parameters for rough severity, tree penalization, and green speed (stimpmeter) that modulate how each lie type behaves without changing the underlying hole geometry.
3. A **Markov Decision Process** formulation of the hole, solved by vectorized value iteration, producing a player-specific value function and optimal policy under the chosen conditions.
4. A **map-based interface** for both **loading classic example holes** (TPC Sawgrass #17, Augusta #13, Road Hole, Pebble #7, Cypress #16) shipped with the package, **and** for tracing custom holes from Google Maps satellite imagery. Players edit their skill profile, dial in course conditions, and the system produces optimal-aim heatmaps and value surfaces overlaid on the map.

This is **explicitly not** another strokes-gained-vs-tour-baseline tool. Standard SG is descriptive (vs population average). This project is prescriptive (vs the _player's own_ optimal policy under their _own_ distribution under the _current course conditions_).

This project is a downstream companion to **AffineDrift**: AffineDrift models the swing as a control-affine biomechanical system; `sg-optimizer` models the strategic consequences of swing variability. A future integration point is parameterizing the shot model from AffineDrift simulation output.

---

## 1. Conceptual Foundation

### 1.1 The Shot Pattern as a Tilted Bivariate Distribution

A shot's outcome relative to the intended aim line is modeled as a 2D random variable $(\Delta_\parallel, \Delta_\perp)$, where $\Delta_\parallel$ is along-target distance error and $\Delta_\perp$ is lateral error (left positive by convention).

**Why a tilted ellipse, not axis-aligned.** Clubface angle at impact is the primary driver of both starting line direction _and_ dynamic loft. A closed face sends the ball left **and** delofts the club, producing a longer, lower flight; an open face sends the ball right **and** adds loft, producing a shorter, higher flight. The result is that "long misses tend to be left misses" and "short misses tend to be right misses" — a positive correlation between $\Delta_\parallel$ and $\Delta_\perp$, which manifests as an ellipse tilted clockwise from the target line (for a right-handed golfer).

This tilt is empirically small (~5–15°) but **matters significantly when an aim line passes near a hazard**, because the worst miss is no longer perpendicular to target.

**Mathematical form:**

$$\begin{pmatrix} \Delta_\parallel \\ \Delta_\perp \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} b_\parallel \\ b_\perp \end{pmatrix}, \Sigma\right), \quad \Sigma = \begin{pmatrix} \sigma_\parallel^2 & \rho\,\sigma_\parallel\sigma_\perp \\ \rho\,\sigma_\parallel\sigma_\perp & \sigma_\perp^2 \end{pmatrix}$$

with $\rho \in (0, 0.5)$ for right-handed golfers (negative for left-handed by mirror symmetry). The tilt angle of the principal axes is

$$\theta_{\text{tilt}} = \tfrac{1}{2}\arctan\!\left(\frac{2\rho\sigma_\parallel\sigma_\perp}{\sigma_\parallel^2 - \sigma_\perp^2}\right).$$

For better tail behavior (slice/hook tails), a 2-component Gaussian mixture or skew-normal is supported as a Phase-2 extension. The Phase-1 model is Gaussian.

### 1.2 Player Skill as Multiplicative Departures from Baseline

Tour-baseline shot patterns (from published Broadie/Fawcett data) form the **reference distribution**. The player profile defines departures:

$$\sigma_{\parallel, \text{player}}(c) = m_{\parallel}(c) \cdot \sigma_{\parallel, \text{baseline}}(c)$$
$$\sigma_{\perp, \text{player}}(c) = m_{\perp}(c) \cdot \sigma_{\perp, \text{baseline}}(c)$$
$$\mu_{\text{dist}, \text{player}}(c) = \mu_{\text{dist}, \text{baseline}}(c) + \Delta\mu(c)$$

per club $c$. Bias terms $b_\parallel, b_\perp$ are absolute (yards), not multiplicative — a chronic 5-yard push-right is 5 yards regardless of skill level.

### 1.3 Putting: A Separate Model

Putting is qualitatively different and must not be shoehorned into the full-swing dispersion model:

- **Outcome is discrete**: holed or not.
- **Make probability** is a function of distance (and break, slope, green speed) with empirically well-fit logistic shape.
- **Conditional leave distribution** given a miss must be modeled separately (drives 3-putt rates).

Two-component model:

$$P(\text{make} \mid d, s_{\text{stimp}}) = \sigma_{\text{logistic}}(\alpha + \beta \log d) \cdot k_{\text{player}}(d) \cdot k_{\text{stimp}}(d, s_{\text{stimp}})$$

where $k_{\text{player}}(d)$ is a player-specific multiplier that can vary with distance (letting users encode "poor short putter, decent lag putter" by setting $k(3\text{ft}) = 0.85$ while $k(25\text{ft}) = 1.1$) and $k_{\text{stimp}}$ is the green-speed modifier described in §1.4.

### 1.4 Course Conditions — Rough, Trees, Greens

Skill-relative shot patterns aren't enough on their own. The same player faces a different problem on a soft, slow-greens resort course than on a US-Open setup with heavy rough and 13.5-stimp greens. The strategy optimizer accepts **course condition parameters** that modulate the shot model and putting model conditional on lie class, independently of the hole's geometry.

**Why orthogonal to geometry.** The same `hole.geojson` (tee, fairway, rough, water, OB outlines) plays radically differently depending on conditions. A drivable par 4 in the rain with 4-inch rough is not a drivable par 4. By keeping conditions separate, you can ask "how would this hole play under tournament conditions?" without re-tracing anything, and you can ship a single classic-hole geometry with multiple condition presets.

Three primary tunable conditions:

#### 1.4.1 Rough Severity

Parameterized as $r \in [0, 1]$ (or via presets: `light` / `medium` / `heavy` / `us_open`). Modulates any shot played **from** a `rough` lie:

- **Distance multiplier**: $\mu_{\text{dist}}(r) = 1 - 0.08\,r - 0.12\,r^2$.
  Light rough costs ~5%, medium ~13%, heavy ~25%, US-Open-like ~30%.
- **Lateral dispersion multiplier**: $1 + 0.4\,r$. Heavy rough nearly doubles lateral variance.
- **Flyer probability**: $p_{\text{flyer}}(r)$ — small at low $r$, peaks at medium rough (the dangerous "between lies" regime), drops at heavy rough where the ball sits down.
- **Spin / rollout effect**: shots from rough have reduced backspin, rolling out further on landing — relevant when calculating whether an approach holds the green.

#### 1.4.2 Tree Penalization

Parameterized as $t \in [0, 1]$ representing the playability of `trees` lies:

- $t = 0$: trees are decorative; ball usually finds an opening; treat similarly to rough.
- $t = 0.5$: typical; can advance with reduced distance, wedge-only club selection.
- $t = 1$: dense undergrowth, thick pine straw, or heavy canopy; **mandatory pitch-out sideways** to the nearest fairway or playable rough.

The transition from a trees lie samples a recovery distribution: distance is compressed (typically <100 yards advancement), direction is heavily constrained toward the nearest fairway centroid, and lateral dispersion is wide. At $t = 1$ the action set is restricted to "punch out" with no aim choice beyond direction.

#### 1.4.3 Green Speed (Stimpmeter)

$s_{\text{stimp}} \in [8, 14]$. Modifies:

- **Putting make probability**: faster greens make breaks more punishing. The hole effectively "shrinks" because the ball reaches it carrying more speed. Use a multiplier on the baseline make curve:
  $$k_{\text{stimp}}(d, s_{\text{stimp}}) = 1 - \alpha\,(s_{\text{stimp}} - 10)\,g(d)$$
  where $g(d)$ grows with distance — fast greens hurt short putts a little, longer putts more.
- **Three-putt likelihood**: faster greens widen the leave-distance distribution from lag putts.
- **Approach shot rollout**: faster greens reduce effective landing-window depth. Modeled as a multiplier on the green's "effective depth" (the depth of green a player can land in and still hold), tightening the approach-shot landing target.
- **Pin position difficulty**: optionally couples with the `pin_position_difficulty` parameter — back-right pins on a fast green are harder than the same pin on a slow one.

All three condition parameters are stored in a `CourseConditions` config attached to the hole (or to a round of play). Changing conditions re-solves the MDP; the player profile is untouched.

`# AGENT-NOTE:` Conditions affect the **shot model**, not the **hole geometry**. The same `hole.geojson` can be played under benign or punitive conditions and produce different optimal strategies. Keep these strictly orthogonal in the code: hole geometry parses to a `LieRaster`, conditions parse to a `CourseConditions` object, the `ShotModel` consumes both. Do not bake condition parameters into the rasterized lie codes.

### 1.5 The MDP

- **State** $s = (x, y, \ell)$: ball position in hole-frame coordinates, with discrete lie class $\ell \in \{$tee, fairway, rough, trees, sand, recovery, green, water, OB, holed$\}$.
- **Action** $a = (c, \theta_{\text{aim}}, \text{shot type})$: club, aim line, and shot type (full / knockdown / draw / fade — Phase 2 extension).
- **Transition** $P_i(s' \mid s, a; \text{Conditions})$: sample landing from the player's tilted bivariate, **modulated by conditions and starting lie**, project to world frame, look up landing lie. Water/OB handled by deterministic drop transitions with added stroke. Trees handled by the recovery model when $t$ is high.
- **Cost** $+1$ per stroke, absorbing at holed.
- **Bellman**: $V^*(s) = \min_a [1 + \mathbb{E}\, V^*(s')]$.

Solved by vectorized value iteration on a discretized grid.

---

## 2. Repository Structure

```
sg-optimizer/
├── pyproject.toml                  # single source of truth (Black, Ruff, mypy, pytest config)
├── README.md
├── AGENTS.md                       # agent-legibility notes
├── .pre-commit-config.yaml
├── .github/workflows/              # self-hosted runner on BRICK
├── docs/
│   ├── design/                     # this spec lives here
│   ├── math/                       # derivations (LaTeX or markdown)
│   ├── data_sources.md             # cited references for all baseline numbers
│   └── user_guide/
├── src/sg_optimizer/
│   ├── __init__.py
│   ├── shot_model/
│   │   ├── README.md
│   │   ├── AGENTS.md
│   │   ├── distributions.py        # TiltedBivariateGaussian
│   │   ├── baseline.py             # tour/scratch reference data
│   │   ├── player_profile.py       # PlayerProfile dataclass + YAML I/O
│   │   ├── putting.py              # make-pct curves + leave model
│   │   └── shot_model.py           # top-level: (state, action, profile, conditions) -> samples
│   ├── course/
│   │   ├── README.md
│   │   ├── AGENTS.md
│   │   ├── geometry.py             # GeoJSON, projection (lat/lon -> UTM)
│   │   ├── rasterize.py            # polygon -> lie raster
│   │   ├── conditions.py           # rough / trees / greens models
│   │   ├── library.py              # classic-holes loader
│   │   ├── features.py             # distance to hazards, fairway width, etc.
│   │   └── course_io.py
│   ├── maps/
│   │   ├── README.md
│   │   ├── google_maps_client.py   # Static Maps API, tile fetch, caching
│   │   ├── osm_fallback.py         # OpenStreetMap golf feature query
│   │   └── tile_cache.py
│   ├── mdp/
│   │   ├── README.md
│   │   ├── AGENTS.md
│   │   ├── state.py
│   │   ├── action.py
│   │   ├── transition.py           # uses shot_model + course + conditions
│   │   ├── value_iteration.py      # vectorized solver
│   │   ├── policy.py
│   │   └── risk_objectives.py      # CVaR, quantile (Phase 2)
│   ├── ui/
│   │   ├── README.md
│   │   ├── app.py                  # PyQt6 main window
│   │   ├── map_widget.py           # QtWebEngine wrapper for map + tracing
│   │   ├── profile_editor.py       # skill profile editing dialog
│   │   ├── conditions_panel.py     # rough / trees / stimp tuning controls
│   │   ├── library_browser.py      # classic-holes selector
│   │   ├── strategy_view.py        # aim heatmap, value surface overlay
│   │   └── web/                    # JS/HTML loaded into QtWebEngine
│   │       ├── index.html
│   │       ├── trace_tool.js       # polygon tracing on map
│   │       └── strategy_overlay.js # aim heatmap rendering
│   ├── data/
│   │   ├── baselines/              # YAML: tour, scratch, bogey reference profiles
│   │   ├── clubs/                  # default club specs
│   │   └── courses/
│   │       ├── classics/           # shipped reference holes
│   │       │   ├── sawgrass_17/
│   │       │   ├── augusta_13/
│   │       │   ├── pebble_7/
│   │       │   ├── road_hole_17/
│   │       │   └── cypress_16/
│   │       └── user/               # user-created holes (gitignored)
│   └── cli.py                      # command-line entry for headless solves
├── tests/
│   ├── unit/
│   ├── integration/
│   └── property/                   # hypothesis-based property tests
└── notebooks/                      # exploratory analysis, not in production path
```

Conventions inherited from your existing projects:

- `pyproject.toml` is single source of truth for all tooling config.
- Black, Ruff, mypy enforced via pre-commit and CI on self-hosted runner.
- Per-directory `README.md` describes what the module does and how it's used.
- Per-directory `AGENTS.md` (where relevant) gives agent-specific guidance: what to touch, what to leave alone, key invariants.
- `# AGENT-NOTE:` inline comments mark places where an agent should be especially careful.
- TDD where feasible: failing test, then implementation. Design-by-contract assertions at module boundaries.
- DRY: shared data structures live in `shot_model.player_profile`, `course.conditions`, and `mdp.state`; not duplicated.

---

## 3. Module Specifications

### 3.1 `shot_model.distributions`

Implements the tilted bivariate Gaussian and its operations.

```python
@dataclass(frozen=True)
class TiltedBivariateGaussian:
    sigma_long: float       # yards
    sigma_lat: float        # yards
    rho: float              # correlation in [-1, 1]
    bias_long: float = 0.0  # yards, positive = past target
    bias_lat: float = 0.0   # yards, positive = left of target

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...
    def covariance_matrix(self) -> np.ndarray: ...
    def tilt_angle_degrees(self) -> float: ...
    def confidence_ellipse(self, level: float = 0.95) -> dict: ...
    def scaled(self, mult_long: float, mult_lat: float) -> "TiltedBivariateGaussian": ...
```

**Invariants** (enforce via `__post_init__`): `sigma_long > 0`, `sigma_lat > 0`, `-0.99 < rho < 0.99`.

**Tests**: empirical mean / covariance recovery from samples; tilt angle matches eigendecomposition; 95% ellipse contains ~95% of samples; `scaled()` produces correct multiplied σ values (used by conditions module).

### 3.2 `shot_model.baseline`

Reference distributions for tour / scratch / bogey golfers, sourced from published Broadie data. Stored as YAML in `data/baselines/`.

```yaml
# data/baselines/pga_tour.yaml
name: PGA Tour Average
source: "Broadie, Every Shot Counts (2014) — Tables A.1–A.4"
clubs:
  driver:
    carry_mean: 275
    total_mean: 295
    sigma_long: 14
    sigma_lat: 18
    rho: 0.25
    bias_long: 0
    bias_lat: 0
  7_iron:
    carry_mean: 172
    total_mean: 178
    sigma_long: 5
    sigma_lat: 7
    rho: 0.20
  # ... full bag
```

`# AGENT-NOTE:` Numerical values **must** be sourced from `docs/data_sources.md`, not invented. Cite in YAML comments. If a value is uncertain or interpolated, mark it explicitly: `# INTERPOLATED:` or `# ESTIMATE:`.

### 3.3 `shot_model.player_profile`

```python
@dataclass
class ClubSkill:
    skill_mult_long: float = 1.0
    skill_mult_lat: float = 1.0
    distance_offset: float = 0.0   # yards added/subtracted from baseline
    bias_long: float = 0.0
    bias_lat: float = 0.0
    enabled: bool = True

@dataclass
class PuttingSkill:
    make_pct_multipliers: dict[float, float]  # distance_ft -> multiplier
    three_putt_avoidance: float = 1.0

@dataclass
class PlayerProfile:
    name: str
    baseline: str                          # which baseline file
    clubs: dict[str, ClubSkill]
    putting: PuttingSkill
    short_game: dict[str, float]
    last_updated: datetime
    notes: str = ""

    def effective_distribution(self, club: str) -> TiltedBivariateGaussian: ...
    def to_yaml(self, path: Path) -> None: ...
    @classmethod
    def from_yaml(cls, path: Path) -> "PlayerProfile": ...
```

### 3.4 `shot_model.putting`

Logistic make-percentage model plus a leave-distance distribution conditional on miss. Takes both the player's putting skill and the green-speed condition as inputs.

```python
def make_probability(distance_ft: float,
                     profile: PuttingSkill,
                     greens: GreenModel,
                     baseline_curve: Callable[[float], float]) -> float: ...

def leave_distance_distribution(distance_ft: float,
                                profile: PuttingSkill,
                                greens: GreenModel) -> rv_continuous: ...
```

### 3.5 `course.geometry`

Handles all coordinate conversions. **Critical:** all internal computation happens in a local UTM projection (meters, converted to yards at I/O boundaries), never in lat/lon, to avoid the latitude-dependent yard-per-degree problem.

```python
def project_hole(hole_geojson: dict) -> ProjectedHole:
    """Reproject all polygons from WGS84 to a hole-local UTM zone.
       Anchor origin at tee for convenience."""

def world_to_hole_frame(point: tuple[float, float], hole: ProjectedHole) -> tuple[float, float]: ...
def hole_frame_to_world(point: tuple[float, float], hole: ProjectedHole) -> tuple[float, float]: ...
```

`# AGENT-NOTE:` Use `pyproj` for projection. Cache the UTM zone in the hole metadata to ensure round-tripping is exact.

### 3.6 `course.rasterize`

Polygon-to-raster with **lie-class priority resolution** for overlapping polygons:

Priority order (higher index wins for overlapping polygons):
`tee < fairway < rough < trees < sand < water < OB < green < holed`

The rationale: when polygons overlap (which happens often, e.g. a tree polygon drawn over rough), the higher-priority class is the actual playing condition. Green wins over everything except holed — if a polygon is marked green it's playable as a putting surface regardless of any other classification.

```python
LIE_CODES = {
    "tee": 0, "fairway": 1, "rough": 2, "trees": 3,
    "sand": 4, "water": 5, "ob": 6, "green": 7, "holed": 8,
}

def rasterize_hole(hole: ProjectedHole, resolution_yd: float = 1.0) -> LieRaster: ...
```

`# AGENT-NOTE:` Hazard boundaries are sharp and matter. At a 1-yard grid, anti-aliasing is wrong; use point-in-polygon, not pixel coverage. Test that a thin water hazard one yard wide still appears in the raster.

### 3.7 `course.conditions` (NEW)

Course condition models that modulate the shot model. **The key abstraction of v2.**

```python
@dataclass(frozen=True)
class RoughModel:
    severity: float  # [0, 1]

    def distance_multiplier(self) -> float:
        r = self.severity
        return 1 - 0.08*r - 0.12*r*r

    def dispersion_multiplier(self) -> float:
        return 1 + 0.4*self.severity

    def flyer_probability(self) -> float: ...
    def spin_reduction(self) -> float: ...

    @classmethod
    def preset(cls, name: str) -> "RoughModel":
        return {
            "light":   cls(severity=0.20),  # resort / casual
            "medium":  cls(severity=0.50),  # typical PGA Tour
            "heavy":   cls(severity=0.75),  # PGA major
            "us_open": cls(severity=0.95),  # punitive USGA setup
        }[name]


@dataclass(frozen=True)
class TreeModel:
    penalization: float  # [0, 1]

    def is_forced_punch_out(self) -> bool:
        return self.penalization > 0.85

    def recovery_distribution(self, state: State, hole: LieRaster) -> RecoveryAction:
        """Returns the distribution over recovery shot outcomes.
           At penalization=1.0, deterministically a sideways pitch-out
           to the nearest fairway. At lower values, allows forward advancement."""

    @classmethod
    def preset(cls, name: str) -> "TreeModel":
        return {
            "decorative": cls(penalization=0.10),
            "typical":    cls(penalization=0.50),
            "dense":      cls(penalization=0.80),
            "jail":       cls(penalization=1.00),
        }[name]


@dataclass(frozen=True)
class GreenModel:
    stimp: float  # [8, 14]

    def make_pct_modifier(self, distance_ft: float) -> float:
        # Faster greens degrade make % more at longer distances
        alpha = 0.015
        g = min(distance_ft / 10.0, 3.0)
        return 1.0 - alpha * (self.stimp - 10.0) * g

    def leave_distribution_modifier(self, distance_ft: float) -> float:
        # Faster greens widen leave distribution from lag putts
        return 1.0 + 0.08 * max(0.0, self.stimp - 10.0)

    def effective_green_depth_multiplier(self) -> float:
        # Faster greens reduce holdable landing area
        return 1.0 - 0.06 * max(0.0, self.stimp - 10.0)

    @classmethod
    def preset(cls, name: str) -> "GreenModel":
        return {
            "slow":       cls(stimp=9.0),
            "medium":     cls(stimp=10.5),
            "fast":       cls(stimp=12.0),
            "tournament": cls(stimp=13.0),
            "masters":    cls(stimp=13.5),
        }[name]


@dataclass(frozen=True)
class CourseConditions:
    rough: RoughModel
    trees: TreeModel
    greens: GreenModel
    pin_position_difficulty: float = 0.5  # [0, 1], affects make-% on long putts
    wind: WindModel | None = None         # Phase 3+

    @classmethod
    def benign(cls) -> "CourseConditions":
        return cls(RoughModel.preset("light"),
                   TreeModel.preset("decorative"),
                   GreenModel.preset("medium"))

    @classmethod
    def tournament(cls) -> "CourseConditions":
        return cls(RoughModel.preset("medium"),
                   TreeModel.preset("typical"),
                   GreenModel.preset("fast"))

    @classmethod
    def major_championship(cls) -> "CourseConditions":
        return cls(RoughModel.preset("us_open"),
                   TreeModel.preset("dense"),
                   GreenModel.preset("tournament"),
                   pin_position_difficulty=0.85)

    def to_yaml(self, path: Path) -> None: ...
    @classmethod
    def from_yaml(cls, path: Path) -> "CourseConditions": ...
```

The `shot_model.shot_model` module takes `CourseConditions` as a parameter and applies the relevant modifiers based on the starting lie. The MDP transition function also takes conditions, so the same hole + same player can be analyzed under multiple condition presets.

**Acceptance**: identical `PlayerProfile` + `HoleGeometry` produce demonstrably different optimal policies under `benign()` vs `major_championship()` conditions. Specifically: on a hole with right-side rough between fairway and water, increasing rough severity should shift the optimal aim closer to fairway center (because the cost of missing fairway has gone up).

### 3.8 `course.library` (NEW)

A curated collection of well-known holes shipped with the package, each as a `(GeoJSON, conditions.yaml, README.md)` bundle in `data/courses/classics/`.

```python
def list_classics() -> list[ClassicHoleMeta]:
    """Returns metadata for all shipped classic holes."""

def load_classic(name: str) -> ClassicHole:
    """Returns (ProjectedHole, default CourseConditions, metadata)."""

@dataclass
class ClassicHole:
    name: str
    description: str
    par: int
    yardage: int
    course: str
    geometry: ProjectedHole
    default_conditions: CourseConditions
    strategic_notes: str  # what the hole tests, from README
```

**Phase 2 starter set:**

1. **`sawgrass_17`** — TPC Sawgrass #17, "Island Green." 137-yard par 3 over water.
   _Tests:_ carry-versus-bail-out, lateral dispersion's effect on aim point.
   _Default:_ tournament conditions, stimp 12.
   _Expected optimal:_ for a high-dispersion player, aim shifts to the front-left bail-out, not the pin.

2. **`augusta_13`** — Augusta National #13, "Azalea." 510-yard par 5 with a creek fronting the green.
   _Tests:_ lay-up-vs-go-for-it, the lay-up distance decision (full wedge vs. partial), risk-reward.
   _Default:_ tournament, stimp 13, dense trees.
   _Expected optimal:_ tour-level players go for it; high-handicaps lay up to 80-100 yards.

3. **`pebble_7`** — Pebble Beach #7. 106-yard par 3 to a tiny green over ocean.
   _Tests:_ penal small-target strategy, wind sensitivity (when wind ships).
   _Default:_ tournament, fast greens.

4. **`road_hole_17`** — St Andrews #17. 495-yard par 4 with the hotel/wall right of the tee and the road bunker greenside.
   _Tests:_ route choice (aggressive over the hotel vs. safe left), approach to a green with severe back trouble.
   _Default:_ links (firm fairways, fast greens, low rough).

5. **`cypress_16`** — Cypress Point #16. 231-yard par 3 over Pacific Ocean.
   _Tests:_ long carry risk/reward, bail-out left option.
   _Default:_ tournament.

Each `data/courses/classics/<name>/` directory contains:

- `hole.geojson` — hand-traced from satellite imagery; provenance in metadata.
- `conditions.yaml` — default conditions (typically `tournament`).
- `README.md` — strategic interest, expected optimal play, citations.

`# AGENT-NOTE:` These are reference holes for testing and demos. They are committed to the repo and serve as integration tests: "the optimal strategy on Augusta 13 for a tour-level player under tournament conditions should be to go for the green." Build at least one integration test per classic hole that asserts a qualitative property of the optimal policy.

### 3.9 `course.features`

Computes feature-engineered diagnostics from a state:

```python
@dataclass
class StateFeatures:
    distance_to_pin: float
    distance_to_nearest_water: float | None
    distance_to_nearest_ob: float | None
    distance_to_nearest_bunker: float | None
    distance_to_nearest_trees: float | None
    effective_fairway_width_at_landing: float
    risk_adjusted_landing_zone_width: float  # à la Fawcett
    green_depth_remaining: float | None      # how much green to work with
    green_depth_effective: float | None      # after applying GreenModel modifier
```

Used for both UI display and as inputs to heuristic/approximate policies for benchmarking.

### 3.10 `maps.google_maps_client`

Google Maps Platform integration with billing safety:

- API key from environment variable `GOOGLE_MAPS_API_KEY`.
- **Hard daily request cap** configured in settings (default: 100), separate from Google's quota.
- All tile/imagery requests cached in SQLite (`tile_cache.py`) keyed by (lat, lon, zoom).
- Fallback to OpenStreetMap if API unavailable or quota exceeded.

`# AGENT-NOTE:` Never commit the API key. Provide `.env.example` only. The CI on BRICK should run all map-related tests in mock mode with cached fixtures, not live API.

### 3.11 `mdp.value_iteration`

Vectorized value iteration. **This is the performance-critical module.**

```python
class HoleMDP:
    def __init__(self,
                 lie_raster: LieRaster,
                 profile: PlayerProfile,
                 conditions: CourseConditions,
                 action_set: ActionSet,
                 hazard_rules: HazardRules):
        ...

    def bellman_backup(self) -> np.ndarray:
        """Single sweep. Vectorized over all (state, action) pairs."""

    def solve(self, tol: float = 1e-3, max_iter: int = 200,
              warm_start: np.ndarray | None = None) -> SolveResult: ...

    def optimal_action(self, state: State) -> Action: ...
    def expected_strokes(self, state: State) -> float: ...
    def reanalyze(self, new_conditions: CourseConditions) -> SolveResult:
        """Re-solve with same geometry+profile but different conditions, warm-started."""
```

**Performance targets** (on a typical par-4, 250×400 yard bounding box, 1-yard grid, ~12 clubs × ~20 aim angles):

- Cold solve: under 30s on a modern desktop.
- Warm-start re-solve after condition change: under 5s.

If hit, advance to next phase. If missed, profile and optimize before adding features.

`# AGENT-NOTE:` The big-O killer is the action loop. Vectorize the expected-value computation across all aim angles for a given (state, club) by sampling once per club per state and reweighting by aim direction analytically where possible. Pre-compute lie-lookup as integer indexing, not function calls.

### 3.12 `mdp.transition`

```python
def sample_transitions(state: State, action: Action,
                       profile: PlayerProfile,
                       conditions: CourseConditions,
                       raster: LieRaster,
                       hazard_rules: HazardRules,
                       n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Returns (n_samples,) array of (next_state, additional_cost) tuples.
       Applies conditions modifiers based on starting lie."""
```

Hazard rules (drop locations, stroke penalties) are configurable; default to standard USGA.

### 3.13 `ui.app`

PyQt6 main application. Tabs:

1. **Player Profile** — sliders/spinboxes for each club skill, putting curve editor (drag points on a make% vs distance plot), profile save/load.
2. **Course Library** — browser for classic holes (thumbnails, descriptions, strategic notes). One-click load.
3. **Course Editor** — map view with polygon-tracing tools, lie-class palette (including trees), save as GeoJSON to `data/courses/user/`.
4. **Conditions** — sliders / preset buttons for rough severity, tree penalization, green stimp, pin difficulty. Live preview of how the value function shifts as conditions change (this is the "feature" of v2 — see how Augusta 13 plays in benign vs major conditions in real time).
5. **Strategy** — select hole, select player profile, select conditions, click "Solve." Shows value heatmap overlay on map, optimal-aim arrow from current ball location, expected-strokes distribution histogram, alternative-strategy comparison panel.
6. **Real-Time** (Phase 5) — GPS-based current location, recommended shot.

### 3.14 `ui.conditions_panel` (NEW)

Dedicated widget for tuning `CourseConditions`. Sliders and preset buttons; emits a `conditionsChanged` signal that the strategy view subscribes to for live re-solving (warm-started for responsiveness).

### 3.15 `ui.library_browser` (NEW)

Browser for shipped classic holes plus user-traced holes. Thumbnail grid with hole names, pars, yardages, and strategic-interest descriptions.

### 3.16 `ui.map_widget`

`QtWebEngineView` loading `web/index.html`. JavaScript handles map rendering (Google Maps JS API or Leaflet+tiles), polygon tracing, and overlay rendering. Python ↔ JS communication via `QWebChannel`.

`# AGENT-NOTE:` Keep the JS side dumb — it handles user interactions and renders what Python tells it. All math, MDP, and solving stays in Python. Don't put policy logic in JS.

---

## 4. Data Schemas

All persistent data is YAML or GeoJSON, human-readable, version-tagged.

### 4.1 `player_profile.yaml`

See §3.3.

### 4.2 `hole.geojson`

GeoJSON FeatureCollection with custom properties:

```json
{
  "type": "FeatureCollection",
  "properties": {
    "schema_version": "1.1",
    "name": "Pebble Beach #7",
    "par": 3,
    "yardage": 106,
    "course": "Pebble Beach Golf Links",
    "hole_number": 7,
    "utm_zone": "10N",
    "pin": [-121.9486, 36.5680],
    "tee": [-121.9487, 36.5685],
    "provenance": "Traced from Google Maps satellite imagery, 2024-XX-XX",
    "default_conditions_file": "conditions.yaml"
  },
  "features": [
    {"type": "Feature",
     "geometry": {"type": "Polygon", "coordinates": [...]},
     "properties": {"lie_class": "fairway"}},
    {"type": "Feature",
     "geometry": {"type": "Polygon", "coordinates": [...]},
     "properties": {"lie_class": "trees"}},
    {"type": "Feature",
     "geometry": {"type": "Polygon", "coordinates": [...]},
     "properties": {"lie_class": "water"}}
  ]
}
```

Valid `lie_class` values: `tee`, `fairway`, `rough`, `trees`, `sand`, `water`, `ob`, `green`. (`holed` is computed at runtime from the pin location.)

### 4.3 `baseline.yaml`

See §3.2.

### 4.4 `conditions.yaml` (NEW)

```yaml
schema_version: "1.0"
name: Augusta National - Masters Sunday
notes: Heavy rough not really applicable; greens 13.5+ stimp; pin positions punishing

rough:
  severity: 0.45 # Augusta's rough is moderate, but second cut is real
  # or: preset: medium

trees:
  penalization: 0.75
  # or: preset: dense

greens:
  stimp: 13.5
  # or: preset: masters

pin_position_difficulty: 0.85
```

### 4.5 Classic-hole bundle layout

```
data/courses/classics/sawgrass_17/
├── hole.geojson
├── conditions.yaml      # default conditions for this hole
└── README.md            # strategic notes, expected optimal play, citations
```

---

## 5. Implementation Phases & Acceptance Criteria

Each phase produces a working, tested, mergeable PR to `dev`. Do **not** start phase N+1 until phase N's acceptance criteria are met.

### Phase 1 — Core shot model + MDP solver + basic conditions, headless

- [ ] `shot_model.distributions`, `shot_model.baseline`, `shot_model.player_profile`, `shot_model.putting` implemented with full type hints and docstrings.
- [ ] `course.conditions` implemented with at least the `RoughModel` and `GreenModel` working end-to-end (`TreeModel` can be a stub that treats trees as heavy rough).
- [ ] `course.rasterize` working on a hand-coded synthetic hole defined in Python.
- [ ] `mdp.value_iteration` solves the synthetic hole and returns sensible value function, accepting `CourseConditions`.
- [ ] `cli.py` runs end-to-end: takes a player YAML, a synthetic hole spec, and a conditions YAML; outputs JSON with optimal aim and expected strokes.
- [ ] Tests: ≥85% line coverage, including property tests on the distribution and value-function invariants.
- [ ] Benchmark: synthetic par-4 cold solve under 30s on BRICK.

**Sanity-check integration tests (must all pass before merging):**

- Increasing `sigma_lat` for driver → optimal aim on a synthetic hole with right-side water shifts left.
- Increasing `rough.severity` → optimal approach-shot strategy from rough becomes more conservative (smaller percentage of the green targeted).
- Increasing `greens.stimp` → expected strokes from 30 ft on the green increases (harder to lag close, more 3-putts).

### Phase 2 — GeoJSON I/O + classic holes library + trees model

- [ ] `course.geometry` projection in/out is round-trip exact to <1cm.
- [ ] `course.course_io` reads/writes GeoJSON conforming to §4.2 schema v1.1.
- [ ] `course.features` computes all listed features and they're displayed in the CLI output.
- [ ] `course.library` implemented with `list_classics()` and `load_classic()`.
- [ ] All five Phase-2 classic holes (`sawgrass_17`, `augusta_13`, `pebble_7`, `road_hole_17`, `cypress_16`) traced, committed, and pass their qualitative integration tests.
- [ ] `TreeModel` fully implemented including forced-punch-out behavior at high penalization.
- [ ] CLI gains `--classic` flag: `sg-optimizer solve --classic sawgrass_17 --profile dieter.yaml --conditions tournament`.

### Phase 3 — PyQt6 player profile editor + conditions panel

- [ ] Profile editor tab works standalone.
- [ ] Conditions panel tab works standalone, with preset buttons and individual sliders.
- [ ] Live preview: when conditions change, a small status display shows the recomputed expected strokes for a fixed reference state (e.g., from the tee on the currently loaded hole). Uses warm-started re-solve.

### Phase 4 — Library browser + map widget + tracing

- [ ] Library browser tab shows shipped classic holes with thumbnails and metadata.
- [ ] QtWebEngine widget loads imagery (Google or OSM fallback) for a given lat/lon.
- [ ] User can place polygon vertices, assign lie class from a palette (including trees), save GeoJSON to `data/courses/user/`.
- [ ] Existing GeoJSON loads and renders correctly.
- [ ] Google Maps API key handling, billing cap, and tile cache all work.

### Phase 5 — Strategy view + real-time

- [ ] Strategy tab shows the value-function heatmap overlaid on the map.
- [ ] Click on map → places ball state → shows optimal aim arrow + expected strokes distribution.
- [ ] Comparison panel: optimal vs. user-specified alternative strategy, both with expected-strokes distributions.
- [ ] **Headline feature**: changing conditions while viewing strategy live-updates the heatmap. The user can drag the rough-severity slider and watch the optimal-aim field shift.
- [ ] Phase 5b (stretch): GPS-based current location for real-time use during a round.

### Phase 6 — AffineDrift integration (research)

- [ ] Hook to derive shot-model parameters from AffineDrift biomechanical simulation output.
- [ ] Documentation: "from drift/control decomposition to dispersion parameters."

---

## 6. Pitfalls and Things To Watch

These are the failure modes I've seen bite people doing this kind of work. The agent should review this list before writing code in each module.

1. **Confusing the population SG baseline with the player's value function.** $B(s)$ from Broadie's data is what tour pros _do_; $V^*(s)$ is what an optimal player _can achieve_. Initialize $V$ to something reasonable but iterate via Bellman. Don't use the baseline as $V$.

2. **Coordinate frame confusion.** Three frames in play: WGS84 lat/lon (input/output), UTM meters (internal geometry), aim-frame yards (shot model). Write explicit conversion utilities and unit-test them with at-least-three known points. Mixing frames produces silent miscalibrations of the optimal aim by yards.

3. **Hazard discontinuities.** $V$ has cliffs at hazard boundaries. Any interpolation across a hazard boundary is wrong. Either use grid-cell lookup (no interpolation) at the hazard margin, or refine the grid locally.

4. **Aim discretization too coarse.** A 5° aim grid will miss optimal lines that thread 3° gaps. Adaptive resolution near hazards, or finer global grid in Phase 1, then optimize.

5. **Estimation error masquerading as signal.** With ≤100 shots per club, dispersion estimates have wide uncertainty. Don't ship "this is the optimal aim" without uncertainty bounds in Phase 2+.

6. **Putting wedged into the swing model.** Keep `shot_model.putting` strictly separate. The greenside MDP transitions use the putting model with the `GreenModel` modifier; the through-the-bag transitions use the swing model with `RoughModel` modifier. Don't unify these prematurely.

7. **Action space explosion.** (clubs) × (aim angles) × (shot types) gets big fast. For Phase 1, prune obviously dominated actions. Cache the per-club distance distribution.

8. **Mean-strokes vs. tournament objectives.** The Bellman recursion as written optimizes expected strokes. Match play and cut-line pressure want CVaR or quantile objectives. Design `risk_objectives.py` interface so this swaps cleanly in Phase 2+.

9. **Stationarity.** Player skills drift. Profile file records `last_updated`; UI warns if >3 months stale.

10. **Google Maps billing.** Easy to rack up charges accidentally. Hard cap in code, mock in tests, document setup carefully.

11. **Vectorization correctness.** Vectorizing the Bellman backup is essential for speed but easy to get subtly wrong. Keep a slow scalar reference implementation and property-test that the vectorized version matches on small grids.

12. **Per-lie shot model variation.** A 7-iron from rough has _different_ dispersion than a 7-iron from fairway, not just lower mean distance. This is exactly what the `RoughModel.dispersion_multiplier()` exists for. Make sure the shot model honors the **starting** lie, not the **landing** lie, when applying the modifier.

13. **Tight coupling between UI and solver.** The solver must run headless (CLI). The UI is a consumer, not a participant.

14. **Conditions baked into geometry.** The single biggest v2 trap. Do **not** encode "this is heavy rough" as a separate lie class in the raster. Lie classes are geometric (where the ball is); conditions are configurable (how punishing the lies are). If you find yourself wanting `rough_light`, `rough_medium`, `rough_heavy` as separate lie codes, stop and re-read §1.4.

15. **Tree recovery non-determinism.** At `tree.penalization = 1.0`, the punch-out destination is not unique — it depends on where the nearest playable lie is. The recovery model needs access to the lie raster, not just the current state. Pass the raster into `TreeModel.recovery_distribution()`.

16. **Conditions YAML overwrite risk.** When a user adjusts conditions in the UI on a classic hole, do **not** overwrite the shipped `data/courses/classics/<name>/conditions.yaml` — write to a user-config location instead. The shipped defaults are part of the integration-test contract.

17. **Stimp coupling with approach shot model.** Faster greens reduce effective landing-window depth. Make sure both the putting model AND the approach-shot decision (which green-depth to target) honor `GreenModel.effective_green_depth_multiplier()`. Easy to wire only one of them.

18. **Float comparison in geometry.** `pyproj` round-trips are not bit-exact; use tolerances. Don't use `==` on projected coordinates.

19. **JSON vs YAML vs pickled state.** All persistent player and course data is YAML/JSON. Solver state (value functions) is allowed to be `.npz`. Never pickle anything that will survive a Python version upgrade.

20. **Classic-hole provenance.** When tracing classic holes, satellite imagery is copyrighted. Record source and date in `hole.geojson` metadata. The traced polygons (your representations) are fine, but document the workflow.

---

## 7. Coding Conventions

- Python ≥3.11.
- `pyproject.toml` only — no `setup.py`, no `requirements.txt`.
- Black (line length 100), Ruff, mypy strict on `src/`.
- pytest with `pytest-hypothesis` for property tests.
- Type hints required on all public APIs.
- Docstrings: NumPy style.
- Every module has a `README.md` and (where logic is non-trivial) an `AGENTS.md`.
- `# AGENT-NOTE:` for "be careful here" inline comments.
- Pre-commit hooks for Black, Ruff, mypy, and a tests-on-touched-files runner.
- CI on self-hosted runner (BRICK): full test suite + benchmark regression check.
- Dev-branch-first; PRs to `main` require all checks green.

---

## 8. Out of Scope (for v1)

- Multi-hole / full-round optimization. v1 is per-hole.
- Wind, elevation, lie-angle, temperature corrections beyond the basic conditions framework.
- Opponent modeling, match-play strategy (CVaR/quantile is hookable but not v1).
- Mobile deployment. v1 is desktop PyQt6.
- Multiplayer / cloud sync. All data is local.
- Live integration with launch monitors or shot trackers. Phase 2+.

---

## 9. Verbatim Task Block for Agent

The following is the message to paste into Jules / Claude Code for kickoff:

> **Task: Set up the `sg-optimizer` project, Phase 1.**
>
> You are setting up a new Python project for personalized golf strategy optimization, owned by Dieter, in the `D-sorganization` GitHub org. The full design specification (v2) lives at `docs/design/STROKES_GAINED_OPTIMIZER_SPEC.md` (this file). Read it end-to-end before writing any code.
>
> **Phase 1 deliverables (§5, Phase 1):**
>
> 1. Initialize the repository structure exactly as specified in §2.
> 2. Configure `pyproject.toml` as the single source of truth for Black, Ruff, mypy, and pytest. Inherit settings from Dieter's other repos under `D-sorganization` where reasonable.
> 3. Set up pre-commit hooks and a GitHub Actions workflow targeting the self-hosted runner on BRICK.
> 4. Implement Phase 1 modules:
>    - `shot_model.distributions` — tilted bivariate Gaussian per §3.1.
>    - `shot_model.baseline` — load PGA Tour reference from YAML. Cite all numerical sources in `docs/data_sources.md`.
>    - `shot_model.player_profile` — dataclass + YAML I/O per §3.3.
>    - `shot_model.putting` — make-pct + leave model per §3.4. Must accept a `GreenModel` argument.
>    - `course.conditions` — full implementation of `RoughModel`, `GreenModel`, `CourseConditions` per §3.7. `TreeModel` may be a partial stub that treats trees as heavy rough; full implementation comes in Phase 2.
>    - `course.rasterize` — synthetic hole rasterization (no GeoJSON yet). Lie codes per §3.6.
>    - `mdp.value_iteration` — vectorized value iteration accepting `CourseConditions`, meeting the 30s cold-solve target.
>    - `cli.py` — end-to-end headless run accepting `--profile`, `--hole-spec`, `--conditions` arguments.
> 5. Tests: aim for ≥85% line coverage. Use property tests (Hypothesis) for distribution and value-function invariants. Implement **all three** sanity-check integration tests listed in §5 Phase 1.
> 6. Populate per-directory `README.md` and `AGENTS.md` files. Use `# AGENT-NOTE:` comments at all points flagged in the design doc.
> 7. Do not start Phase 2 (classic-holes library, full trees model, GeoJSON I/O). When Phase 1 is complete and tests pass, open a PR to `dev` titled "Phase 1: shot model + MDP solver + conditions framework" and stop.
>
> **Read §6 (Pitfalls) before writing each module — it's directly relevant to the failure modes for this kind of code. Pay special attention to pitfalls #1 (baseline vs value function), #2 (coordinate frames), #6 (putting separation), #14 (conditions baked into geometry), and #17 (stimp coupling with approach shots).**
>
> **Constraints:**
>
> - Do not introduce dependencies beyond: numpy, scipy, pyyaml, pyproj, shapely, hypothesis, pytest, ruff, black, mypy. Justify any others in the PR.
> - No Qt imports in `src/sg_optimizer/{shot_model,course,mdp}/`. UI integration is Phase 3+.
> - Keep a slow scalar reference implementation of the Bellman backup alongside the vectorized one, used in tests.
> - Cite numerical sources (Broadie, Fawcett) in YAML comments and `docs/data_sources.md`. Do not invent numbers; if you must interpolate, mark with `# INTERPOLATED:`.

---

## 10. Open Questions for Dieter

1. **Reference baseline:** PGA Tour, scratch amateur, or both? Tour is more data-available; scratch is more relatable for most users.
2. **Default user for testing:** Want to define your own profile up front (driver carry, dispersion estimates, putting curve) so the Phase 1 integration tests use realistic numbers?
3. **AffineDrift integration depth:** Phase 6 placeholder, or design a `BiomechanicalShotModel` interface in `shot_model/` from the start?
4. **Repo home:** New repo under `D-sorganization`, or a subdirectory inside AffineDrift?
5. **Classic-hole tracing workflow:** Will you trace these yourself (and commit the GeoJSON), or do you want the agent to trace them from public satellite imagery as part of Phase 2? If the agent traces them, accuracy will be ±5 yards which is fine for demos but worth noting.
6. **Conditions sliders vs presets:** Should the conditions panel default to preset buttons (`benign`/`tournament`/`major`) with sliders as an "advanced" reveal, or expose all sliders by default? Affects UI design.
7. **Sample shot-data ingestion:** Worth designing a Phase 2 Arccos/Shot Scope CSV importer up front so the dataclasses are compatible with that schema?
