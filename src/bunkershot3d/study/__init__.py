"""Design-study layer: DOE, sensitivity, surrogate modelling and optimisation.

This is the numerics half of the design tool described in ADR-0032. It knows
nothing about sand, wedges or solvers: everything here operates on a
:class:`~bunkershot3d.study.design_space.DesignSpace` and a vectorised
callable, so it can be tested against analytic functions with published
answers rather than against the physics it will eventually drive.

The intended workflow, cheapest step first:

1. :func:`~bunkershot3d.study.morris.morris_screening` -- ``r (D + 1)``
   evaluations to find which factors matter at all;
2. :func:`~bunkershot3d.study.sensitivity.sobol_analysis` -- ``N (D + 2)``
   evaluations to quantify first-order and total-order contributions of the
   survivors, with bootstrap intervals;
3. :class:`~bunkershot3d.study.surrogate.GaussianProcess` plus
   :func:`~bunkershot3d.study.optimisation.bayesian_optimisation` -- to search
   the reduced space without another full sweep;
4. :func:`~bunkershot3d.study.comparison.compare_designs` -- to rank the final
   candidates *with* their uncertainty.

Every entry point returns a
:class:`~bunkershot3d.study.manifest.StudyManifest` recording the 128-bit
entropy and the NumPy/SciPy versions the stream depended on, so a sweep is
replayable from its artifact.

No new dependencies: numpy and scipy only (ADR-0032 decision 8).
"""

from .analytic_benchmarks import (
    AnalyticIndices,
    ishigami,
    ishigami_indices,
    ishigami_space,
    sobol_g,
    sobol_g_indices,
    sobol_g_space,
)
from .comparison import (
    DesignComparison,
    compare_designs,
    compare_predicted_designs,
)
from .design_space import (
    DesignParameter,
    DesignSample,
    DesignSpace,
    is_power_of_two,
)
from .manifest import StudyManifest
from .morris import (
    MorrisDesign,
    MorrisResult,
    morris_design,
    morris_screening,
    morris_statistics,
)
from .optimisation import (
    AcquisitionSettings,
    BayesOptResult,
    bayesian_optimisation,
    expected_improvement,
    propose_location,
)
from .ranking import (
    BandedRanking,
    RankingVerdict,
    rank_with_bands,
)
from .rng import SeedRecord, as_generator, new_seed_record
from .sensitivity import (
    SaltelliDesign,
    SobolIndices,
    saltelli_design,
    sobol_analysis,
    sobol_indices_from_outputs,
)
from .surrogate import GaussianProcess, GPHyperparameters

__all__ = [
    "AcquisitionSettings",
    "AnalyticIndices",
    "BandedRanking",
    "BayesOptResult",
    "DesignComparison",
    "DesignParameter",
    "DesignSample",
    "DesignSpace",
    "GPHyperparameters",
    "GaussianProcess",
    "MorrisDesign",
    "MorrisResult",
    "RankingVerdict",
    "SaltelliDesign",
    "SeedRecord",
    "SobolIndices",
    "StudyManifest",
    "as_generator",
    "bayesian_optimisation",
    "compare_designs",
    "compare_predicted_designs",
    "expected_improvement",
    "is_power_of_two",
    "ishigami",
    "ishigami_indices",
    "ishigami_space",
    "morris_design",
    "morris_screening",
    "morris_statistics",
    "new_seed_record",
    "propose_location",
    "rank_with_bands",
    "saltelli_design",
    "sobol_analysis",
    "sobol_g",
    "sobol_g_indices",
    "sobol_g_space",
    "sobol_indices_from_outputs",
]
