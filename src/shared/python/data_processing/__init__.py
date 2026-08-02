"""Shared data processing module extracted from Data_Processor_Integrated.py.

Provides a clean, UI-agnostic API for:
- Loading data from CSV, Excel, Parquet, DAT/DBF formats
- Transforming data (filtering, resampling, formula application)
- Analyzing data (statistics, PCA, regression, spectral analysis)
- Exporting data to multiple formats

See issue #407 for the extraction rationale.

Usage::

    from shared.python.data_processing import DataProcessor

    dp = DataProcessor()
    dp.load("data.csv")
    dp.apply_filter("butterworth", cutoff=10, order=4, columns=["sensor1"])
    dp.resample(target_rate=100)
    stats = dp.describe()
    dp.export("output.parquet")
"""

from .processor import DataProcessor

__all__ = ["DataProcessor"]
