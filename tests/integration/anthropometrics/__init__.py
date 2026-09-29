"""Integration tests for the ``anthropometrics`` package (issue #4819).

Covers cross-cutting behaviour that the per-module unit suites cannot:

* validation against the published de Leva / Dempster /
  Zatsiorsky-Seluyanov tables;
* lossless round-trips through every available engine adapter;
* end-to-end pipeline from synthetic subject through estimator and
  URDF/OpenSim/JSON adapters back to a canonical record.
"""
