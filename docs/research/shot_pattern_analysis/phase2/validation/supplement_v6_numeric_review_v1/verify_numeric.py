"""Read-only numerical and planned-name reconciliation; no figure access."""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

MAIN = Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift')
SOURCE = Path('/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift')
PHASE = MAIN / 'docs/research/shot_pattern_analysis/phase2'
OUTPUT = PHASE / 'supplement_report_v1'
REFERENCE = PHASE / 'validation/protocol_name_recovery_v6/expected_numeric_hashes.json'
START = PHASE / 'validation/supplement_rendering_revision_v6/start.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

reference_bytes = REFERENCE.read_bytes()
committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(REFERENCE.relative_to(MAIN))], cwd=MAIN)
require(reference_bytes == committed, 'Expected numeric reference differs from committed bytes')
expected = json.loads(reference_bytes)
require(len(expected) == 7, 'Expected exactly seven numerical artifacts')
start_sha = sha(START)
start = json.loads(START.read_text())
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
require(source_commit == start['candidate_commit'] == '4023111d2aaf8b20449a36f8691a295f4a94ceff', 'Frozen producer commit mismatch')
sys.path.insert(0, str(SOURCE))
from src.tools.shot_pattern_analysis.matched_supplement_figures import _slug, pair_comparison_identity, pair_comparison_slug, validate_planned_figure_identities
from src.tools.shot_pattern_analysis.matched_supplement_pipeline import _report_source_hashes
sources = _report_source_hashes()
require(sources == start['producer_sha256'], 'Producer source map differs from guarded start')
actual = {name: sha(OUTPUT / name) for name in expected}
require(actual == expected, 'Numerical artifact bytes differ from committed reference')
comparisons = json.loads((OUTPUT / 'REPORT_SUPPLEMENT_COMPARISONS.json').read_text())['comparisons']
cells = json.loads((OUTPUT / 'REPORT_SUPPLEMENT.json').read_text())['unique_cells']
validate_planned_figure_identities(comparisons, cells)
identities = [pair_comparison_identity(row) for row in comparisons]
stems = [pair_comparison_slug(row) for row in comparisons]
require(len(identities) == len(set(identities)) == len(set(stems)) == 384, 'Comparison identity/stem inventory mismatch')
require(len(cells) == 26, 'Unique executed cell inventory mismatch')
counts = dict(Counter(row['comparison_kind'] for row in comparisons))
require(counts == {'across_lean': 56, 'curve_strength': 164, 'shaft_rotation_sd': 164}, 'Contrast inventory mismatch')
suffixes = [f'density_{pattern}_social.{ext}' for pattern in ('straight', 'draw', 'fade') for ext in ('png', 'pdf', 'svg')]
suffixes += [f'{kind}.{ext}' for kind in ('density', 'overhead') for ext in ('png', 'pdf', 'svg')]
suffixes += ['overhead_social.png']
pair_paths = [f'comparisons/{stem}_{suffix}' for stem in stems for suffix in suffixes]
cell_paths = [f"cells/{_slug(row.get('alias_cell') or row['cell'])}_{kind}.{ext}" for row in cells for kind in ('density', 'density_social', 'overhead', 'overhead_social') for ext in ('png', 'pdf', 'svg')]
require(len(pair_paths) == len(set(pair_paths)) == 6144, 'Pair path collision')
require(len(cell_paths) == len(set(cell_paths)) == 312, 'Cell path collision')
require(len(set(pair_paths + cell_paths)) == 6456, 'Combined planned path collision')
require(actual == {name: sha(OUTPUT / name) for name in expected}, 'Numeric artifacts changed during read')
require(sources == _report_source_hashes(), 'Producer sources changed during read')
require(sha(START) == start_sha and REFERENCE.read_bytes() == reference_bytes, 'Reference/start changed')
receipt = {'status': 'numerical_and_planned_identity_checks_passed', 'full_publication_accepted': False,
           'live_figure_assets_read': False, 'statistics_or_simulations_rerun': False,
           'producer_commit': source_commit, 'producer_source_sha256': sources,
           'start_receipt_sha256': start_sha, 'committed_reference_sha256': sha(REFERENCE),
           'actual_numeric_sha256': actual, 'all_seven_byte_identical': True,
           'comparison_counts': counts, 'distinct_pair_identities': len(set(identities)),
           'unique_executed_cells': len(cells), 'planned_pair_paths':len(pair_paths),
           'planned_cell_paths':len(cell_paths), 'planned_total_paths':6456,
           'script_sha256': sha(Path(__file__)),
           'limitation':'Live producer; planned paths are not rendered-file or visual acceptance. Terminal receipt and full output review remain pending.'}
Path('/tmp/v6_numeric_independent_review.json').write_text(json.dumps(receipt, indent=2)+'\n')
sys.stdout.write(json.dumps({'status':receipt['status'],'receipt_sha256':sha(Path('/tmp/v6_numeric_independent_review.json'))})+'\n')
