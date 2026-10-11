import datetime
import hashlib
import importlib.util
import json
import pathlib
import re

BASE = pathlib.Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/validation/supplement_v6_final_graphics_review_v2')
FIGURES = BASE.parent.parent / 'supplement_report_v1/figures/comparisons'
OUTPUT = BASE / 'all_density_colorbar_diagnostic.json'
SOURCE = BASE / 'curve_strength_144_layout_diagnostic.py'
spec = importlib.util.spec_from_file_location('layout_diagnostic', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted(FIGURES.glob('*_density.svg'))
    assert len(files) == 384
    crowded = []
    for path in files:
        svg = path.read_text()
        titles = re.findall(r'<!-- (Fade · [^\n]*?) -->', svg)
        if len(titles) != 2: raise ValueError((path,titles))
        title=titles[-1]
        p,b,_=module.glyph(title,svg)
        match=re.search(r'<g id="axes_7">\s*<g id="patch_\d+">\s*<path d="M ([\d.]+) ([\d.]+)\s*L ([\d.]+) ([\d.]+)',svg)
        if match is None:raise ValueError(f'missing colorbar patch {path}')
        colorbar_left=float(match.group(1));colorbar_right=float(match.group(3))
        clearance=colorbar_left-b[2]
        if clearance<2:
            crowded.append({'svg_path':str(path),'svg_sha256':sha(path),'matching_png_path':str(path.with_suffix('.png')),'matching_pdf_path':str(path.with_suffix('.pdf')),'lower_fade_title':title,'title_glyph_bounds_pt':b,'colorbar_patch_left_pt':colorbar_left,'colorbar_patch_right_pt':colorbar_right,'horizontal_clearance_pt':clearance})
    result={'schema':'supplement-all-density-colorbar-clearance/1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':'/tmp/shot-pattern-validation-venv-20261009/bin/python '+str(pathlib.Path(__file__).resolve()),'threshold_pt':2.0,'svg_count':len(files),'crowded_count':len(crowded),'source_sha256':sha(SOURCE),'crowded':crowded}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(OUTPUT),'sha256':sha(OUTPUT),'crowded_count':len(crowded),'min_clearance_pt':min(x['horizontal_clearance_pt'] for x in crowded) if crowded else None}))


if __name__=='__main__':main()
