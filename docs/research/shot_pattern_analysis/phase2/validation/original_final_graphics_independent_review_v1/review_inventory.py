"""Read-only primary graphic hash audit and complete PNG contact sheets."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = ROOT / 'final_figures_v1'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
mfile = FIG / 'figure_manifest.json'
initial = sha(mfile)
m = json.loads(mfile.read_text())
exit_path = Path('/tmp/p2-v2-final_figures_v1-exit.json')
exit_record = json.loads(exit_path.read_text())
assert exit_record['exit_code'] == 0, exit_record
for relative, expected in m['outputs'].items():
    assert sha(FIG / relative) == expected, relative
actual = {str(p.relative_to(FIG)) for p in FIG.rglob('*') if p.is_file()}
assert actual == set(m['outputs']) | {'figure_manifest.json'}
source = Path(m['source_root'])
assert sha(source / 'matrix_plan.json') == m['plan_sha256']
for cell, expected in m['cell_manifest_sha256'].items():
    assert sha(source / cell / 'manifest.json') == expected, cell
ledgers = {}
for name in ('availability','comparison_availability','facebook_ready','facebook_overhead','facebook_pattern_density'):
    with (FIG / f'{name}.csv').open() as f:
        ledgers[name] = list(csv.DictReader(f))
assert len(ledgers['availability']) == 324
assert Counter(x['status'] for x in ledgers['availability']) == {'completed':270,'unavailable':54}
assert set(x['cell'] for x in ledgers['availability']) == set(m['cell_manifest_sha256'])
assert len(ledgers['comparison_availability']) == 36
assert Counter(x['status'] for x in ledgers['comparison_availability']) == {'completed':30,'unavailable':6}
for row in ledgers['facebook_ready']:
    if row['status'] == 'completed':
        assert row['png'] in m['outputs']
        assert row['first_cell'] in m['cell_manifest_sha256']
        assert row['second_cell'] in m['cell_manifest_sha256']
for name in ('facebook_overhead','facebook_pattern_density'):
    assert len(ledgers[name]) == 24
pngs = sorted(p for p in m['outputs'] if p.endswith('.png'))
index = []
for page, start in enumerate(range(0,len(pngs),16),1):
    canvas = Image.new('RGB',(2000,1640),'#dddddd')
    draw = ImageDraw.Draw(canvas)
    for slot, relative in enumerate(pngs[start:start+16]):
        x,y = slot%4*500,slot//4*410
        with Image.open(FIG / relative) as source_image:
            im = source_image.convert('RGB'); im.thumbnail((488,365))
            canvas.paste(im,(x+(500-im.width)//2,y))
        label=f'{start+slot+1:03d} '+relative
        draw.text((x+6,y+368),label[:67],fill='black')
        draw.text((x+6,y+383),label[67:134],fill='black')
        index.append({'index':start+slot+1,'page':page,'path':relative,'sha256':m['outputs'][relative]})
    canvas.save(OUT / f'contact_{page:02}.jpg',quality=90)
assert sha(mfile) == initial
receipt={'scope':'Byte and inventory checks; visual verdict recorded separately','manifest_sha256':initial,'exit_record':exit_record,'exit_record_sha256':sha(exit_path),'source_plan_sha256':m['plan_sha256'],'outputs_verified':len(m['outputs']),'pngs':len(pngs),'extensions':dict(Counter(Path(p).suffix for p in m['outputs'])),'cell_manifest_count':len(m['cell_manifest_sha256']),'ledger_rows':{k:len(v) for k,v in ledgers.items()},'contact_sheets':18,'png_index':index,'script_sha256':sha(Path(__file__))}
(OUT/'inventory_review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='png_index'},indent=2))
