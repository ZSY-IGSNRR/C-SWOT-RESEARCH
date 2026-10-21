"""Read-only manuscript/response inventory for a fivefold-CV migration."""
from pathlib import Path
import json,re,zipfile,xml.etree.ElementTree as ET,sys,os
sys.stdout.reconfigure(encoding='utf-8')
from _paths import work_root
ROOT=work_root()
OUT=ROOT/'checks/cv_migration_review';OUT.mkdir(exist_ok=True)
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
FILES=[Path(p) for p in json.loads(os.environ.get("CSWOT_REVIEW_DOCUMENTS", "[]"))]
pattern=re.compile(r'cross.valid|\bCV\b|\bfold|\bsplit|\bten\b|\b70|30%|repeat|bootstrap|sample.size|sensitivity|assignment|multistart|209|433|22,562|732|104 gauges|factor error|WSE.stratified',re.I)
for path in FILES:
    with zipfile.ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
    paragraphs=[]
    for i,p in enumerate(root.findall('.//w:body/w:p',N)):
        text=''.join(p.itertext()) if False else ''.join(p.findall('.//w:t',N)[j].text or '' for j in range(len(p.findall('.//w:t',N))))
        style=p.find('./w:pPr/w:pStyle',N)
        paragraphs.append({'index':i,'style':style.get('{'+N['w']+'}val') if style is not None else None,'text':text})
    obj={'file':str(path),'paragraph_count':len(paragraphs),'paragraphs':paragraphs,'CV_hits':[p for p in paragraphs if pattern.search(p['text'])]}
    (OUT/(path.stem+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FILE',path.name,'PARAGRAPHS',len(paragraphs),'HITS',len(obj['CV_hits']))
    if 'Response_' not in path.name:
        for p in paragraphs:
            if p['style'] and ('Heading' in p['style'] or p['style'] in ['1','2','3']):print('HEADING',p['index'],p['text'])
        for p in obj['CV_hits']:
            if p['index']>=61:print(f"[{p['index']}] {p['text']}")
    else:
        for p in obj['CV_hits']:
            if re.search(r'ten|70%|30%|433|209|22,562|732|repeated|uncompleted|not performed|future|forward',p['text'],re.I):
                print(f"[{p['index']}] {p['text'][:1800]}")
