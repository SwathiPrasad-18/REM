#!/usr/bin/env python3
"""Conservative parser for REM raw HTML malware-analysis reports (CAPE, Tencent HABO, VT wrappers).
Only extracts evidence present in the HTML. It does not infer ground-truth labels or runtime API calls.
Requires: beautifulsoup4 (pip install beautifulsoup4)
"""
import argparse, csv, json, re, zipfile, hashlib
from pathlib import Path
from bs4 import BeautifulSoup

HASH_RE = re.compile(r'(?i)\b[a-f0-9]{64}\b')
IP_RE = re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b')
URL_RE = re.compile(r'https?://[^\s\"\'<>]+', re.I)
DOMAIN_RE = re.compile(r'(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|edu|gov|io|co|uk|cn|ru|info|biz|xyz|top|site|cloud|app|dev|me|in|de|jp|au|us|local)\b')

def clean(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()
def uniq(xs):
    out=[]; seen=set()
    for x in xs:
        x=clean(x)
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out
def table_rows(table):
    rows=[]
    for tr in table.find_all('tr'):
        cells=[clean(c.get_text(' ',strip=True)) for c in tr.find_all(['th','td'], recursive=False)]
        if not cells: cells=[clean(c.get_text(' ',strip=True)) for c in tr.find_all(['th','td'])]
        if cells: rows.append(cells)
    return rows
def next_section_text(heading):
    parts=[]
    for el in heading.next_elements:
        if el is heading: continue
        if getattr(el,'name',None) in ('h1','h2','h3','h4','h5'):
            break
        if getattr(el,'name',None) in ('script','style','nav'): continue
        if isinstance(el,str):
            x=clean(el)
            if x and (not parts or parts[-1] != x): parts.append(x)
    return clean(' '.join(parts))
def get_label_values(soup):
    d={}
    for table in soup.find_all('table'):
        rows=table_rows(table)
        for row in rows:
            # key/value rows are often pairs; preserve the first value for known metadata
            for i in range(0,len(row)-1,2):
                k=clean(row[i]).rstrip(':').lower()
                v=clean(row[i+1])
                if k and v and len(k)<40 and k not in d: d[k]=v
    # HABO sometimes renders metadata as table cell pairs and/or plain text
    txt=clean(soup.get_text(' ',strip=True))
    for key, pat in [('sha256',r'SHA256\s*:\s*([a-f0-9]{64})'),('file_type',r'File type\s*:\s*([^|]{1,80}?)(?=\s+(?:Copyright|Version|Shell or compiler|Key behaviour|Behaviour)\s*:|$)'),('copyright',r'Copyright\s*:\s*(.{1,100}?)(?=\s+(?:Version|Shell or compiler|Key behaviour|Behaviour)\s*:|$)'),('version',r'Version\s*:\s*(.{1,100}?)(?=\s+(?:Shell or compiler|Key behaviour|Behaviour)\s*:|$)')]:
        m=re.search(pat,txt,re.I)
        if m and key not in d: d[key]=clean(m.group(1))
    return d
def parse_file(name, raw):
    soup=BeautifulSoup(raw.decode('utf-8','replace'),'html.parser')
    text=clean(soup.get_text(' ',strip=True))
    title=clean(soup.title.get_text(' ',strip=True)) if soup.title else ''
    basename=Path(name).name
    filehash=HASH_RE.search(basename)
    filehash=filehash.group(0).lower() if filehash else ''
    md=get_label_values(soup)
    sha=md.get('sha256','')
    if not re.fullmatch(r'(?i)[a-f0-9]{64}',sha or ''): sha=filehash
    if not sha:
        m=HASH_RE.search(text); sha=m.group(0).lower() if m else ''
    if 'habo analysis system' in text.lower(): fmt='Tencent HABO'
    elif 'cape sandbox' in title.lower() or 'cuckoo version' in text.lower(): fmt='CAPE/Cuckoo'
    elif 'virustotal report' in title.lower() or 'zenbox verdict' in text.lower(): fmt='VirusTotal HTML wrapper'
    else: fmt='Unknown/other HTML'
    started=md.get('started on','')
    completed=md.get('completed on','')
    duration=md.get('duration','')
    filename=md.get('filename','') or md.get('file name','')
    filetype=md.get('file type','') or md.get('type','')
    filesize=md.get('file size','')
    md5=md.get('md5',''); sha1=md.get('sha1','')
    # headings and their content, avoiding navigation menu labels as evidence
    section={}
    for h in soup.find_all(['h1','h2','h3','h4','h5']):
        label=clean(h.get_text(' ',strip=True)).lower()
        if label and label not in section: section[label]=next_section_text(h)
    # report verdict is a source-reported verdict, never ground truth
    verdict=''
    m=re.search(r'(?i)\b(?:Zenbox\s+Verdict|Verdict)\s*[: ]+([\w -]{1,50}?)(?=\s+(?:Report generated|Overview|Indicators|Mitre Attack|Network Info)|$)',text)
    if m: verdict=clean(m.group(1))
    # collect observable network indicators from visible text/tables
    ips=uniq(IP_RE.findall(text))
    urls=uniq(URL_RE.findall(text))
    domains=uniq([d.lower() for d in DOMAIN_RE.findall(text) if not re.fullmatch(r'\d+(?:\.\d+)+',d)])
    # section-backed signatures / ATT&CK only; do not use nav labels as signatures
    sig=[]; mitre=[]; behavior_tags=[]; process_details=[]; dropped=[]
    for key in ('signatures','behavior summary','behaviour','key behaviour','mitre attack','indicators','dropped files','payloads','processes'):
        val=section.get(key,'')
        if val:
            if key=='signatures': sig.extend([x for x in re.split(r'\s{2,}|\s*;\s*',val) if x and x.lower() not in ('info','file','network','screenshots','behavior','behaviour')])
            if key in ('behaviour','key behaviour','behavior summary'): behavior_tags.extend([x for x in re.split(r'\s{2,}|\s*;\s*',val) if x])
            if key=='mitre attack': mitre.extend(re.findall(r'\bT\d{4}(?:\.\d{3})?\b[^\n]{0,100}',val))
            if key in ('dropped files','payloads'): dropped.extend([x for x in re.split(r'\s{2,}',val) if x])
            if key=='processes': process_details.append(val[:10000])
    # HABO behavior tables contain explicit Behaviour and Detail info labels.
    events=[]
    for ti,table in enumerate(soup.find_all('table')):
        rows=table_rows(table)
        flat=' '.join(c for r in rows for c in r)
        if re.search(r'(?i)Behaviour\s*:|Detail info\s*:',flat):
            current=''
            for row in rows:
                for i,cell in enumerate(row):
                    if re.match(r'(?i)^Behaviour\s*:?$',cell):
                        if i+1<len(row): current=row[i+1]
                    elif re.match(r'(?i)^Detail info\s*:?$',cell):
                        detail=' | '.join(x for x in row[i+1:] if x)
                        if detail or current:
                            events.append({'event_type':current,'detail':detail,'evidence_source':'HABO explicit behavior table'})
            # flattened cells may contain repeated label/value pairs
            for row in rows:
                i=0
                while i<len(row)-1:
                    if re.match(r'(?i)^Behaviour\s*:?$',row[i]):
                        ev=row[i+1]; detail=''
                        if i+2<len(row) and re.match(r'(?i)^Detail info\s*:?$',row[i+2]): detail=' | '.join(row[i+3:])
                        if ev and (detail or ev): events.append({'event_type':ev,'detail':detail,'evidence_source':'HABO explicit behavior table'})
                    i+=1
    # CAPE structured network tables and process headings are evidence, not inferred API calls
    if fmt=='CAPE/Cuckoo':
        for h in soup.find_all(['h4','h5']):
            label=clean(h.get_text(' ',strip=True))
            if re.search(r'(?i)PID\s*:',label) and re.search(r'(?i)Full Path:',label):
                process_details.append(label)
        for h in soup.find_all(['h4','h5']):
            label=clean(h.get_text(' ',strip=True))
            if label.lower() in ('behavior summary','signatures'):
                body=next_section_text(h)
                if label.lower()=='behavior summary' and body:
                    events.append({'event_type':'CAPE behavior summary','detail':body[:5000],'evidence_source':'CAPE behavior summary section'})
                if label.lower()=='signatures' and body:
                    sig.extend([x for x in re.split(r'\s{2,}',body) if x and x.lower() not in ('info','file','network','screenshots')])
    # Filter generic section/nav artifacts and de-duplicate
    sig=uniq([x for x in sig if len(x)>2 and x.lower() not in {'info','file','network','screenshots','behavior','behaviour','volatility','category'}])
    behavior_tags=uniq([x for x in behavior_tags if len(x)>2 and x.lower() not in {'info','file','network','screenshots','behavior','behaviour'}])
    # ATT&CK identifiers may be embedded in nested list elements not captured by heading traversal.
    for m in re.finditer(r'\bT\d{4}(?:\.\d{3})?\b', text):
        left=max(0,m.start()-100); right=min(len(text),m.end()+160)
        mitre.append(clean(text[left:right]))
    mitre=uniq(mitre)
    events_unique=[]; seen=set()
    for e in events:
        k=(e['event_type'],e['detail'])
        if k not in seen: seen.add(k); events_unique.append(e)
    # file metadata fallback from explicit key-value rows
    if not filename:
        m=re.search(r'(?i)Filename\s+(.{1,120}?)\s+(?:File Type|File Size|MD5|SHA1|SHA256)',text)
        if m: filename=clean(m.group(1))
    if not filesize:
        m=re.search(r'(?i)File Size\s+(\d+\s+bytes?)',text)
        if m: filesize=m.group(1)
    parse_status='parsed_with_evidence' if (sha or events_unique or ips or sig or filename) else 'parsed_limited_evidence'
    rec={
      'report_file':basename,'report_format':fmt,'report_title':title,'sha256':sha,'sha256_from_filename':filehash,
      'filename':filename,'file_type':filetype,'file_size':filesize,'md5':md5,'sha1':sha1,
      'analysis_started':started,'analysis_completed':completed,'analysis_duration':duration,
      'report_verdict':verdict,'ground_truth_label':'','runtime_api_calls_json':'[]',
      'runtime_api_calls_status':'not_exposed_as_structured_events_in_parsed_report',
      'behavior_tags_json':json.dumps(behavior_tags,ensure_ascii=False),'signatures_json':json.dumps(sig,ensure_ascii=False),
      'mitre_techniques_json':json.dumps(mitre,ensure_ascii=False),'process_evidence_json':json.dumps(uniq(process_details),ensure_ascii=False),
      'dropped_payload_evidence_json':json.dumps(uniq(dropped),ensure_ascii=False),'ips_json':json.dumps(ips,ensure_ascii=False),
      'domains_json':json.dumps(domains,ensure_ascii=False),'urls_json':json.dumps(urls,ensure_ascii=False),
      'behavior_event_count':len(events_unique),'network_ip_count':len(ips),'domain_count':len(domains),'url_count':len(urls),
      'signature_count':len(sig),'mitre_count':len(mitre),'html_bytes':len(raw),'parse_status':parse_status,
      'source_verdict_is_ground_truth':'NO','notes':'Empty field means not extracted/present; runtime API calls deliberately not inferred from imports or generic text.'
    }
    for i,e in enumerate(events_unique): e.update({'report_file':basename,'sha256':sha,'event_index':i+1})
    return rec,events_unique

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',help='ZIP containing raw HTML reports or a directory'); ap.add_argument('-o','--output',default='REM_Parser_Output'); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    sources=[]
    p=Path(args.input)
    if p.is_file() and zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            sources=[(n,z.read(n)) for n in z.namelist() if n.lower().endswith(('.html','.htm')) and not n.endswith('/')]
    elif p.is_dir():
        sources=[(str(f),f.read_bytes()) for f in p.rglob('*') if f.suffix.lower() in ('.html','.htm')]
    else: raise SystemExit('Input must be a ZIP or directory')
    reports=[]; all_events=[]
    for name,raw in sources:
        try:
            rec,events=parse_file(name,raw); reports.append(rec); all_events.extend(events)
        except Exception as e:
            reports.append({'report_file':Path(name).name,'report_format':'parse_error','sha256':'','parse_status':'error','notes':f'{type(e).__name__}: {e}'})
    def write_csv(path, rows):
        if not rows: path.write_text('',encoding='utf-8'); return
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    write_csv(out/'report_features.csv',reports)
    write_csv(out/'behavior_events.csv',all_events)
    log=[]
    for r in reports:
        log.append({'report_file':r.get('report_file',''),'report_format':r.get('report_format',''),'sha256':r.get('sha256',''),'parse_status':r.get('parse_status',''),'behavior_event_count':r.get('behavior_event_count',0),'network_ip_count':r.get('network_ip_count',0),'domain_count':r.get('domain_count',0),'url_count':r.get('url_count',0),'signature_count':r.get('signature_count',0),'mitre_count':r.get('mitre_count',0),'notes':r.get('notes','')})
    write_csv(out/'extraction_log.csv',log)
    formats={}
    for r in reports: formats[r.get('report_format','unknown')]=formats.get(r.get('report_format','unknown'),0)+1
    valid=[r for r in reports if r.get('parse_status')!='error']
    def count(field): return sum(bool(r.get(field)) for r in valid)
    quality={
      'input':str(p.name),'html_reports_found':len(sources),'reports_parsed':len(valid),'parse_errors':len(reports)-len(valid),
      'formats':formats,'reports_with_sha256':count('sha256'),'reports_with_filename':count('filename'),
      'reports_with_behavior_events':sum(int(r.get('behavior_event_count',0) or 0)>0 for r in valid),
      'reports_with_network_ips':sum(int(r.get('network_ip_count',0) or 0)>0 for r in valid),
      'reports_with_domains':sum(int(r.get('domain_count',0) or 0)>0 for r in valid),
      'reports_with_urls':sum(int(r.get('url_count',0) or 0)>0 for r in valid),
      'reports_with_signatures':sum(int(r.get('signature_count',0) or 0)>0 for r in valid),
      'reports_with_mitre':sum(int(r.get('mitre_count',0) or 0)>0 for r in valid),
      'total_behavior_events':len(all_events),
      'critical_limitations':['No ground-truth labels assigned.','Report verdicts remain separate from ground truth.','Runtime API call sequences are not inferred; this report set does not expose a consistently structured API-event trace to this parser.','IP/domain/URL regex extraction can include benign sandbox infrastructure or text mentions; treat as indicators for review, not confirmed malicious C2.','Behavior extraction is report-format dependent and requires manual spot-checking before model training.']
    }
    (out/'quality_summary.json').write_text(json.dumps(quality,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'README.txt').write_text('REM raw report parser output\n\nFiles:\n- report_features.csv: one row per HTML report; list-valued fields are JSON strings.\n- behavior_events.csv: behavior evidence extracted from explicit HABO behavior tables and selected CAPE behavior summaries.\n- extraction_log.csv: per-report extraction counts and status.\n- quality_summary.json: aggregate extraction coverage and limitations.\n- parse_reports.py: reproducible parser.\n\nRun: python parse_reports.py Raw_Reports.zip -o REM_Parser_Output\nDependency: beautifulsoup4\n\nImportant: this is an extraction baseline, not a training-ready labeled dataset. Ground-truth labels are intentionally blank; VirusTotal/sandbox verdicts are not ground truth. Runtime API call sequences are intentionally left empty unless structured runtime API evidence can be reliably extracted. Review the reports and spot-check extracted rows before training.\n',encoding='utf-8')
    print(json.dumps(quality,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
