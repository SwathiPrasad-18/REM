#!/usr/bin/env python3
"""REM Parser v5: format-aware extraction for CAPE/Cuckoo and Tencent HABO reports.
Conservative: source evidence only; never assigns ground-truth labels or infers API traces.
"""
import argparse, csv, json, re, zipfile
from pathlib import Path
from bs4 import BeautifulSoup

H64=re.compile(r'(?i)\b[a-f0-9]{64}\b'); H32=re.compile(r'(?i)\b[a-f0-9]{32}\b'); H40=re.compile(r'(?i)\b[a-f0-9]{40}\b')
IP=re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b')
DOMAIN=re.compile(r'(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|edu|gov|io|co|uk|cn|ru|info|biz|xyz|top|site|cloud|app|dev|me|in|de|jp|au|us|local)\b')
URL=re.compile(r'https?://[^\s"\'<>]+',re.I)
PLACEHOLDERS={'nothing to display','no data','no results','none','n/a','not available','no screenshots available.'}
def clean(x): return re.sub(r'\s+',' ',str(x or '')).strip()
def uniq(xs):
 out=[]; seen=set()
 for x in xs:
  x=clean(x)
  if x and x not in seen: seen.add(x); out.append(x)
 return out
def direct_cells(tr): return [clean(c.get_text(' ',strip=True)) for c in tr.find_all(['th','td'],recursive=False)]
def table_rows(table):
 out=[]
 for tr in table.find_all('tr'):
  cs=direct_cells(tr)
  if not cs: cs=[clean(c.get_text(' ',strip=True)) for c in tr.find_all(['th','td'])]
  if cs: out.append(cs)
 return out
def canonical(k):
 k=clean(k).lower().rstrip(':').replace('_',' ')
 return {'file name':'filename','file name:':'filename','file size':'file_size','file type':'file_type','sha 256':'sha256','sha-256':'sha256','sha 256:':'sha256','sha1':'sha1','sha-1':'sha1','md5':'md5','started on':'analysis_started','completed on':'analysis_completed','duration':'analysis_duration','zenbox verdict':'report_verdict','verdict':'report_verdict'}.get(k,k.replace(' ','_'))
def extract_table_metadata(soup):
 out={}
 # Key/value metadata from rows; never interpret a row containing only column headings as values.
 for table in soup.find_all('table'):
  for row in table_rows(table):
   if len(row)==2:
    k=canonical(row[0]); v=clean(row[1])
    if k in {'filename','file_size','file_type','md5','sha1','sha256','analysis_started','analysis_completed','analysis_duration','report_verdict'} and v and v.lower() not in PLACEHOLDERS and k not in out: out[k]=v
   # explicit alternating label/value pairs, but don't consume header rows
   for i in range(0,len(row)-1,2):
    k=canonical(row[i]); v=clean(row[i+1])
    if k in {'filename','file_size','file_type','md5','sha1','sha256','analysis_started','analysis_completed','analysis_duration','report_verdict'} and v and v.lower() not in PLACEHOLDERS and k not in out:
     if k not in {canonical(x) for x in row if canonical(x) in {'started_on','completed_on','duration'}}: out[k]=v
 return out
def get_cape_section(h):
 # Collect nodes after a section heading until the next section heading in document order.
 # CAPE templates vary; heading siblings may be direct nodes or nested in containers.
 out=[]
 for node in h.next_elements:
  if node is h: continue
  if getattr(node,'name',None) in {'h1','h2','h3','h4','h5'}:
   break
  if getattr(node,'name',None) in {'script','style','nav'}: continue
  if getattr(node,'name',None) in {'p','li','td','dd','div','tr','section','ul','ol'}:
   # Use text only on meaningful leaf-ish content to avoid duplicating parent + child text.
   children=getattr(node,'find_all',lambda *a,**k:[])(['p','li','td','dd'])
   txt=clean(node.get_text(' ',strip=True))
   if txt and txt.lower() not in PLACEHOLDERS and (node.name in {'p','li','td','dd'} or not children):
    out.append((node,txt))
 # de-duplicate text while preserving order
 seen=set(); unique=[]
 for node,txt in out:
  if txt not in seen: seen.add(txt); unique.append((node,txt))
 return unique
def valid_filename(v):
 v=clean(v).strip(' =:\t')
 if not v or len(v)>260 or v.lower() in PLACEHOLDERS: return ''
 if v.lower() in {'filename','file','info','network','signatures','screenshots','behavior','behaviour','category','duration','started on','completed on'}: return ''
 if re.match(r'^=\s*',v): return ''
 if v.endswith(':') and ' ' not in v: return ''
 if v.lower().startswith(('c:\\windows','c:/windows')) and not re.search(r'\.(exe|dll|sys|dat|bin|scr|ps1|js|vbs|docx?|pdf)$',v,re.I): return ''
 return v
def parse_one(name,raw):
 soup=BeautifulSoup(raw.decode('utf-8','replace'),'html.parser'); text=clean(soup.get_text(' ',strip=True)); title=clean(soup.title.get_text(' ',strip=True)) if soup.title else ''; low=text.lower(); base=Path(name).name
 if 'habo analysis system' in low: fmt='Tencent HABO'
 elif 'cape sandbox' in title.lower() or 'cuckoo version' in low: fmt='CAPE/Cuckoo'
 elif 'virustotal report' in title.lower() or 'zenbox verdict' in low: fmt='VirusTotal HTML wrapper'
 else: fmt='Unknown/other HTML'
 md=extract_table_metadata(soup)
 fname_hash=H64.search(base); fname_hash=fname_hash.group(0).lower() if fname_hash else ''
 # Hashes: explicit label first, then filename hash only. Don't take arbitrary page hashes.
 sha=clean(md.get('sha256',''))
 if not H64.fullmatch(sha):
  m=re.search(r'(?i)\bSHA256\s*:?\s*([a-f0-9]{64})',text); sha=m.group(1).lower() if m else fname_hash
 md5=clean(md.get('md5','')); sha1=clean(md.get('sha1',''))
 if not H32.fullmatch(md5):
  m=re.search(r'(?i)\bMD5\s*:?\s*([a-f0-9]{32})',text); md5=m.group(1).lower() if m else ''
 if not H40.fullmatch(sha1):
  m=re.search(r'(?i)\bSHA-?1\s*:?\s*([a-f0-9]{40})',text); sha1=m.group(1).lower() if m else ''
 # Exact metadata fallback; don't use broad non-greedy capture across unrelated headings.
 filename=valid_filename(md.get('filename',''))
 if not filename:
  for pat in [r'(?i)\bFilename\s*:?\s*([^|]{1,180}?)(?=\s+(?:File Type|File Size|MD5|SHA-?1|SHA256)\b)',r'(?i)\bFile Name\s*:?\s*([^|]{1,180}?)(?=\s+(?:File Type|File Size|MD5|SHA-?1|SHA256)\b)']:
   m=re.search(pat,text)
   if m:
    cand=valid_filename(m.group(1))
    if cand: filename=cand; break
 file_size=clean(md.get('file_size',''))
 if not file_size:
  m=re.search(r'(?i)\bFile Size\s*:?\s*(\d[\d,]*(?:\.\d+)?\s*(?:bytes?|KB|MB|GB))',text); file_size=m.group(1) if m else ''
 file_type=clean(md.get('file_type',''))
 if not file_type:
  m=re.search(r'(?i)\bFile Type\s*:?\s*([^|]{1,100}?)(?=\s+(?:File Size|MD5|SHA-?1|SHA256|Filename)\b)',text); file_type=clean(m.group(1)) if m else ''
 started=clean(md.get('analysis_started','')); completed=clean(md.get('analysis_completed','')); duration=clean(md.get('analysis_duration','')); verdict=clean(md.get('report_verdict',''))
 # Source-specific verdict fallback: only accept explicit verdict labels, not arbitrary words like malicious in prose.
 if not verdict:
  for pat in [r'(?i)\bZenbox\s+Verdict\s*[:：]?\s*([A-Za-z][A-Za-z _-]{1,35})', r'(?i)\bVerdict\s*[:：]\s*([A-Za-z][A-Za-z _-]{1,35}?)(?=\s+(?:Report generated|Analysis|Overview|Indicators|MITRE|Network|File|$))', r'(?i)\bDetection\s+verdict\s*[:：]?\s*([A-Za-z][A-Za-z _-]{1,35})']:
   vm=re.search(pat,text)
   if vm:
    candidate=clean(vm.group(1)).strip(' :-')
    if candidate and candidate.lower() not in PLACEHOLDERS and len(candidate)<40:
     verdict=candidate; break

 # CAPE: map exact table column headers to the following row, overriding contaminated generic metadata.
 if fmt=='CAPE/Cuckoo':
  if started.lower() in {'started on','category','machine','label','manager'}: started=''
  if completed.lower() in {'completed on','duration','cuckoo version','shutdown on'}: completed=''
  if duration.lower() in {'duration','cuckoo version','started on','completed on'}: duration=''
  for table in soup.find_all('table'):
   trs=table.find_all('tr')
   for i,tr in enumerate(trs[:-1]):
    labels=[clean(c.get_text(' ',strip=True)).lower().rstrip(':') for c in tr.find_all(['th','td'],recursive=False)]
    vals=[clean(c.get_text(' ',strip=True)) for c in trs[i+1].find_all(['th','td'],recursive=False)]
    if labels and vals and len(labels)==len(vals) and any(x in labels for x in ('started on','completed on','duration')):
     for lab,val in zip(labels,vals):
      if lab=='started on' and re.search(r'\d{4}-\d{2}-\d{2}',val): started=val
      elif lab=='completed on' and re.search(r'\d{4}-\d{2}-\d{2}',val): completed=val
      elif lab=='duration' and re.search(r'\d+\s*(?:seconds?|minutes?|hours?)',val,re.I): duration=val
 # CAPE signatures are alert/signature elements under the Signatures section.
 sig=[]; events=[]; process=[]
 if fmt=='CAPE/Cuckoo':
  for h in soup.find_all(['h1','h2','h3','h4','h5']):
   label=clean(h.get_text(' ',strip=True)).lower()
   if label not in {'signatures','behavior summary','behaviour summary','processes','process tree'}: continue
   items=get_cape_section(h)
   if label=='signatures':
    for node,txt in items:
     # Exclude generic nav words and placeholder rows; retain text under the signatures section.
     if txt.lower() not in PLACEHOLDERS and txt.lower() not in {'info','file','network','screenshots','behavior','behaviour','volatility','category'} and len(txt)>3:
      sig.append(txt)
   elif 'behavior summary' in label:
    # CAPE often represents behavior categories as table rows (e.g. Mutexes / Executed Commands).
    for node,txt in items:
     if node.name in {'p','li','td','dd','tr'} and txt.lower() not in PLACEHOLDERS and len(txt)>3:
      events.append({'event_type':'CAPE behavior summary','detail':txt,'evidence_source':'CAPE Behavior Summary section'})
   else:
    for node,txt in items:
     if node.name in {'p','li','td','dd','tr'} and txt.lower() not in PLACEHOLDERS: process.append(txt[:5000])
  for h in soup.find_all(['h4','h5']):
   label=clean(h.get_text(' ',strip=True))
   if re.search(r'(?i)PID\s*:',label) and re.search(r'(?i)Full Path:',label): process.append(label)
 elif fmt=='Tencent HABO':
  for table in soup.find_all('table'):
   tr_rows=table_rows(table); flat=' '.join(c for r in tr_rows for c in r)
   if not re.search(r'(?i)\bBehaviour\s*:|\bDetail info\s*:',flat): continue
   current=''
   for row in tr_rows:
    i=0
    while i<len(row):
     if re.match(r'(?i)^Behaviour\s*:?$',row[i]) and i+1<len(row): current=row[i+1]
     elif re.match(r'(?i)^Detail info\s*:?$',row[i]):
      detail=' | '.join(x for x in row[i+1:] if x)
      if current and current.lower() not in PLACEHOLDERS and detail.lower() not in PLACEHOLDERS:
       events.append({'event_type':current,'detail':detail,'evidence_source':'HABO explicit behavior table'})
     i+=1
 # Remove duplicate/placeholder event rows only; do not silently drop legitimate empty-detail events.
 clean_events=[]; seen=set(); removed=0
 for e in events:
  typ=clean(e.get('event_type')); detail=clean(e.get('detail'))
  if not typ or typ.lower() in PLACEHOLDERS or detail.lower() in PLACEHOLDERS or typ.lower() in {'behaviour','behavior','detail info'}:
   removed+=1; continue
  key=(typ,detail,e.get('evidence_source',''))
  if key in seen: continue
  seen.add(key); clean_events.append({'report_file':base,'sha256':sha,'event_index':len(clean_events)+1,'event_type':typ,'detail':detail,'evidence_source':e.get('evidence_source','')})
 ips=uniq(IP.findall(text)); urls=uniq(u.rstrip('.,);]') for u in URL.findall(text)); domains=uniq(DOMAIN.findall(text))
 mitre=uniq(m.group(0) for m in re.finditer(r'\bT\d{4}(?:\.\d{3})?\b',text))
 rec={'report_file':base,'report_format':fmt,'report_title':title,'sha256':sha,'sha256_from_filename':fname_hash,'filename':filename,'file_type':file_type,'file_size':file_size,'md5':md5,'sha1':sha1,'analysis_started':started,'analysis_completed':completed,'analysis_duration':duration,'report_verdict':verdict,'ground_truth_label':'','runtime_api_calls_json':'[]','runtime_api_calls_status':'not inferred; structured runtime trace not consistently exposed','behavior_tags_json':'[]','signatures_json':json.dumps(uniq(sig),ensure_ascii=False),'mitre_techniques_json':json.dumps(mitre,ensure_ascii=False),'process_evidence_json':json.dumps(uniq(process),ensure_ascii=False),'dropped_payload_evidence_json':'[]','ips_json':json.dumps(ips,ensure_ascii=False),'domains_json':json.dumps(domains,ensure_ascii=False),'urls_json':json.dumps(urls,ensure_ascii=False),'behavior_event_count':len(clean_events),'network_ip_count':len(ips),'domain_count':len(domains),'url_count':len(urls),'signature_count':len(uniq(sig)),'mitre_count':len(mitre),'html_bytes':len(raw),'parse_status':'parsed_with_evidence' if sha or filename or clean_events or ips or sig else 'parsed_limited_evidence','source_verdict_is_ground_truth':'NO','placeholder_events_removed':removed,'notes':'Source evidence only; indicator matches may include benign infrastructure; ground truth intentionally blank.'}
 return rec,clean_events
def write_csv(path,rows):
 if not rows: path.write_text('',encoding='utf-8'); return
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields: fields.append(k)
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',default='REM_Parser_v5_Output'); args=ap.parse_args(); inp=Path(args.input); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
 if inp.is_file() and zipfile.is_zipfile(inp):
  with zipfile.ZipFile(inp) as z: src=[(n,z.read(n)) for n in z.namelist() if n.lower().endswith(('.html','.htm')) and not n.endswith('/')]
 elif inp.is_dir(): src=[(str(p),p.read_bytes()) for p in inp.rglob('*') if p.suffix.lower() in {'.html','.htm'}]
 else: raise SystemExit('Input must be a ZIP or directory')
 reports=[]; events=[]; errors=[]
 for name,raw in src:
  try: r,e=parse_one(name,raw); reports.append(r); events.extend(e)
  except Exception as ex: errors.append({'report_file':Path(name).name,'error':f'{type(ex).__name__}: {ex}'}); reports.append({'report_file':Path(name).name,'report_format':'parse_error','sha256':'','parse_status':'error','notes':errors[-1]['error']})
 write_csv(out/'report_features.csv',reports); write_csv(out/'behavior_events.csv',events); write_csv(out/'parse_errors.csv',errors)
 log=[]
 for r in reports: log.append({k:r.get(k,'') for k in ['report_file','report_format','sha256','parse_status','filename','file_size','analysis_started','analysis_completed','analysis_duration','report_verdict','behavior_event_count','network_ip_count','domain_count','url_count','signature_count','mitre_count','placeholder_events_removed','notes']})
 write_csv(out/'extraction_log.csv',log)
 formats={}
 for r in reports: formats[r.get('report_format','unknown')]=formats.get(r.get('report_format','unknown'),0)+1
 valid=[r for r in reports if r.get('parse_status')!='error']
 def count(f): return sum(bool(r.get(f)) for r in valid)
 q={'input':inp.name,'html_reports_found':len(src),'reports_parsed':len(valid),'parse_errors':len(errors),'formats':formats,'reports_with_sha256':count('sha256'),'reports_with_filename':count('filename'),'reports_with_file_size':count('file_size'),'reports_with_md5':count('md5'),'reports_with_sha1':count('sha1'),'reports_with_analysis_started':count('analysis_started'),'reports_with_analysis_completed':count('analysis_completed'),'reports_with_analysis_duration':count('analysis_duration'),'reports_with_verdict':count('report_verdict'),'reports_with_behavior_events':sum(int(r.get('behavior_event_count',0) or 0)>0 for r in valid),'reports_with_network_ips':sum(int(r.get('network_ip_count',0) or 0)>0 for r in valid),'reports_with_domains':sum(int(r.get('domain_count',0) or 0)>0 for r in valid),'reports_with_urls':sum(int(r.get('url_count',0) or 0)>0 for r in valid),'reports_with_signatures':sum(int(r.get('signature_count',0) or 0)>0 for r in valid),'reports_with_mitre':sum(int(r.get('mitre_count',0) or 0)>0 for r in valid),'total_behavior_events':len(events),'placeholder_events_removed':sum(int(r.get('placeholder_events_removed',0) or 0) for r in valid),'limitations':['No ground-truth labels assigned.','Report verdicts are not ground truth.','Runtime API call sequences are not inferred.','Network indicators may include benign infrastructure or textual mentions.','Coverage counts do not prove correctness; spot-check source reports before training.']}
 (out/'quality_summary.json').write_text(json.dumps(q,indent=2,ensure_ascii=False),encoding='utf-8')
 (out/'README.txt').write_text('REM Parser v5\nRun: py parse_reports_v5.py Raw_Reports.zip -o REM_Parser_v5_Output\nDependency: py -m pip install beautifulsoup4\nOutputs: report_features.csv, behavior_events.csv, extraction_log.csv, parse_errors.csv, quality_summary.json.\nGround-truth labels remain blank; report verdicts are not ground truth. Review source evidence before training.\n',encoding='utf-8')
 print(json.dumps(q,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
