#!/usr/bin/env python3
"""REM unified preprocessing for MalBehavD CSV, Claude CSV, and CAPE/Zenbox/HABO HTML reports.
Only parses files; never executes samples. See README.md for label/leakage policy."""
import argparse, json, re, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

SHA=re.compile(r'(?i)\b[a-f0-9]{64}\b'); MD5=re.compile(r'(?i)\b[a-f0-9]{32}\b')
IP=re.compile(r'\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b')
MITRE=re.compile(r'\bT\d{4}(?:\.\d{3})?\b',re.I)
API=re.compile(r'\b(?:CreateFile[AW]?|WriteFile|ReadFile|RegSetValueEx[AW]?|RegCreateKeyEx[AW]?|VirtualAllocEx|WriteProcessMemory|CreateRemoteThread|WinExec|ShellExecute[AW]?|InternetOpen[AW]?|InternetConnect[AW]?|HttpSendRequest[AW]?|URLDownloadToFile[AW]?)\b')
DOMAIN=re.compile(r'(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|co|ru|cn|info|biz|xyz|site|online|dev|app|cloud|edu|gov|uk|in|de|jp|me|tv)\b')
TERMS=['process injection','remote thread','process hollowing','credential access','keylogging','ransomware','persistence','anti-vm','anti-debug','powershell','command execution','scheduled task','service creation','registry modification','file deletion','file encryption','network connection','dns query','http request','privilege escalation','defense evasion','exfiltration']

def txt(x): return re.sub(r'\s+',' ',str(x or '')).strip()
def norm_label(v):
    if pd.isna(v): return None
    s=str(v).strip().lower()
    return {'0':0,'benign':0,'goodware':0,'clean':0,'normal':0,'1':1,'malware':1,'malicious':1,'badware':1,'infected':1}.get(s)
def load_labels(path):
    if not path:return {}
    d=pd.read_csv(path); cm={str(c).lower():c for c in d.columns}
    hc=next((cm[x] for x in ('sha256','sha-256','hash') if x in cm),None); lc=next((cm[x] for x in ('label','ground_truth','class') if x in cm),None)
    if not hc or not lc: raise ValueError('Verified label file requires columns sha256,label')
    tmp={}
    for _,r in d.iterrows():
        h=str(r[hc]).strip().lower(); y=norm_label(r[lc])
        if SHA.fullmatch(h) and y is not None: tmp.setdefault(h,set()).add(y)
    return {h:(next(iter(v)) if len(v)==1 else None) for h,v in tmp.items()}

def parse_report(p):
    soup=BeautifulSoup(p.read_text(encoding='utf-8',errors='ignore'),'lxml'); text=txt(soup.get_text(' ',strip=True)); title=txt(soup.title.get_text(' ',strip=True)) if soup.title else ''
    intro=(title+' '+p.name+' '+text[:4000]).lower()
    sandbox='HABO' if ('habo' in intro or 'tencent' in intro) else 'CAPE' if 'cape' in intro else 'Zenbox' if 'zenbox' in intro else 'Jujubox' if 'jujubox' in intro else 'Unknown'
    h=next(iter(SHA.findall(text)), '')
    if not h and SHA.fullmatch(p.stem):h=p.stem.lower()
    ips=sorted(set(IP.findall(text))); domains=sorted(set(DOMAIN.findall(text))); mitre=sorted(set(x.upper() for x in MITRE.findall(text)))
    behaviors=sorted(t for t in TERMS if t in text.lower())
    name=''
    for el in soup.find_all(['td','th','dt','dd','span','div','p','h1','h2','h3','a']):
        s=txt(el.get_text(' ',strip=True))
        if len(s)<180 and re.search(r'\.(exe|dll|sys|scr|bat|cmd|ps1|msi|com|jar|js|vbs)\b',s,re.I):
            candidate=s.split()[0]
            if candidate.lower() not in {'report.exe'} and not candidate.lower().startswith('http'):name=candidate;break
    events=[]; seq=[]
    for el in soup.find_all(['tr','li','pre','code']):
        s=txt(el.get_text(' ',strip=True)); low=s.lower()
        if not s or len(s)>1200:continue
        kind=None
        if any(x in low for x in ['process created','process started','process terminated','process tree','command line']):kind='process'
        elif any(x in low for x in ['registry key','regsetvalue','regcreatekey','registry value']):kind='registry'
        elif any(x in low for x in ['file created','file written','file deleted','file dropped','file opened']):kind='file'
        elif any(x in low for x in ['dns query','http request','tcp connection','network connection','connectex','socket']):kind='network'
        elif API.search(s):kind='api_candidate'
        if kind:events.append({'sha256':h,'source_report':p.name,'sandbox':sandbox,'event_type':kind,'event_text':s})
        if kind=='api_candidate':seq.extend(API.findall(s))
    verdict=next((v for v in ['non malicious','non-malicious','malicious','suspicious','undetected'] if v in text[:12000].lower()),'')
    feat={'sha256':h,'sample_name':name,'sandbox':sandbox,'source_report':p.name,'report_title':title,'report_verdict_untrusted':verdict,'ip_count':len(ips),'domain_count':len(domains),'mitre_count':len(mitre),'behavior_term_count':len(behaviors),'network_ips_json':json.dumps(ips),'domains_json':json.dumps(domains),'mitre_ids_json':json.dumps(mitre),'behavior_terms_json':json.dumps(behaviors),'api_candidates_json':json.dumps(seq)}
    return feat,events

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--malbehav');ap.add_argument('--claude');ap.add_argument('--reports');ap.add_argument('--labels');ap.add_argument('--output',default='rem_unified_output');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True); labels=load_labels(a.labels); rows=[];events=[];audit=[]
    if a.malbehav:
        d=pd.read_csv(a.malbehav); low={str(c).strip().lower():c for c in d.columns}; hc=next((low[x] for x in ['sha256','sha-256','hash'] if x in low),None); lc=next((low[x] for x in ['label','labels','class','ground_truth'] if x in low),None)
        if lc is None:raise ValueError('MalBehavD CSV needs a label/labels column')
        sc=sorted([c for c in d.columns if re.fullmatch(r'\d+',str(c).strip())],key=lambda x:int(str(x)))
        for i,r in d.iterrows():
            h=str(r[hc]).lower().strip() if hc and pd.notna(r[hc]) else '';h=h if SHA.fullmatch(h) else ''
            seq=[str(r[c]).strip() for c in sc if pd.notna(r[c]) and str(r[c]).strip().lower() not in {'0','0.0','nan','none',''}]
            raw=str(r[lc]); rows.append({'sample_key':h or f'malbehav_row_{i}','sha256':h,'source':'MalBehavD-V1','sandbox':'','sample_name':'','label':labels.get(h) if h else np.nan,'label_status':'verified_external' if h and h in labels and labels[h] is not None else 'unverified_source_label','source_label_raw':raw,'api_sequence_json':json.dumps(seq),'api_sequence_length':len(seq),'numeric_features_json':'{}','ip_count':0,'domain_count':0,'mitre_count':0,'behavior_term_count':0,'network_ips_json':'[]','domains_json':'[]','mitre_ids_json':'[]','behavior_terms_json':'[]','report_count':0})
        audit.append({'source':'MalBehavD-V1','path':a.malbehav,'records':len(d),'note':'Original label mapping retained as unverified; use --labels for independently verified SHA labels.'})
    if a.claude:
        d=pd.read_csv(a.claude)
        if 'label' not in d.columns:raise ValueError('Claude CSV needs label column')
        for i,r in d.iterrows():
            h=str(r.get('sha256','')).lower().strip();h=h if SHA.fullmatch(h) else ''; sid=str(r.get('sample_id',i))
            nums={str(c):float(r[c]) for c in d.columns if c not in {'label','sample_id','family','sha256'} and pd.api.types.is_numeric_dtype(d[c]) and pd.notna(r[c])}
            rows.append({'sample_key':h or f'claude_{sid}','sha256':h,'source':'Claude-generated-unverified','sandbox':'','sample_name':'','label':labels.get(h) if h else np.nan,'label_status':'verified_external' if h and h in labels and labels[h] is not None else 'unverified_claude_label','source_label_raw':str(r['label']),'api_sequence_json':'[]','api_sequence_length':0,'numeric_features_json':json.dumps(nums),'ip_count':int(nums.get('network_dns_queries',0)),'domain_count':0,'mitre_count':0,'behavior_term_count':sum(int(v>0) for k,v in nums.items() if any(t in k.lower() for t in ['injection','ransomware','persistence','credential','anti_vm','anti_debug'])),'network_ips_json':'[]','domains_json':'[]','mitre_ids_json':'[]','behavior_terms_json':'[]','report_count':0})
        audit.append({'source':'Claude-generated-unverified','path':a.claude,'records':len(d),'note':'Claude labels are not used as ground truth unless joined to verified labels by SHA-256.'})
    if a.reports:
        paths=sorted(list(Path(a.reports).rglob('*.html'))+list(Path(a.reports).rglob('*.htm')))
        for p in paths:
            try:
                f,ev=parse_report(p);events.extend(ev);h=f['sha256'];rows.append({'sample_key':h or 'report_'+hashlib.sha256(str(p).encode()).hexdigest()[:16],'sha256':h,'source':'sandbox_report','sandbox':f['sandbox'],'sample_name':f['sample_name'],'label':labels.get(h) if h else np.nan,'label_status':'verified_external' if h and h in labels and labels[h] is not None else 'unlabeled_report','source_label_raw':'','api_sequence_json':f['api_candidates_json'],'api_sequence_length':len(json.loads(f['api_candidates_json'])),'numeric_features_json':'{}','ip_count':f['ip_count'],'domain_count':f['domain_count'],'mitre_count':f['mitre_count'],'behavior_term_count':f['behavior_term_count'],'network_ips_json':f['network_ips_json'],'domains_json':f['domains_json'],'mitre_ids_json':f['mitre_ids_json'],'behavior_terms_json':f['behavior_terms_json'],'report_count':1,'report_features_json':json.dumps(f),'report_verdict_untrusted':f['report_verdict_untrusted']})
            except Exception as e:audit.append({'source':'sandbox_report','path':str(p),'records':0,'note':'ERROR '+repr(e)})
        audit.append({'source':'sandbox_reports','path':a.reports,'records':len(paths),'note':'Heuristic parsing; validate output against original HTML.'})
    if not rows:raise ValueError('Provide at least one input option.')
    d=pd.DataFrame(rows);d['group_key']=d['sha256'].where(d['sha256'].fillna('').ne(''),d['sample_key'])
    # Detect conflicting verified labels per hash.
    for h,ix in d[d.sha256.fillna('').ne('')].groupby('sha256').groups.items():
        ys=set(int(x) for x in d.loc[ix,'label'].dropna())
        if len(ys)>1:d.loc[ix,'label']=np.nan;d.loc[ix,'label_status']='conflicting_verified_labels'
    combined=[]
    for key,g in d.groupby('group_key',sort=False,dropna=False):
        r=g.iloc[0].to_dict();r['sources_json']=json.dumps(sorted(set(g.source.astype(str))));r['sandboxes_json']=json.dumps(sorted(set(g.sandbox.dropna().astype(str))-{''}));r['report_count_total']=int(g.report_count.fillna(0).sum())
        ys=set(int(x) for x in g.label.dropna());r['label']=next(iter(ys)) if len(ys)==1 else np.nan
        if len(ys)>1:r['label_status']='conflicting_verified_labels'
        elif pd.isna(r['label']):r['label_status']='unverified_or_unlabeled'
        seq=[]
        for v in g.api_sequence_json.fillna('[]'):
            try:seq.extend(json.loads(v))
            except Exception:pass
        r['api_sequence_json']=json.dumps(seq);r['api_sequence_length']=len(seq)
        for c in ['ip_count','domain_count','mitre_count','behavior_term_count']:
            r[c]=int(pd.to_numeric(g[c],errors='coerce').fillna(0).max())
        combined.append(r)
    u=pd.DataFrame(combined);u.to_csv(out/'samples.csv',index=False)
    with open(out/'sequences.jsonl','w',encoding='utf-8') as f:
        for _,r in u.iterrows():f.write(json.dumps({'sample_key':r.sample_key,'sha256':r.sha256,'label':None if pd.isna(r.label) else int(r.label),'label_status':r.label_status,'sequence':json.loads(r.api_sequence_json or '[]')})+'\n')
    pd.DataFrame(events,columns=['sha256','source_report','sandbox','event_type','event_text']).to_csv(out/'report_events.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'source_audit.csv',index=False)
    labeled=u[u.label.notna()].copy(); splits=[]
    if len(labeled)>=10 and labeled.label.nunique()==2:
        from sklearn.model_selection import GroupShuffleSplit
        gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);tr,te=next(gss.split(labeled,labeled.label,groups=labeled.group_key))
        train=labeled.iloc[tr];test=labeled.iloc[te];val_keys=set();train_keys=set(train.group_key);test_keys=set(test.group_key)
        if len(train)>=10:
            gss2=GroupShuffleSplit(n_splits=1,test_size=.125,random_state=43);tri,vi=next(gss2.split(train,train.label,groups=train.group_key));train_keys=set(train.iloc[tri].group_key);val_keys=set(train.iloc[vi].group_key)
        for _,r in labeled.iterrows():splits.append({'sample_key':r.sample_key,'sha256':r.sha256,'group_key':r.group_key,'label':int(r.label),'split':'train' if r.group_key in train_keys else 'validation' if r.group_key in val_keys else 'test'})
    pd.DataFrame(splits,columns=['sample_key','sha256','group_key','label','split']).to_csv(out/'split_manifest.csv',index=False)
    schema={'trusted_labels':'Only --labels SHA-256 joins are trusted ground truth.','malbehav':'Raw dataset labels are retained but not assumed to map to benign/malware.','claude':'Claude labels are unverified and not used as ground truth by default.','sandbox':'Report verdicts are metadata only, never ground truth.','leakage':'Rows with the same valid SHA-256 are collapsed before split; sample_id/family are not output as model features.','parser':'HTML extraction is heuristic; inspect events and compare with source reports. Static imports may be mistaken for event candidates and must be validated.'}
    (out/'schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
    print(f'Created {out.resolve()} with {len(u):,} unique sample/group records; {int(u.label.notna().sum()):,} have externally verified labels. Review source_audit.csv before training.')
if __name__=='__main__':main()
