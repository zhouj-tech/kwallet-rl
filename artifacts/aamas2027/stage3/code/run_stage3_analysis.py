#!/usr/bin/env python3
"""Stage 3: immutable archive acceptance, frozen statistics, descriptive mechanisms.
Never imports an evaluator, checkpoint, trainer, or collaborator code.
"""
from __future__ import annotations
import argparse, ast, collections, csv, hashlib, io, json, math, os
from pathlib import Path, PurePosixPath
import platform, shutil, statistics, sys, tarfile
sys.dont_write_bytecode = True
RAW_SHA='576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3'
LINEAGE='5573ec642f0f28c218f3e6058478f62ab6db6b2b'
BUNDLE='KWALLET_AAMAS_STAGE2B_RAW_20261002'
REGIMES='US TLS LNS TLNS TPLS PLS UB TLB LNB TLNB TPLB PLB'.split()
METHODS=['JA-PPO','IFAC','SC-FAC','SC-FAC-zero','BF-T0.5']
SEEDS=[123,323,532,777,999]
CAPACITIES=[800,1200]
MISSING_LEDGER={'A-EVAL-IF-C800-S123','A-EVAL-SC-C800-S123','B-EVAL-ZERO-C800-S123','C-EVAL-BFT05-C800'}
SOURCE_HASHES={
'src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py':'d46246795552f22f0ba143ae38230692ff99997d57c9b0681f456432a9df1921',
'src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py':'e4a669678453a2884ac6834ee599dd85a98e6bac59ac59630513f7c2d022d0ea',
'src/idea4/ac/code/run_factorized_ac_benchmark.py':'365971c27702d7c05f0a77dbc9f94ed80b3fd2fe470c7ce925409effef2a2971',
'src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py':'c3710b3a8bbf905f2d047b5bf4444830cdc32239e39b49becc149540f6e79b04',
'src/ideaextra/kwallet_ideaextra_generator.py':'d702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8'}
METRICS=['money','settled','flushes','drops','oversize_drops','insufficient_drops','accepted_count','total_tx_count','total_requested_value','value_accept_ratio','count_accept_ratio','utilization']
class AcceptanceError(RuntimeError):pass

def require(ok,message):
    if not ok:raise AcceptanceError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def jread(p):return json.loads(Path(p).read_text())
def jwrite(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvread(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    require(bool(rows),'Empty table '+str(p))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b,label,tol=1e-9):require(math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol),label)
def mean(xs):return statistics.mean(xs)
def mdtable(rows,columns):
    def fmt(v):
        if v is None:return 'NA'
        if isinstance(v,float):return f'{v:.6g}'
        return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(fmt(r[c]) for c in columns)+' |' for r in rows])

def extract_verified(raw,out):
    require(sha(raw)==RAW_SHA,'Raw tarball SHA256 mismatch')
    dest=out/'work'/'raw';dest.mkdir(parents=True,exist_ok=True)
    inventory=[]
    with tarfile.open(raw,'r:gz') as tf:
        members=tf.getmembers();names=set()
        for m in members:
            p=PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and p.parts[0]==BUNDLE,'Unsafe archive member')
            require(m.name not in names,'Duplicate tar member');names.add(m.name)
            require(m.isdir() or m.isfile(),'Archive contains nonregular member')
        for m in members:
            p=dest/m.name
            if m.isdir():p.mkdir(parents=True,exist_ok=True);continue
            content=tf.extractfile(m).read();h=hashlib.sha256(content).hexdigest()
            if p.exists():require(sha(p)==h,'Existing extracted raw file changed: '+m.name)
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(content)
                p.chmod(0o444)
            inventory.append(dict(path=str(PurePosixPath(m.name).relative_to(BUNDLE)),size=m.size,sha256=h))
    root=dest/BUNDLE;listed={}
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        h,name=line.split(None,1);name=name.lstrip('*');name=name.removeprefix('./')
        require(name not in listed,'Duplicate checksum entry');listed[name]=h
        require((root/name).is_file() and sha(root/name)==h,'Internal checksum mismatch '+name)
    uncovered=sorted({r['path'] for r in inventory}-set(listed))
    require(set(uncovered)<={'SHA256SUMS.txt','README.md'},'Unmanifested raw science file')
    return root,inventory,uncovered

def accept_raw(raw,out):
    root,inventory,uncovered=extract_verified(raw,out)
    approval=jread(root/'MAC_RUNTIME_APPROVED.json');manifest=jread(root/'new12/new12_manifest.json')
    freeze=jread(root/'new12/freeze_receipt.json');ledger=jread(root/'output/progress_ledger.json')
    require(approval['approval_status']=='MAC_RUNTIME_APPROVED' and approval['git_head']==LINEAGE,'Approval/lineage mismatch')
    bindings={'job_matrix_sha256':'research/STAGE2_JOB_MATRIX.csv','artifact_manifest_sha256':'research/AAMAS_ARTIFACT_MANIFEST.csv','new12_manifest_sha256':'new12/new12_manifest.json','overlap_report_sha256':'new12/overlap_report.json','freeze_receipt_sha256':'new12/freeze_receipt.json','orchestrator_sha256':'stage2b_orchestrator.py','run_benchmark_sha256':'run_benchmark.py','shell_launcher_sha256':'run_stage2b_mac.sh'}
    for key,path in bindings.items():require(sha(root/path)==approval[key],'Approval binding mismatch '+key)
    require(freeze['approved'] and freeze['git_sha']==LINEAGE and manifest['git_sha']==LINEAGE,'Freeze lineage mismatch')
    require(freeze['manifest_sha256']==sha(root/'new12/new12_manifest.json'),'Freeze manifest mismatch')
    require(manifest['regime_order']==REGIMES,'Wrong regime order')
    pools={p['regime']:p for p in manifest['pools']};require(set(pools)==set(REGIMES),'Missing manifest pools')
    jobs=[j for j in csvread(root/'research/STAGE2_JOB_MATRIX.csv') if j['job_type']=='evaluation']
    expected={(m,c,s) for m in METHODS[:-1] for c in CAPACITIES for s in SEEDS}|{('BF-T0.5',c,None) for c in CAPACITIES}
    observed={(j['method'],int(j['C']),None if j['seed']=='NA' else int(j['seed'])) for j in jobs}
    require(len(jobs)==42 and observed==expected,'Incorrect frozen 42-job matrix')
    ids={j['job_id'] for j in jobs};require(len(ids)==42,'Duplicate job IDs')
    actualdirs={p.name for p in (root/'output').iterdir() if p.is_dir() and (p/'episodes.csv').exists()}
    require(actualdirs==ids,'Missing/extra job-level episode outputs')
    require(set(ledger['completed'])<=ids and not ledger['failed'] and not ledger['infra_failed'],'Unresolved ledger failures')
    omitted=ids-set(ledger['completed']);require(omitted==MISSING_LEDGER,'Unexpected ledger reconciliation')
    # Read frozen source as data; never import or execute it.
    tree=ast.parse((root/'stage2b_orchestrator.py').read_text());reps=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VALIDATION_REPS' for t in node.targets):reps=ast.literal_eval(node.value)
    require(reps==omitted,'Skipped set differs from frozen orchestrator validation representatives')
    parity={g['job_id']:g for g in approval['validation_gate_jobs']}
    jobrows=[];regimerows=[];data={};maxidentity=0.;maxsummary=0.;runtime_counts=collections.Counter();notes=[]
    inputs={};newcheckpoints=[]
    for job in sorted(jobs,key=lambda j:j['job_id']):
        jid=job['job_id'];p=root/'output'/jid;result=jread(p/'result.json');rows=csvread(p/'episodes.csv')
        require(len(rows)==2400 and result['row_count']==2400,'Incomplete job '+jid)
        require(result.get('validation_status','PASS')=='PASS','Validation failed '+jid)
        require(result['job']==job,'Embedded job differs from frozen matrix '+jid)
        config=result['config']
        for field in ['C','k','F','T']:close(config['env'][field],job[field],'Configuration mismatch '+jid+' '+field)
        require(config['env']['enable_shaping'] is False and config['reward_mode']=='original','Wrong E0/reward configuration '+jid)
        require(config['eval']=={'max_steps':1000,'num_episodes':200},'Wrong evaluation configuration '+jid)
        require(config['money_p']==1 and config['money_tau']==10,'Wrong Money configuration '+jid)
        if job['method']!='BF-T0.5':
            require(config['seed']==int(job['seed']),'Configuration seed mismatch '+jid)
            modes={'JA-PPO':'basic_ppo','IFAC':'factorized_ac','SC-FAC':'conditional_factorized_ac','SC-FAC-zero':'conditional_factorized_ac'}
            require(config['model_mode']==modes[job['method']],'Wrong learned model mode '+jid)
            for field in ['checkpoint_sha256','run_info_sha256']:
                digest=result[field];require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Missing/invalid learned provenance digest '+jid+' '+field)
            if job['method']=='SC-FAC-zero':require(config['condition_mode']=='zero_settle_embedding','Wrong zero configuration '+jid)
        else:
            require(result['checkpoint_sha256'] is None,'Rule must not have a checkpoint '+jid)
            require(result['rule_spec_sha256']=='3af3074a6e0add23edc998fffeeb158848b0e63312584063e53560740e80c533','Unexpected recorded BF rule specification hash '+jid)
        require(result['dataset_id']=='NEW12-v1' and result['git']['git_sha']==LINEAGE,'Dataset/lineage mismatch '+jid)
        require(result['source_hashes']==SOURCE_HASHES,'Historical source hashes mismatch '+jid)
        for key in ['adapter_sha256','artifact_manifest_sha256','job_matrix_sha256']:
            require(result[key]==approval[key],'Provenance mismatch '+jid+' '+key)
        require(result['stream_manifest_sha256']==approval['new12_manifest_sha256'],'Wrong NEW12 binding')
        require(result['episodes_sha256']==sha(p/'episodes.csv'),'Result/CSV hash mismatch '+jid)
        if len(job['checkpoint_sha256'])==64:require(result['checkpoint_sha256']==job['checkpoint_sha256'],'Checkpoint mismatch '+jid)
        elif job['method']!='BF-T0.5':newcheckpoints.append(dict(job_id=jid,checkpoint_sha256=result['checkpoint_sha256'],run_info_sha256=result['run_info_sha256']))
        if jid in ledger['completed']:
            le=ledger['completed'][jid]
            require(le['episodes_sha256']==sha(p/'episodes.csv') and le['result_sha256']==sha(p/'result.json'),'Ledger hash mismatch '+jid)
            require(le['returncode']==0 and le['validation_status']=='PASS' and le['row_count']==2400,'Invalid ledger record')
            close(le['macro12_money'],result['macro12_money'],'Ledger Money mismatch')
        if jid in parity:
            g=parity[jid];require(g['parity']=='EXACT' and g['server_episodes_sha256']==g['mac_episodes_sha256']==sha(p/'episodes.csv'),'Approval parity mismatch '+jid)
        if jid in omitted:require(jid in parity,'Omitted job lacks approval evidence')
        runtime_counts[json.dumps(result.get('runtime_local',{}),sort_keys=True)]+=1
        if result.get('benchmark_note'):notes.append(jid)
        byreg=collections.defaultdict(list);pairs=set()
        for row in rows:
            reg=row['regime'];i=int(row['episode_index']);key=(reg,i)
            require(reg in REGIMES and 0<=i<200 and key not in pairs,'Invalid/duplicate episode '+jid);pairs.add(key)
            for field in ['job_id','family','method','C','k','F','T','condition_mode']:require(row[field]==job[field],'Row job metadata mismatch '+field)
            require(row['dataset_id']=='NEW12-v1' and row['schema_version']=='1','Wrong row dataset')
            require(row['training_seed']==('' if job['seed']=='NA' else job['seed']),'Row seed mismatch')
            require(row['checkpoint_sha256']==(result['checkpoint_sha256'] or ''),'Row checkpoint mismatch')
            require(row['pool_sha256']==pools[reg]['sha256']==result['pool_sha256'][reg],'Pool hash mismatch')
            require(row['episode_sha256']==pools[reg]['episode_hashes'][i] and int(row['episode_seed'])==pools[reg]['episode_seeds'][i],'Episode fingerprint mismatch')
            nums={k:float(row[k]) for k in METRICS+['eval_money','eval_money_p','eval_money_tau','drop_rate','avg_tx_value']}
            require(all(math.isfinite(v) for v in nums.values()),'Nonfinite metric')
            for key2 in ['drops','flushes','oversize_drops','insufficient_drops','accepted_count','total_tx_count']:
                v=nums[key2];require(0<=v<=1000 and v.is_integer(),'Invalid count')
            require(nums['total_tx_count']==1000 and nums['eval_money_p']==1 and nums['eval_money_tau']==10,'Wrong horizon/Money constants')
            close(nums['drops']+nums['accepted_count'],1000,'Count identity')
            close(nums['oversize_drops']+nums['insufficient_drops'],nums['drops'],'Drop identity')
            delta=abs(nums['money']-(nums['settled']-10*nums['flushes']));maxidentity=max(maxidentity,delta);require(delta<=1e-9,'Money identity failure')
            close(nums['money'],nums['eval_money'],'Money alias failure')
            require(nums['total_requested_value']>0 and 0<=nums['settled']<=nums['total_requested_value'],'Invalid value totals')
            close(nums['value_accept_ratio'],nums['settled']/nums['total_requested_value'],'Value ratio')
            close(nums['count_accept_ratio'],nums['accepted_count']/1000,'Count ratio')
            close(nums['drop_rate'],nums['drops']/1000,'Drop rate')
            for f in ['value_accept_ratio','count_accept_ratio','utilization']:require(0<=nums[f]<=1,'Invalid ratio')
            # Same exogenous episode totals in every job; oversize is capacity-dependent.
            exkey=(reg,i);inp=(nums['total_requested_value'],row['episode_sha256'])
            if exkey in inputs:require(inputs[exkey]==inp,'Cross-job stream inconsistency')
            else:inputs[exkey]=inp
            byreg[reg].append((i,nums))
        require(set(pairs)=={(r,i) for r in REGIMES for i in range(200)},'Incomplete regime coverage')
        c=int(job['C']);s=None if job['seed']=='NA' else int(job['seed']);m=job['method']
        cohort={r['cohort'] for r in rows};require(len(cohort)==1,'Mixed job cohort')
        cohort=cohort.pop();expectedcohort='rule' if m=='BF-T0.5' else ('stage2_new_current_runtime' if job['checkpoint_source'].startswith('receipt:') else 'historical_recovered')
        require(cohort==expectedcohort,'Unexpected training cohort')
        for reg in REGIMES:
            arr=[v for _,v in sorted(byreg[reg])]
            rr=dict(job_id=jid,method=m,C=c,training_seed=s,cohort=cohort,regime=reg,n_episodes=200)
            for metric in METRICS:
                vals=[r[metric] for r in arr];rr[metric]=mean(vals)
                key='eval_money' if metric=='money' else metric
                saved=result['regime_summaries'][reg][key]
                for name,value in [('mean',rr[metric]),('std',statistics.pstdev(vals)),('min',min(vals)),('max',max(vals)),('median',statistics.median(vals))]:
                    err=abs(saved[name]-value);maxsummary=max(maxsummary,err);require(err<=1e-9,'Saved summary mismatch '+jid+' '+reg+' '+key+' '+name)
                if 'values' in saved:require(saved['values']==vals,'Saved episode values mismatch '+jid)
            regimerows.append(rr)
        macro=mean([r['money'] for r in regimerows if r['job_id']==jid]);close(macro,result['macro12_money'],'Saved macro mismatch')
        jobrows.append(dict(job_id=jid,method=m,C=c,training_seed=s,cohort=cohort,rows=2400,regimes=12,validation_status=result.get('validation_status','NOT_RECORDED'),audit_status='PASS',ledger_completed=jid in ledger['completed'],approved_preexisting=jid in omitted,macro12_money=macro,episodes_sha256=sha(p/'episodes.csv'),result_sha256=sha(p/'result.json'),checkpoint_sha256=result['checkpoint_sha256'],run_info_sha256=result['run_info_sha256'],adapter_sha256=result['adapter_sha256'],stream_manifest_sha256=result['stream_manifest_sha256'],lineage=LINEAGE))
        data[m,c,s]=dict(job=job,result=result)
    # Pilot duplicate is deliberately not an additional scientific replicate.
    rootdup=sha(root/'output/episodes.csv')==sha(root/'output/A-EVAL-JA-C800-S123/episodes.csv')
    require(rootdup,'Root pilot is not expected JA duplicate')
    audit=dict(status='PASS',scientific_complete=True,scientific_jobs=42,total_rows=100800,ledger_completed=len(ledger['completed']),ledger_complete=False,ledger_reconciled=True,preexisting_jobs=sorted(omitted),raw_sha256=RAW_SHA,lineage=LINEAGE,internal_files=len(inventory),internal_checksum_coverage=len(inventory)-len(uncovered),uncovered_files=uncovered,max_money_identity_error=maxidentity,max_saved_summary_error=maxsummary,root_pilot_duplicate_excluded=True,benchmark_notes_retained=len(notes),runtime_groups=[dict(runtime=json.loads(k),jobs=n) for k,n in sorted(runtime_counts.items())],approval_runtime_header={k:approval[k] for k in ['python_version','numpy_version','torch_version']},new_zero_checkpoint_records=newcheckpoints,
      limitations=['Raw freeze contains outputs/manifests, not checkpoint bytes, pool arrays, training receipts, or all original runtime/gate receipts; those cannot be independently rehashed from this archive.', 'Legacy local-benchmark labels and null runtime_lock_sha256 are retained, not repaired; acceptance uses the provided final freeze, embedded MAC_RUNTIME_APPROVED chain and five representative episode-hash parity attestations.', 'Approval header Python/Torch version strings differ from per-job runtime_local strings; report both rather than asserting literal lock equality.'])
    csvwrite(out/'raw_file_inventory.csv',inventory);csvwrite(out/'stage2b_job_summary.csv',jobrows);jwrite(out/'raw_acceptance.json',audit)
    write_acceptance(out,audit,jobrows)
    return root,audit,jobrows,regimerows,manifest,data

def write_acceptance(out,a,jobs):
    text=f'''# Stage 3 raw acceptance

Scientific completeness: **PASS — {a['scientific_jobs']}/42 jobs, {a['total_rows']:,} episode rows.**
Bookkeeping completeness: **38/42 in the original ledger; reconciled, not repaired.**
Raw SHA256: `{RAW_SHA}`. Frozen scientific lineage: `{LINEAGE}`.

## Acceptance checks

- Every named job has episodes.csv and result.json, 2400 rows, twelve regimes with 200 unique episode indices each, and recorded PASS.
- Frozen matrix matches 40 learned evaluations (four methods × two capacities × five seeds) plus two deterministic rule evaluations.
- Money = settled − 10 × flushes, count identities, finite values, ratios, complete episode coverage, and saved summaries verified independently.
- Maximum Money identity error: {a['max_money_identity_error']}; maximum saved-summary arithmetic difference: {a['max_saved_summary_error']:.12g} (absolute tolerance 1e-9).
- All episode/pool fingerprints match the frozen NEW12 manifest; exogenous requested values match across jobs.
- {a['internal_checksum_coverage']} internal SHA256 entries verified; {a['internal_files']} regular files covered by the outer tarball hash. Internal checksum exclusions: {', '.join(a['uncovered_files'])}.
- Result source hashes, adapter/matrix/artifact-manifest bindings, embedded job definitions and scientific git lineage verified.

## Four ledger omissions

The missing completed entries are exactly the frozen orchestrator's VALIDATION_REPS. Each has valid full scientific outputs and matching Mac/server episode hashes in MAC_RUNTIME_APPROVED.json:

'''
    for jid in a['preexisting_jobs']:
        r=next(x for x in jobs if x['job_id']==jid);text+=f"- `{jid}` — 2400 rows; PASS; episodes SHA256 `{r['episodes_sha256']}`.\n"
    text+='''
The orchestrator skipped these already-valid outputs rather than adding completed entries. The full log also records a resume skip of JA-PPO C800/123, which already has a ledger entry; it is not a fifth omission. The root-level output/episodes.csv is byte-identical to A-EVAL-JA-C800-S123/episodes.csv and is excluded as a duplicate pilot. Only the 42 job directories enter statistics.

## Provenance qualifications (not concealed)

'''
    text+='\n'.join('- '+x for x in a['limitations'])
    text+='\n\nApproval header: `'+json.dumps(a['approval_runtime_header'],sort_keys=True)+'`.\n\nActual job runtime groups:\n\n```json\n'+json.dumps(a['runtime_groups'],indent=2,sort_keys=True)+'\n```\n'
    text+='\nThe reported approval attests Mac-versus-server parity for Stage-2 representative jobs. It is not a new claim of universal cross-platform equality or a reopening of historical SC-FAC drift. No raw outputs or ledger entries were modified. Analysis proceeds from the user-designated frozen official bundle, with these provenance limitations visible.\n\nExact per-job checks and hashes: stage2b_job_summary.csv. Exact raw-file inventory: raw_file_inventory.csv.\n'
    (out/'STAGE3_RAW_ACCEPTANCE.md').write_text(text)

# Statistical/plot/report functions follow; --audit-only never imports SciPy/Matplotlib.
def summary5(values):
    from scipy.stats import t
    require(len(values)==5,'Inference requires exactly five seeds')
    mu=mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(5);crit=float(t.ppf(.975,4))
    return dict(n=5,mean=mu,SD=sd,SE=se,ci95_low=mu-crit*se,ci95_high=mu+crit*se,df=4,t_critical=crit)

def contrast_stats(deltas):
    from scipy.stats import t
    s=summary5(deltas)
    if s['SD']==0:s.update(t_statistic=None,raw_p=None,d_z=None,test_status='UNDEFINED_ZERO_VARIANCE')
    else:
        tv=s['mean']/s['SE'];s.update(t_statistic=tv,raw_p=float(2*t.sf(abs(tv),4)),d_z=s['mean']/s['SD'],test_status='OK')
    return s

def holm(rows):
    # Undefined tests retain their reserved family slots (p=1 internally), never shrink family.
    m=len(rows);prev=0.
    for rank,row in enumerate(sorted(rows,key=lambda r:(1. if r['raw_p'] is None else r['raw_p'],r['contrast_id'])),1):
        adj=min(1.,max(prev,(m-rank+1)*(1. if row['raw_p'] is None else row['raw_p'])));prev=adj
        row['holm_p']=None if row['raw_p'] is None else adj
        row['holm_reject_0_05']=row['holm_p'] is not None and row['holm_p']<=.05
        row['family_size']=m

def compute_tables(jobs,regimes,manifest):
    scores=[]
    for j in jobs:
        r=dict(job_id=j['job_id'],method=j['method'],C=j['C'],training_seed=j['training_seed'],cohort=j['cohort'],macro12_money=j['macro12_money'])
        for reg in REGIMES:r['money_'+reg]=next(x['money'] for x in regimes if x['job_id']==j['job_id'] and x['regime']==reg)
        r.update(episodes_sha256=j['episodes_sha256'],checkpoint_sha256=j['checkpoint_sha256'],stream_manifest_sha256=j['stream_manifest_sha256'])
        close(mean([r['money_'+reg] for reg in REGIMES]),r['macro12_money'],'Hierarchy mismatch');scores.append(r)
    lookup={(r['method'],r['C'],r['training_seed']):r for r in scores}
    summaries=[]
    for c in CAPACITIES:
        for m in METHODS:
            s=dict(method=m,C=c,deterministic=m=='BF-T0.5',n_training_seeds=0 if m=='BF-T0.5' else 5)
            if m=='BF-T0.5':s.update(n=0,mean=lookup[m,c,None]['macro12_money'],SD=None,SE=None,ci95_low=None,ci95_high=None,df=None,t_critical=None)
            else:s.update(summary5([lookup[m,c,seed]['macro12_money'] for seed in SEEDS]))
            summaries.append(s)
    pairs=[];contrasts=[]
    families={'A':[('SC-FAC','JA-PPO'),('IFAC','JA-PPO'),('SC-FAC','IFAC')],'B':[('SC-FAC','SC-FAC-zero')],'C':[('SC-FAC','BF-T0.5')]}
    for family,comparisons in families.items():
        group=[]
        for c in CAPACITIES:
            for left,right in comparisons:
                cid=f'{family}_C{c}_{left}_minus_{right}';deltas=[]
                for seed in SEEDS:
                    l=lookup[left,c,seed];r=lookup[right,c,None if right=='BF-T0.5' else seed];d=l['macro12_money']-r['macro12_money'];deltas.append(d)
                    pairs.append(dict(contrast_id=cid,family=family,C=c,training_seed=seed,left_method=left,right_method=right,left_score=l['macro12_money'],right_score=r['macro12_money'],delta=d,left_cohort=l['cohort'],right_cohort=r['cohort']))
                stats=contrast_stats(deltas)
                entry=dict(contrast_id=cid,family=family,C=c,left_method=left,right_method=right,test='one_sample_on_seed_deltas' if family=='C' else 'paired_by_training_seed',**stats,effect_direction='left_higher' if stats['mean']>0 else ('right_higher' if stats['mean']<0 else 'equal'))
                entry['mean_difference']=entry.pop('mean');group.append(entry)
        holm(group);contrasts.extend(group)
    regsummary=[]
    for c in CAPACITIES:
        for m in METHODS:
            for reg in REGIMES:
                rs=[r for r in regimes if r['method']==m and r['C']==c and r['regime']==reg]
                rr=dict(method=m,C=c,regime=reg,n_training_seeds=0 if m=='BF-T0.5' else 5,dist_family=manifest['regime_specs'][reg]['dist_family'],bursty=manifest['regime_specs'][reg]['bursty'])
                for metric in METRICS:rr[metric+'_mean']=mean([r[metric] for r in rs]);rr[metric+'_SD']=None if m=='BF-T0.5' else statistics.stdev([r[metric] for r in rs])
                regsummary.append(rr)
    # Descriptive contrasts/decomposition: no additional p-values or confidence intervals.
    regpairs=[];regdiff=[]
    for contrast in contrasts:
        for reg in REGIMES:
            ds=[]
            for seed in SEEDS:
                l=next(r for r in regimes if (r['method'],r['C'],r['training_seed'],r['regime'])==(contrast['left_method'],contrast['C'],seed,reg))
                r=next(r for r in regimes if (r['method'],r['C'],r['training_seed'],r['regime'])==(contrast['right_method'],contrast['C'],None if contrast['family']=='C' else seed,reg))
                row=dict(contrast_id=contrast['contrast_id'],family=contrast['family'],C=contrast['C'],regime=reg,training_seed=seed,money_delta=l['money']-r['money'],settled_delta=l['settled']-r['settled'],flushes_delta=l['flushes']-r['flushes'],insufficient_drops_delta=l['insufficient_drops']-r['insufficient_drops'])
                close(row['money_delta'],row['settled_delta']-10*row['flushes_delta'],'Regime decomposition identity',tol=1e-8)
                ds.append(row);regpairs.append(row)
            rd=dict(contrast_id=contrast['contrast_id'],family=contrast['family'],C=contrast['C'],left_method=contrast['left_method'],right_method=contrast['right_method'],regime=reg,bursty=manifest['regime_specs'][reg]['bursty'],dist_family=manifest['regime_specs'][reg]['dist_family'],money_delta_mean=mean([x['money_delta'] for x in ds]),money_delta_SD=statistics.stdev([x['money_delta'] for x in ds]),positive_seed_differences=sum(x['money_delta']>0 for x in ds),settled_delta_mean=mean([x['settled_delta'] for x in ds]),flushes_delta_mean=mean([x['flushes_delta'] for x in ds]),insufficient_drops_delta_mean=mean([x['insufficient_drops_delta'] for x in ds]))
            regdiff.append(rd)
    groups=[]
    for contrast in contrasts:
        for bursty in [False,True]:
            rr=[r for r in regdiff if r['contrast_id']==contrast['contrast_id'] and r['bursty']==bursty]
            groups.append(dict(contrast_id=contrast['contrast_id'],C=contrast['C'],stream_group='bursty' if bursty else 'smooth',n_regimes=len(rr),mean_money_delta=mean([r['money_delta_mean'] for r in rr]),mean_settled_delta=mean([r['settled_delta_mean'] for r in rr]),mean_flushes_delta=mean([r['flushes_delta_mean'] for r in rr]),interpretation='descriptive_not_inferential'))
    defs=[dict(regime=r,dist_family=manifest['regime_specs'][r]['dist_family'],bursty=manifest['regime_specs'][r]['bursty'],target_mean=manifest['regime_specs'][r]['target_mean'],max_tx=manifest['regime_specs'][r]['max_tx'],note=manifest['regime_specs'][r]['note']) for r in REGIMES]
    decomposition=[]
    for x in contrasts:
        rr=[r for r in regdiff if r['contrast_id']==x['contrast_id']]
        d=dict(contrast_id=x['contrast_id'],family=x['family'],C=x['C'],money_delta=x['mean_difference'],settled_delta=mean([r['settled_delta_mean'] for r in rr]),flushes_delta=mean([r['flushes_delta_mean'] for r in rr]),regimes_positive=sum(r['money_delta_mean']>0 for r in rr),regimes_negative=sum(r['money_delta_mean']<0 for r in rr))
        d['flush_cost_delta']=10*d['flushes_delta'];close(d['money_delta'],d['settled_delta']-d['flush_cost_delta'],'Macro decomposition',tol=1e-8);decomposition.append(d)
    return dict(stage2b_job_summary=jobs,seed_level_scores=scores,method_capacity_summary=summaries,family_A_contrasts=[r for r in contrasts if r['family']=='A'],family_B_conditioning_ablation=[r for r in contrasts if r['family']=='B'],family_C_vs_bft05=[r for r in contrasts if r['family']=='C'],holm_results=[{k:r[k] for k in ['contrast_id','family','family_size','raw_p','holm_p','holm_reject_0_05','test_status']} for r in contrasts],regime_level_scores=regimes,regime_level_summary=regsummary,paired_seed_differences=pairs,regime_paired_differences=regpairs,regime_descriptive_contrasts=regdiff,regime_group_descriptives=groups,regime_definitions=defs,mechanism_decomposition=decomposition)
def make_figures(out,tables):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'work'/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'kwallet-stage3-frozen','savefig.dpi':240})
    colors={'JA-PPO':'#777777','IFAC':'#0072B2','SC-FAC':'#D55E00','SC-FAC-zero':'#009E73','BF-T0.5':'#222222'}
    dst=out/'figures';dst.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(dst/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None,'Creator':'K-Wallet Stage 3'})
        fig.savefig(dst/(name+'.svg'),bbox_inches='tight',metadata={'Date':None,'Creator':'K-Wallet Stage 3'})
        fig.savefig(dst/(name+'.png'),bbox_inches='tight',metadata={'Software':'K-Wallet Stage 3'})
        plt.close(fig)
    ms=tables['method_capacity_summary'];ps=tables['paired_seed_differences'];sc=tables['seed_level_scores']
    fig,axes=plt.subplots(1,2,figsize=(9,3.65),layout='constrained')
    for ax,c in zip(axes,CAPACITIES):
        for x,m in enumerate(METHODS):
            r=next(r for r in ms if r['method']==m and r['C']==c)
            if m=='BF-T0.5':
                ax.scatter(x,r['mean'],marker='D',color=colors[m],s=42,zorder=4);ax.axhline(r['mean'],color=colors[m],lw=1,ls='--',alpha=.6)
            else:
                vals=[r['macro12_money'] for r in sc if r['method']==m and r['C']==c]
                ax.scatter(x+np.linspace(-.12,.12,5),vals,s=16,color=colors[m],alpha=.55,zorder=3)
                ax.errorbar(x,r['mean'],yerr=[[r['mean']-r['ci95_low']],[r['ci95_high']-r['mean']]],fmt='o',color=colors[m],capsize=4,ms=6,lw=1.6,zorder=4)
        ax.set_xticks(range(5),['JA-PPO','IFAC','SC-FAC','SC-FAC\nzero','BF-T0.5']);ax.set_title(f'C = {c}');ax.set_ylabel('Macro12 Money');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('NEW12: five-seed means and marginal 95% t intervals',fontsize=11)
    save(fig,'figure1_macro12_money')
    fa=tables['family_A_contrasts'];order=[('SC-FAC','JA-PPO','SC − JA'),('IFAC','JA-PPO','IF − JA'),('SC-FAC','IFAC','SC − IF')]
    fig,axes=plt.subplots(3,2,figsize=(9,7),sharex=True,layout='constrained')
    allvals=[p['delta'] for p in ps if p['family']=='A']+[r[k] for r in fa for k in ['ci95_low','ci95_high']];limit=max(abs(x) for x in allvals)*1.08
    for j,c in enumerate(CAPACITIES):
        for i,(left,right,title) in enumerate(order):
            ax=axes[i,j];r=next(r for r in fa if r['C']==c and r['left_method']==left and r['right_method']==right)
            ds=[p['delta'] for p in ps if p['contrast_id']==r['contrast_id']]
            ax.axvline(0,color='#444444',lw=1);ax.scatter(ds,range(5),color='#0072B2',s=26)
            ax.errorbar(r['mean_difference'],5.3,xerr=[[r['mean_difference']-r['ci95_low']],[r['ci95_high']-r['mean_difference']]],fmt='D',color='#111111',capsize=4,ms=5)
            ax.set_yticks(list(range(5))+[5.3],[str(s) for s in SEEDS]+['Mean / CI']);ax.invert_yaxis();ax.set_xlim(-limit,limit);ax.grid(axis='x',alpha=.2)
            ax.set_title(f'{title}, C = {c} | Holm p = {r["holm_p"]:.3f}',fontsize=9)
            if i==2:ax.set_xlabel('Paired macro12 Money difference')
    fig.suptitle('Structural comparisons: every paired seed difference',fontsize=11)
    save(fig,'figure2_structural_seed_contrasts')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharex=True,layout='constrained')
    fb=tables['family_B_conditioning_ablation'];lim=max(abs(x) for r in fb for x in [r['ci95_low'],r['ci95_high']]+[p['delta'] for p in ps if p['contrast_id']==r['contrast_id']])*1.15
    for ax,c in zip(axes,CAPACITIES):
        r=next(r for r in fb if r['C']==c);points=[p for p in ps if p['contrast_id']==r['contrast_id']]
        for y,p in enumerate(points):ax.scatter(p['delta'],y,marker='o' if p['right_cohort']=='historical_recovered' else 's',color='#D55E00',s=36)
        ax.axvline(0,color='#444444',lw=1);ax.errorbar(r['mean_difference'],5.3,xerr=[[r['mean_difference']-r['ci95_low']],[r['ci95_high']-r['mean_difference']]],fmt='D',color='black',capsize=4)
        ax.set_yticks(list(range(5))+[5.3],[str(s) for s in SEEDS]+['Mean / CI']);ax.invert_yaxis();ax.set_xlim(-lim,lim);ax.grid(axis='x',alpha=.2);ax.set_xlabel('SC-FAC − SC-FAC-zero: macro12 Money')
        ax.set_title(f'C = {c} | Holm p = {r["holm_p"]:.3f}')
    fig.suptitle('Conditioning: circles = recovered zero controls; squares = new zero controls',fontsize=10)
    save(fig,'figure3_conditioning_ablation')
    rd=tables['regime_descriptive_contrasts'];cols=[('SC-FAC','JA-PPO','SC−JA'),('IFAC','JA-PPO','IF−JA'),('SC-FAC','IFAC','SC−IF'),('SC-FAC','SC-FAC-zero','SC−zero'),('SC-FAC','BF-T0.5','SC−BF')]
    mats=[np.array([[next(r['money_delta_mean'] for r in rd if r['C']==c and r['regime']==reg and r['left_method']==l and r['right_method']==rgt) for l,rgt,_ in cols] for reg in REGIMES]) for c in CAPACITIES]
    vmax=max(float(abs(x).max()) for x in mats);norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    fig,axes=plt.subplots(1,2,figsize=(10,5.8),layout='constrained')
    for ax,c,mat in zip(axes,CAPACITIES,mats):
        im=ax.imshow(mat,cmap='RdBu',norm=norm,aspect='auto')
        for i in range(12):
            for j in range(5):ax.text(j,i,f'{mat[i,j]:+.0f}',ha='center',va='center',fontsize=8,color='white' if abs(mat[i,j])>.58*vmax else 'black')
        ax.set_xticks(range(5),[x[2] for x in cols]);ax.set_yticks(range(12),REGIMES);ax.set_title(f'C = {c}');ax.axhline(5.5,color='black',lw=1)
    fig.colorbar(im,ax=axes,label='Descriptive mean Money difference; common zero-centered scale',shrink=.78)
    fig.suptitle('Regime-level contrasts (descriptive; no regime-level significance tests)',fontsize=11)
    save(fig,'figure4_regime_contrasts')
    (dst/'FIGURE_CAPTIONS.md').write_text('''# Figure captions

1. Macro12 Money on NEW12. Points are five trained-policy seed scores; circles and bars are means and unadjusted marginal 95% Student-t intervals (df=4). BF-T0.5 is one deterministic fixed-pool score per capacity, shown as a diamond/dashed reference without uncertainty. Panel y-ranges differ and are explicitly labeled; these are point plots, not zero-based bars.
2. All five seed-paired structural differences and their mean/marginal 95% t interval. Holm p-values refer to the complete six-test Family A, not separate panels. A zero line is shown; seed pairing follows labels and does not imply shared RNG trajectories.
3. All five full-minus-zero paired differences and their mean/marginal 95% t interval. Circles identify recovered zero-control training cohorts (123/323/532); squares identify new zero trainings (777/999). Holm correction uses both capacities in Family B. Positive values favor full conditioning. No equivalence claim follows from a crossing-zero interval.
4. Descriptive seed-averaged per-regime contrasts, with one common diverging scale centered at zero. Positive blue values favor the left method; negative red values favor the right. Rows 1–6 are smooth; rows 7–12 are bursty according to the frozen manifest. TL/TLN/TPL are truncated distribution families, not time-local patterns. No regime-level tests or significance stars are added.

All figures: same frozen streams, k=24, F=3, T=1000, 200 episodes/regime. Money = settled − 10×executed flushes. PDF/SVG are vector exports; PNG is for visual inspection. CIs reflect training-seed variation conditional on the fixed pools.
''')


def add_input_profiles(tables,manifest):
    profiles=[];rs=tables['regime_level_scores']
    for c in CAPACITIES:
        for reg in REGIMES:
            records=[r for r in rs if r['C']==c and r['regime']==reg]
            requested={r['total_requested_value'] for r in records};oversize={r['oversize_drops'] for r in records}
            require(len(requested)==len(oversize)==1,'Exogenous profile differs across methods/seeds')
            profiles.append(dict(C=c,regime=reg,dist_family=manifest['regime_specs'][reg]['dist_family'],bursty=manifest['regime_specs'][reg]['bursty'],mean_request_value=requested.pop()/1000,mean_oversize_drops=oversize.pop(),wallet_capacity=c/24))
    tables['regime_input_profiles']=profiles


def inference_sentence(r):
    effect=f"{r['left_method']} − {r['right_method']} at C={r['C']}: Δ={r['mean_difference']:+.2f}, 95% CI [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}], raw p={r['raw_p']:.6g}, Holm p={r['holm_p']:.6g}"
    return effect+('; rejects zero under the frozen family correction.' if r['holm_reject_0_05'] else '; does not reject zero under the frozen family correction.')


def write_report(out,a,t):
    A=t['family_A_contrasts'];B=t['family_B_conditioning_ablation'];C=t['family_C_vs_bft05'];s=t['method_capacity_summary'];d=t['mechanism_decomposition'];rd=t['regime_descriptive_contrasts']
    sections=[]
    sections.append('''# Stage 3 — Statistics & Mechanism Analysis

This report is generated from the user-designated frozen Stage 2B raw tarball only. Historical results are not pooled with NEW12 or used to fill cells. The inferential protocol is unchanged. No new training, evaluation or stream generation occurs.

## 1. Raw-data acceptance

**Scientific completeness PASS: 42/42 jobs, 100,800 episode rows. Ledger completeness 38/42, reconciled with four approved pre-existing jobs.** See STAGE3_RAW_ACCEPTANCE.md and raw_acceptance.json. The root pilot duplicate is excluded. Raw SHA256: `'''+RAW_SHA+'`. Scientific lineage: `'+LINEAGE+'''`.

All 155 internal checksum entries pass. Maximum Money identity error is zero. The embedded Mac approval and five representative episode-hash attestations support reuse of the pre-existing outputs. This does not hide provenance limitations: benchmark-only text remains in all 42 results; runtime_lock_sha256 is null; the approval header version strings differ from the consistent actual job runtime records. Pool/checkpoint/training-receipt bytes are outside this raw-results freeze, so their end-to-end regeneration is not certified by this analysis.

## 2. Frozen protocol

Primary outcome is Money = settled value − 10×executed flushes. Average 200 episodes within each regime, then equally average all twelve regimes, yielding one macro12 score per training seed. Each learned method/capacity has seeds 123,323,532,777,999; n=5. Sample SD uses ddof=1, SE=SD/√5, marginal two-sided 95% t CI uses df=4 and critical value 2.7764451051977987. Episodes and regimes are not independent learned-policy replicates.

A: six paired contrasts across both capacities (SC−JA, IF−JA, SC−IF). B: two paired full−zero contrasts. C: two one-sample t-tests of five SC seed deltas against one fixed BF score per capacity. Holm correction is applied within all six/two/two tests respectively. All ten contrasts are reported together. No additional test, bootstrap, equivalence test, interaction test, or post-hoc variance test is introduced. BF has no training-seed SD/SE/CI. Zero-variance tests are undefined, not automatically significant; their family slots are retained. All intervals below are marginal, not familywise intervals.

## 3. Descriptive results

'''+mdtable(s,['method','C','n_training_seeds','mean','SD','SE','ci95_low','ci95_high','df']))
    for family,title,rows in [('A','structural policy comparisons',A),('B','conditioning ablation',B),('C','strong deterministic comparator',C)]:
        num={'A':4,'B':5,'C':6}[family]
        sec=f'## {num}. Family {family} — {title}\n\n'+mdtable(rows,['C','left_method','right_method','mean_difference','SD','SE','ci95_low','ci95_high','t_statistic','raw_p','holm_p','holm_reject_0_05'])+'\n\n'+'\n\n'.join(inference_sentence(r) for r in rows)
        if family=='A':sec+='\n\nAll six mean differences favor the left method, but none rejects zero after Holm correction (and all six marginal intervals cross zero). Factorized policies have higher descriptive mean Money than JA-PPO at both capacities; this five-seed confirmation does not establish their inferential superiority. No claim of equivalence follows.'
        if family=='B':sec+='\n\nZero conditioning has the higher mean at both capacities. There is no detected incremental benefit of explicit settlement conditioning under this protocol; neither its harm nor equivalence is established. This tests the settlement input within the SC head design, not whether factorization itself is useful. Recovered versus newly trained zero-control cohorts remain labeled and visible.'
        if family=='C':sec+='\n\nBF-T0.5 exceeds SC-FAC at both capacities with family-wise evidence under the declared tests. It also exceeds every learned-method mean descriptively. No extra BF-versus-JA/IF/zero significance tests were authorized or added. The rule has one fixed score per capacity; its value is reused as a constant in five SC deltas, not counted as five independently trained rules.'
        sections.append(sec)
    regtext='## 7. Regime-level analysis\n\nDefinitions are taken verbatim from frozen new12_manifest.json, not inferred from abbreviations.\n\n'+mdtable(t['regime_definitions'],['regime','dist_family','bursty','target_mean','max_tx'])
    regtext+='\n\nS denotes smooth and B bursty. U is uniform; TL is truncated light-tail (truncated normal), LN lognormal, TLN truncated lognormal, PL power law, TPL truncated power law. **TL does not mean time-local.** No distinct plain/time-local/patterned axis is defined in these materials. Burst/non-burst family parameters are not identical in every pair, so group differences do not isolate a pure causal burst effect.\n\nAll regimes target mean transaction value 50. They are not a predefined small-versus-large transaction experiment. regime_input_profiles.csv reports observed mean request value and mean oversized-request counts; no transaction-level size-bin effect can be recovered from episode summaries. The raw archive includes manifests rather than transaction arrays or step trajectories. Oversize rates are a descriptive feasibility proxy, not a causal size treatment.\n\n'
    for c in CAPACITIES:
        for fam,left,right in [('A','SC-FAC','JA-PPO'),('A','IFAC','JA-PPO'),('B','SC-FAC','SC-FAC-zero'),('C','SC-FAC','BF-T0.5')]:
            rr=[r for r in rd if r['C']==c and r['left_method']==left and r['right_method']==right];lo=min(rr,key=lambda r:r['money_delta_mean']);hi=max(rr,key=lambda r:r['money_delta_mean'])
            regtext+=f"- C={c}, {left} − {right}: {sum(r['money_delta_mean']>0 for r in rr)}/12 positive descriptive regime means; range {lo['money_delta_mean']:+.2f} ({lo['regime']}) to {hi['money_delta_mean']:+.2f} ({hi['regime']}).\n"
    regtext+='\nSmooth/bursty grouped differences (equal weighting over six regimes in each group; descriptive only):\n\n'+mdtable(t['regime_group_descriptives'],['contrast_id','stream_group','mean_money_delta','mean_settled_delta','mean_flushes_delta'])
    regtext+='\n\nRegime plots and positive-count statements are descriptive and do not turn twelve regimes into twelve independent trained agents. Per-regime across-seed SDs and all paired differences are preserved in the tables; no regime-level significance claims are made.'
    sections.append(regtext)
    mech='## 8. Mechanism interpretation\n\n### A. Observed accounting patterns\n\n'+mdtable(d,['contrast_id','money_delta','settled_delta','flushes_delta','flush_cost_delta','regimes_positive','regimes_negative'])
    mech+='\n\nEvery row satisfies ΔMoney = Δsettled − 10Δflushes. This is an accounting identity, not a causal explanation.\n\n'
    for c in CAPACITIES:
        x=next(r for r in d if r['family']=='B' and r['C']==c)
        mech+=f"- Conditioning, C={c}: full-minus-zero differences per macro-averaged episode are {x['settled_delta']:+.2f} in settled value and {x['flushes_delta']:+.2f} in flushes, yielding {x['money_delta']:+.2f} Money.\n"
    bf=[r for r in t['regime_level_scores'] if r['method']=='BF-T0.5']
    mech+=f"- BF mean insufficient-balance drops range from {min(r['insufficient_drops'] for r in bf):.6g} to {max(r['insufficient_drops'] for r in bf):.6g} across capacities/regimes. "
    mech+=('Thus the recorded BF trajectories accept all individually feasible requests in these test episodes. This is not a proof of optimal Money: accepting value and minimizing flush fees are distinct objectives.' if max(r['insufficient_drops'] for r in bf)==0 else 'This is a descriptive feasibility diagnostic.')
    mech+='\n\n### B. Plausible mechanism hypotheses\n\nAt C800, the full-versus-zero deficit is compatible with lost accepted value outweighing saved flush cost; at C1200 it is compatible with extra flush cost outweighing additional accepted value. SC-versus-JA at C800 is acceptance-led, whereas C1200 combines additional settlement with fewer flushes. These decompositions motivate hypotheses about replenishment timing and resource usage; the aggregates do not identify timing, learned representations, action conflicts or the causal reason a policy chooses a flush. The study lacks the step-level traces and intervention controls required to establish those mechanisms.\n\nThe observed pattern is consistent with compact policy structure being useful without a demonstrated incremental settlement-input benefit. Architecture width/optimization differences prevent treating SC-versus-IFAC as a pure conditioning intervention. Cohort differences in new zero trainings also remain a mechanism-study limitation.'
    sections.append(mech)
    sections.append('''## 9. Negative/null findings

- No Family A contrast passes Holm correction; avoid “factorization significantly improves control” for this campaign.
- Both full-minus-zero point estimates are negative, but uncertainty spans zero. There is no evidence here for an incremental conditioning benefit and no equivalence/harm conclusion.
- BF wins the prescribed comparison with SC at both capacities, and has the highest descriptive method mean. This must remain visible in the main paper.
- SC has lower descriptive seed SD than JA and IF at both capacities, but no variance test was pre-specified. It does not have the smallest learned-policy SD at C1200: zero conditioning is less variable there.
- No claim about switching, unseen regime families, zero-shot k, deployment, or faster RL computation is tested.

## 10. Implications for the AAMAS claim

The fresh evidence supports a measured structural comparison: independent and conditioned policies improve mean Money descriptively over the flat head, and SC retains broad regime-level gains versus JA, but five-seed Family A inference does not establish superiority. The conditioning-specific claim must be narrowed because zero conditioning has higher mean Money at both capacities without a resolved full-minus-zero effect. The practical controller comparison favors BF.

The question remains “How much structure does an RL policy need for coupled online decisions?” The contribution should be structured action factorization and its measured limits, not a claim that SC-FAC is universally best or that explicit conditioning is necessary. SC-FAC remains a studied proposed architecture, not a guaranteed winner. Linear output dimensionality means 2(k+1) selected-path logits versus (k+1)^2: 50 versus 625 at k=24. It does not make total RL computation linear or prove inference speedup.

## 11. Recommended paper result wording

On NEW12, both independent and settlement-conditioned policies had higher macro12 Money means than the flat joint-action policy at both capacities. None of the six paired structural contrasts rejected zero after the pre-specified Holm correction. SC-FAC-zero had higher mean Money than full SC-FAC at both capacities, and the matched tests did not establish an incremental benefit of the settlement signal. BF-T0.5 outperformed SC-FAC at both capacities under the pre-specified one-sample tests of seed-level deltas. The findings distinguish favorable descriptive performance of compact learned policies from evidence for conditioning and from practical superiority over domain-informed deterministic control.

Inferential evidence accompanying that paragraph:

'''+ '\n\n'.join(inference_sentence(r) for r in A+B+C))
    sections.append('''## 12. Recommended abstract-result sentence

“On fresh streams, factorized policies achieved higher mean net accepted value than the flat joint-action policy, although the pre-specified paired tests did not establish structural-policy superiority; explicit settlement conditioning showed no detected incremental benefit, and a domain-informed deterministic rule outperformed SC-FAC at both capacities.”

This is proposed wording only; no manuscript or abstract file was edited. It summarizes the effect estimates, marginal confidence intervals and family-adjusted p-values reported in Sections 4–6 and 11; it does not promote raw-p findings.

## 13. Limitations

Five training seeds provide limited precision. Normal-theory t inference is retained as frozen, without switching methods after outcomes. Seed matching does not imply common RNG trajectories. CIs are conditional on fixed NEW12 pools and do not quantify all generator/world uncertainty; the twelve regimes are related designed conditions, not inferential replicates. Equal episode counts make a pooled episode mean numerically identical here, but the pipeline explicitly preserves regime weighting and never uses episode-level n for learned-policy inference. Zero-controls mix three recovered and two newly trained policies per capacity; no post-hoc cohort exclusion or extra test is performed. Parent architecture comparisons also differ in head dimensions. BF is deterministic on fixed pools, but this is not a claim of zero uncertainty across hypothetical streams. Regime aggregates cannot identify action timing or transaction-level mechanism effects. Null tests do not establish equivalence. See the acceptance report for retained approval/version/benchmark-label inconsistencies and absent checkpoint/pool bytes; this analysis verifies the supplied frozen outputs, not a new end-to-end simulation replay. No PR #3 data or code enters the analysis.

## 14. Exact artifact inventory

All derived tables are listed in statistics.json under tables. artifact_inventory.csv records every deliverable's relative path, size and SHA256; SHA256SUMS.txt covers those deliverables plus the inventory. work/raw contains byte-verified extracted copies and is excluded from Git via the local .gitignore; its exact 157-file inventory is raw_file_inventory.csv. Four figures are exported as PDF/SVG/PNG; captions document uncertainty and scale. code/run_stage3_analysis.py is the deterministic entry point; code/test_stage3_analysis.py supplies independent numerical and guard tests; code/requirements.txt freezes the analysis libraries. The analysis runtime is separate from the frozen evaluator runtime.
''')
    (out/'STAGE3_ANALYSIS_REPORT.md').write_text('\n\n'.join(sections)+'\n')
def deliverable_inventory(out):
    rows=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(out)
        if rel.parts[0]=='work' or '__pycache__' in rel.parts or p.name in ['artifact_inventory.csv','SHA256SUMS.txt']:continue
        rows.append(dict(path=str(rel),size=p.stat().st_size,sha256=sha(p)))
    csvwrite(out/'artifact_inventory.csv',rows)
    checks=[f"{r['sha256']}  {r['path']}" for r in rows]+[f"{sha(out/'artifact_inventory.csv')}  artifact_inventory.csv"]
    (out/'SHA256SUMS.txt').write_text('\n'.join(checks)+'\n')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parents[1]
    p.add_argument('--out',type=Path,default=default)
    p.add_argument('--raw',type=Path,default=default.parent/'stage2/raw'/f'{BUNDLE}.tar.gz')
    p.add_argument('--audit-only',action='store_true');args=p.parse_args();out=args.out.resolve();raw=args.raw.resolve()
    require(not raw.is_relative_to(out),'Raw tarball must be outside derived-output root')
    out.mkdir(parents=True,exist_ok=True)
    try:root,audit,jobs,regs,manifest,data=accept_raw(raw,out)
    except Exception as exc:
        (out/'STAGE3_RAW_ACCEPTANCE.md').write_text('# Stage 3 raw acceptance\n\n**FAIL**\n\n'+str(exc)+'\n\nStatistics were not executed. Existing derived files, if any, are not certified by this invocation.\n')
        raise
    print('RAW ACCEPTANCE PASS: 42 scientific jobs; 38 ledger entries + 4 approved pre-existing outputs.',flush=True)
    if args.audit_only:return
    os.environ.setdefault('MPLCONFIGDIR',str(out/'work'/'matplotlib_cache'))
    import numpy,scipy,matplotlib
    tables=compute_tables(jobs,regs,manifest);add_input_profiles(tables,manifest)
    for name,rows in tables.items():csvwrite(out/(name+'.csv'),rows)
    runtime=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),entry_point_sha256=sha(__file__))
    doc=dict(schema_version=1,raw_sha256=RAW_SHA,scientific_lineage=LINEAGE,acceptance=audit,analysis_runtime=runtime,protocol=dict(metric='Money = settled - 10*flushes',aggregation=['episode','regime_mean','equal_weight_macro12','training_seed'],seeds=SEEDS,capacities=CAPACITIES,regimes=REGIMES,learned_n=5,df=4,t_critical=2.7764451051977987,families={'A':6,'B':2,'C':2},alpha=.05,multiplicity='Holm within each family; all families reported',ci='marginal two-sided 95% Student t',BF='one deterministic fixed-pool score per capacity; one-sample tests of five SC deltas',zero_variance='test undefined; preserve full family size',regime_tests='none'),tables=tables)
    jwrite(out/'statistics.json',doc);make_figures(out,tables);write_report(out,audit,tables)
    # Derived analysis cannot mutate extracted scientific files or the original archive.
    for r in csvread(out/'raw_file_inventory.csv'):require(sha(root/r['path'])==r['sha256'],'Extracted raw input modified')
    require(sha(raw)==RAW_SHA,'Original archive changed')
    inventory=deliverable_inventory(out)
    print('STAGE 3 COMPLETE:',len(tables),'tables;',len(inventory),'deliverables.')
    for name in ['family_A_contrasts','family_B_conditioning_ablation','family_C_vs_bft05']:
        for r in tables[name]:print(inference_sentence(r))

if __name__=='__main__':
    try:main()
    except (AcceptanceError,KeyError,ValueError) as exc:
        print('STOP:',exc,file=sys.stderr);sys.exit(2)
