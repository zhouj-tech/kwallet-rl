#!/usr/bin/env python3
"""Isolated AAMAS evaluator. No training or stream generation entry point.

OLD parity is available before G3. NEW12 requires explicit release receipts.
Historical modules are imported by exact, hash-pinned file identity.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

# Loading historical modules must not create __pycache__ in immutable source trees.
sys.dont_write_bytecode = True

REGIMES = 'US TLS LNS TLNS TPLS PLS UB TLB LNB TLNB TPLB PLB'.split()
ENV = 'src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py'
MODULES = {
 'JA-PPO': 'src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py',
 'IFAC': 'src/idea4/ac/code/run_factorized_ac_benchmark.py',
 'SC-FAC': 'src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py',
 'SC-FAC-zero': 'src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py',
 'BF-T0.5': 'src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py',
}
SOURCE_HASHES = {
 ENV: 'd46246795552f22f0ba143ae38230692ff99997d57c9b0681f456432a9df1921',
 MODULES['JA-PPO']: 'e4a669678453a2884ac6834ee599dd85a98e6bac59ac59630513f7c2d022d0ea',
 MODULES['IFAC']: '365971c27702d7c05f0a77dbc9f94ed80b3fd2fe470c7ce925409effef2a2971',
 MODULES['SC-FAC']: 'c3710b3a8bbf905f2d047b5bf4444830cdc32239e39b49becc149540f6e79b04',
 'src/ideaextra/kwallet_ideaextra_generator.py': 'd702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8',
}
CORE = 'settled flushes drops oversize_drops insufficient_drops accepted_count total_tx_count total_requested_value value_accept_ratio count_accept_ratio utilization eval_money eval_money_p eval_money_tau'.split()
GATES = {'G0_STAGE1_APPROVED', 'G1_ARTIFACT_RUNTIME', 'G2_NEW12_FROZEN', 'G3_ADAPTER_ACCEPTED'}
RULE = 'BF-T0.5:v1;usable=E0._usable;settle=min(balance,index):balance>=tx;flush=min(balance,index):usable,not_settle,balance<0.5*C/k;noop=k;full_refill=E0;oversize_still_flush'

class Stop(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise Stop(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def object_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def read_json(path, expected=None):
    if expected is not None:
        check_hash(path, expected)
    return json.loads(Path(path).read_text())

def check_hash(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, 'Missing valid SHA-256')
    require(Path(path).is_file(), f'Missing file: {path}')
    require(digest(path) == expected, f'SHA-256 mismatch: {path}')

def within(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, f'Unsafe relative path: {relative}')
    p = (root / rel).resolve()
    require(p.is_relative_to(root), 'Path escapes root')
    return p

def remap(path, prefix, root):
    try:
        rel = Path(path).relative_to(prefix)
    except ValueError as exc:
        raise Stop(f'Unexpected recorded path: {path}') from exc
    return within(root, rel)

def safe_output(path, protected):
    p = Path(path).resolve()
    for root in protected:
        root = Path(root).resolve()
        require(not p.is_relative_to(root) and not root.is_relative_to(p), f'Output overlaps protected root: {p}')
    require(not p.exists(), f'Output exists; refusing overwrite: {p}')
    return p

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def publish(destination, files):
    """Prepare complete files in a hidden sibling; atomically publish once valid."""
    destination = Path(destination)
    require(not destination.exists(), 'Output already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.incomplete-', dir=destination.parent))
    try:
        for name, content in files.items():
            (tmp / name).write_text(content)
        require(not destination.exists(), 'Concurrent output collision')
        tmp.rename(destination)
    except BaseException:
        # Preserve incomplete directory for diagnosis; never publish a success receipt.
        raise

def bf_action(env):
    usable = [i for i in range(env.k) if env._usable(i)]
    candidates = [i for i in usable if env.wallets[i] >= env.current_tx]
    s = min(candidates, key=lambda i: (env.wallets[i], i)) if candidates else env.k
    eligible = [i for i in usable if i != s and env.wallets[i] < 0.5 * (env.C / env.k)]
    f = min(eligible, key=lambda i: (env.wallets[i], i)) if eligible else env.k
    return s * (env.k + 1) + f

def validate_episode(r):
    require(all(k in r for k in CORE), 'Missing required episode metric')
    for k, v in r.items():
        require(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), f'Invalid numeric metric {k}')
    for k in ['flushes', 'drops', 'oversize_drops', 'insufficient_drops', 'accepted_count', 'total_tx_count']:
        require(0 <= r[k] <= 1000 and float(r[k]).is_integer(), f'Invalid count {k}')
    require(r['total_tx_count'] == 1000, 'Wrong episode length')
    require(r['drops'] + r['accepted_count'] == 1000, 'Acceptance/drop identity failed')
    require(r['oversize_drops'] + r['insufficient_drops'] == r['drops'], 'Drop breakdown failed')
    require(r['eval_money_p'] == 1 and r['eval_money_tau'] == 10, 'Wrong Money coefficients')
    require(r['total_requested_value'] > 0 and 0 <= r['settled'] <= r['total_requested_value'], 'Invalid value totals')
    equalities = [
        (r['eval_money'], r['settled'] - 10 * r['flushes']),
        (r['value_accept_ratio'], r['settled'] / r['total_requested_value']),
        (r['count_accept_ratio'], r['accepted_count'] / 1000),
    ]
    for a, b in equalities:
        require(math.isclose(a, b, rel_tol=0, abs_tol=1e-9), 'Metric accounting mismatch')
    for k in ['value_accept_ratio', 'count_accept_ratio', 'utilization']:
        require(0 <= r[k] <= 1, f'Invalid ratio {k}')

def validate_result(result, count=200):
    require(result.get('num_episodes') == count and len(result.get('raw_results', [])) == count, 'Incomplete evaluator result')
    import numpy as np
    for r in result['raw_results']:
        validate_episode(r)
    require(len({tuple(sorted(r)) for r in result['raw_results']}) == 1, 'Inconsistent metric keys')
    summary = result.get('summary', {})
    for key in result['raw_results'][0]:
        a = np.asarray([r[key] for r in result['raw_results']], dtype=np.float64)
        expected = dict(mean=float(np.mean(a)), std=float(np.std(a)), min=float(np.min(a)), max=float(np.max(a)), median=float(np.median(a)))
        require(key in summary, f'Missing summary: {key}')
        for stat, value in expected.items():
            observed = summary[key].get(stat)
            require(isinstance(observed, (int, float)) and math.isfinite(observed) and math.isclose(observed, value, rel_tol=0, abs_tol=1e-9), f'Summary mismatch: {key}.{stat}')
        if 'values' in summary[key]:
            require(summary[key]['values'] == a.tolist(), 'Summary episode values mismatch')

def load_jobs(path):
    jobs = list(csv.DictReader(Path(path).open()))
    require(len(jobs) == 46 and len({j['job_id'] for j in jobs}) == 46, 'Expected 46 unique jobs')
    require(sum(j['job_type'] == 'training' for j in jobs) == 4, 'Expected four training rows')
    require(len({j['output_path'] for j in jobs}) == 46, 'Duplicate output path')
    expected = {(m, str(c), str(s)) for m in ['JA-PPO','IFAC','SC-FAC','SC-FAC-zero'] for c in [800,1200] for s in [123,323,532,777,999]}
    expected |= {('BF-T0.5',str(c),'NA') for c in [800,1200]}
    evaluations = [j for j in jobs if j['job_type'] == 'evaluation']
    require(len(evaluations) == 42 and {(j['method'],j['C'],j['seed']) for j in evaluations} == expected, 'Wrong evaluation matrix')
    trainings = [j for j in jobs if j['job_type'] == 'training']
    require({(j['C'], j['seed']) for j in trainings} == {(str(c),str(s)) for c in [800,1200] for s in [777,999]}, 'Wrong training matrix')
    for j in jobs:
        require((j['k'],j['F'],j['T']) == ('24','3','1000'), 'Wrong environment settings')
        deps = set(j['dependencies'].split(';'))
        if j['job_type'] == 'training':
            require(deps == {'G0_STAGE1_APPROVED','G1_ARTIFACT_RUNTIME'}, 'Training must depend only on G0/G1')
        else:
            require(GATES <= deps and j['training_required'] == 'false', 'Missing evaluation gates')
            require(j['evaluation_stream'] == 'NEW12-v1', 'Wrong evaluation stream')
    return {j['job_id']:j for j in jobs}

def verify_sources(repo):
    for rel, h in SOURCE_HASHES.items():
        check_hash(within(repo, rel), h)
    return dict(SOURCE_HASHES)

def artifact_inventory(path, root):
    records = list(csv.DictReader(Path(path).open()))
    require(len(records) == 236, 'Unexpected artifact manifest size')
    require(len({r['transfer_relative_path'] for r in records}) == len(records), 'Duplicate artifact path')
    require(len({r['sha256'] for r in records}) == len(records), 'Duplicate artifact bytes')
    require(sum(r['artifact_type'] == 'best_checkpoint' for r in records) == 36, 'Expected 36 recovered checkpoints')
    require(sum(r['artifact_type'] == 'pool' for r in records) == 14, 'Expected 14 historical pools')
    for r in records:
        p = within(root, r['transfer_relative_path'])
        check_hash(p, r['sha256'])
        require(p.stat().st_size == int(r['size']), f'Artifact size mismatch: {p}')
    return records

def pick(records, kind, method=None, c=None, seed=None, regime=None):
    candidates = [r for r in records if r['artifact_type'] == kind and (method is None or r['method'] == method) and (c is None or r['C'] == str(c)) and (seed is None or r['training_seed'] == str(seed)) and (regime is None or r['regime'] == regime)]
    require(len(candidates) == 1, f'Ambiguous/missing artifact: {kind}/{method}/{c}/{seed}/{regime}')
    return candidates[0]

def git_state(repo):
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args], text=True)
    return {'git_sha':git('rev-parse','HEAD').strip(), 'git_status':git('status','--porcelain'), 'tracked_diff_sha256':hashlib.sha256(git('diff','HEAD','--').encode()).hexdigest()}

def runtime_snapshot():
    import numpy as np
    import torch
    import matplotlib
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        np.show_config()
    return {
        'python_executable':sys.executable, 'python_version':platform.python_version(),
        'platform':platform.platform(), 'machine':platform.machine(), 'processor':platform.processor(),
        'numpy':np.__version__, 'torch':torch.__version__, 'matplotlib':matplotlib.__version__,
        'numpy_build':b.getvalue(), 'torch_build':torch.__config__.show(),
        'torch_threads':torch.get_num_threads(), 'torch_interop_threads':torch.get_num_interop_threads(),
        'torch_default_dtype':str(torch.get_default_dtype()), 'device':'cpu',
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONHASHSEED','XDG_CACHE_HOME','MPLCONFIGDIR','MPLBACKEND']},
        'packages':sorted((d.metadata.get('Name',''),d.version) for d in importlib.metadata.distributions()),
    }

def runtime_document(args):
    verify_sources(args.repo)
    require(Path(args.stage1_report).is_file(), 'Stage 1 report required')
    return {'schema_version':1, 'snapshot':runtime_snapshot(), 'stage1_report_sha256':digest(args.stage1_report), 'source_hashes':SOURCE_HASHES,
        'artifact_manifest_sha256':digest(args.manifest), 'job_matrix_sha256':digest(args.jobs),
        'adapter_sha256':digest(__file__), 'stage1_approved':True, 'runtime_confirmed':False,
        'stage1_interpretation':'JA/IF historical PASS; SC current-runtime exact repeats PASS; accepted negligible historical drift is not a failed gate.',
        'git':git_state(args.repo)}

def verify_runtime(args):
    lock = read_json(args.runtime_lock, args.runtime_lock_sha256)
    require(lock.get('schema_version') == 1 and lock.get('stage1_approved') is True and lock.get('runtime_confirmed') is True, 'Runtime capture not confirmed against approved Stage 1 runtime')
    require(lock['source_hashes'] == SOURCE_HASHES, 'Runtime source hash mismatch')
    for key, path in [('artifact_manifest_sha256',args.manifest),('job_matrix_sha256',args.jobs),('adapter_sha256',__file__)]:
        check_hash(path, lock[key])
    require(canonical(lock['snapshot']) == canonical(runtime_snapshot()), 'Current runtime differs from locked approved runtime')
    return lock

def import_original(repo, method):
    verify_sources(repo)
    for v in ['XDG_CACHE_HOME','MPLCONFIGDIR']:
        require(os.environ.get(v), f'Set writable external {v} before importing historical code')
    env_name = 'kwallet_ctx_attn_fair_benchmark'
    def load(name, rel):
        p = within(repo, rel)
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == p, 'Unexpected existing module origin')
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    load(env_name, ENV)
    module = load('aamas_stage2_' + Path(MODULES[method]).stem, MODULES[method])
    require(module.KWalletEnv is sys.modules[env_name].KWalletEnv, 'Wrong environment class')
    for m in list(sys.modules.values()):
        p = str(getattr(m,'__file__','') or '').replace('\\','/')
        require('/src/kwallet/' not in p and '/collab/' not in p, 'Forbidden collaborator module dependency')
    return module

def check_config(cfg, job):
    require(cfg['env']['C'] == int(job['C']) and cfg['env']['k'] == 24 and cfg['env']['F'] == 3 and cfg['env']['T'] == 1000, 'Config/environment mismatch')
    require(cfg['seed'] == int(job['seed']), 'Config training seed mismatch')
    require(cfg['train']['device'] == 'cpu' and not cfg['env']['enable_shaping'], 'Unapproved device/shaping')
    require(cfg['reward']['reward_mode'] == 'original' and cfg['reward']['money_p'] == 1 and cfg['reward']['money_tau'] == 10, 'Wrong reward/objective')
    expected_mode = {'JA-PPO':'basic_ppo','IFAC':'factorized_ac','SC-FAC':'conditional_factorized_ac','SC-FAC-zero':'conditional_factorized_ac'}[job['method']]
    require(cfg['model_mode'] == expected_mode, 'Wrong policy family')
    if job['method'].startswith('SC-FAC'):
        mode = 'zero_settle_embedding' if job['method']=='SC-FAC-zero' else 'full'
        require(cfg.get('condition_mode','full') == mode and cfg['conditional'].get('condition_mode','full') == mode and job['condition_mode'] == mode, 'Condition mode mismatch')
        require(cfg['conditional']['settle_embed_dim'] == 32 and cfg['conditional']['conditional_hidden_size'] == 256, 'Wrong conditional head shape')

def normalized_config(cfg):
    c = copy.deepcopy(cfg)
    c.pop('seed'); c['env'].pop('C'); c.pop('output')
    return c

def resolve_model(args, job, records):
    if job['method'] == 'BF-T0.5':
        r = pick(records,'run_info','JA-PPO',job['C'],123)
        cfg = read_json(within(args.artifacts,r['transfer_relative_path']))['config']
        check_config(cfg, dict(job, method='JA-PPO', seed='123'))
        return cfg, None, None, r['sha256']
    if job['checkpoint_source'].startswith('receipt:'):
        token = job['checkpoint_source'][len('receipt:'):].split('#')[0]
        path = remap(token,'/data/sijia/aamas2027_stage2',args.work_root)
        receipt = read_json(path)
        tid = job['job_id'].replace('B-EVAL-', 'B-TRAIN-')
        require(receipt.get('status') == 'complete' and receipt.get('job_id') == tid, 'Incomplete/wrong training receipt')
        require(receipt.get('runtime_lock_sha256') == args.runtime_lock_sha256 and receipt.get('source_hashes') == SOURCE_HASHES, 'Training runtime/source mismatch')
        cp = remap(receipt['best_checkpoint_path'],'/data/sijia/aamas2027_stage2',args.work_root)
        ip = remap(receipt['run_info_path'],'/data/sijia/aamas2027_stage2',args.work_root)
        training_root = within(args.work_root, 'training/' + tid)
        require(cp.is_relative_to(training_root) and ip.is_relative_to(training_root) and cp.name=='best_model.pth', 'Training receipt escapes its job')
        check_hash(cp,receipt['best_checkpoint_sha256']);check_hash(ip,receipt['run_info_sha256'])
        cfg = read_json(ip)['config']
        reference = pick(records,'run_info','SC-FAC-zero',800,123)
        old = read_json(within(args.artifacts,reference['transfer_relative_path']))['config']
        require(normalized_config(cfg)==normalized_config(old),'New zero training protocol differs')
        require(receipt.get('config_sha256') == object_hash(cfg),'Training config hash mismatch')
        validate_training_receipt(receipt, args, training_root)
        sha=receipt['best_checkpoint_sha256']; info_sha=receipt['run_info_sha256']
    else:
        cr = pick(records,'best_checkpoint',job['method'],job['C'],job['seed'])
        ir = pick(records,'run_info',job['method'],job['C'],job['seed'])
        cp = remap(job['checkpoint_source'],'/data/sijia/aamas2027_artifacts',args.artifacts)
        ip = remap(job['run_info_source'],'/data/sijia/aamas2027_artifacts',args.artifacts)
        require(cp==within(args.artifacts,cr['transfer_relative_path']) and ip==within(args.artifacts,ir['transfer_relative_path']), 'Job paths differ from artifact manifest')
        require(job['checkpoint_sha256']==cr['sha256'],'Job checkpoint hash mismatch')
        check_hash(cp,cr['sha256']);check_hash(ip,ir['sha256'])
        cfg=read_json(ip)['config'];sha=cr['sha256'];info_sha=ir['sha256']
    check_config(cfg, job)
    return cfg, cp, sha, info_sha

def validate_training_receipt(receipt, args, training_root):
    records = {}
    for kind in ['training_history', 'validation_history', 'result']:
        path = remap(receipt[kind + '_path'], '/data/sijia/aamas2027_stage2', args.work_root)
        require(path.is_relative_to(training_root), 'Training evidence escapes job root')
        records[kind] = read_json(path, receipt[kind + '_sha256'])
    require(receipt.get('exit_status') == 0, 'Unsuccessful training exit')
    returns = records['training_history']['returns']
    require(len(returns) == 1000 and all(math.isfinite(x) for x in returns), 'Incomplete training history')
    vals = records['validation_history']['validation_history']
    require([v['episode'] for v in vals] == list(range(50,1001,50)), 'Wrong validation cadence')
    require(all(v['metric']=='value_accept_ratio' and math.isfinite(v['score']) and v['score']==v['value_accept_ratio'] for v in vals), 'Wrong checkpoint selection metric')
    best = max(vals, key=lambda v: v['score'])  # first maximum preserves strict > tie rule
    require(receipt['selected_validation_episode']==best['episode'] and receipt['selected_validation_score']==best['score'], 'Wrong BEST selection')
    require(object_hash(records['result']['config'])==receipt['config_sha256'], 'Training result config differs')


def learned_evaluation(module, method, cfg, checkpoint, pool, label):
    import torch
    # Do not call main/train_agent/model.eval: preserve original evaluator inference mode.
    env=module.make_env(cfg,max_steps=1000)
    if method=='JA-PPO':
        agent=module.BasicPPOAgent(config=cfg,state_size=env.state_size,k=env.k)
    elif method=='IFAC':
        agent=module.FactorizedPPOAgent(config=cfg,state_size=env.state_size,base_state_size=env.base_state_size,window_size=cfg['attention_context']['window_size'],k=env.k)
    else:
        agent=module.ConditionalFactorizedPPOAgent(config=cfg,state_size=env.state_size,base_state_size=env.base_state_size,k=env.k)
    agent.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return module.evaluate_agent_on_array(agent,cfg,pool,label,200,1000)

def rule_evaluation(module,cfg,pool,label):
    # JA make_env maps to baseline E0; delegate transitions and all environment metrics.
    env=module.make_env(cfg,max_steps=1000)
    rows=[]
    for episode in pool:
        env.reset(tx_stream=episode)
        requested=0.0;accepted=0
        for t in range(1000):
            requested+=float(env.current_tx)
            _,_,done,info=env.step(bf_action(env))
            accepted+=int(bool(info.get('accepted',False)))
            require(done == (t==999), 'Unexpected episode termination')
        metrics=env.get_metrics()
        metrics.update(value_accept_ratio=metrics['settled']/requested,count_accept_ratio=accepted/1000,total_requested_value=requested,total_tx_count=1000,accepted_count=accepted)
        module.add_eval_money_metrics(metrics,cfg);rows.append(metrics)
    summary=module.summarize_episode_metrics(rows)
    module.add_reward_summary_metadata(summary,cfg)
    return dict(label=label,num_episodes=200,summary=summary,raw_results=rows)

def evaluate_one(module,method,cfg,checkpoint,pool,label):
    result = rule_evaluation(module,cfg,pool,label) if method=='BF-T0.5' else learned_evaluation(module,method,cfg,checkpoint,pool,label)
    validate_result(result)
    return result

def load_pool(path, expected, episode_hashes=None):
    import numpy as np
    check_hash(path,expected)
    pool=np.load(path,allow_pickle=False)
    require(pool.shape==(200,1000) and pool.dtype==np.dtype('<i4') and pool.flags.c_contiguous,'Wrong pool shape/dtype/layout')
    require(np.isfinite(pool).all() and (pool>0).all(),'Invalid transaction values')
    hashes=[hashlib.sha256(row.astype('<i4',copy=False).tobytes(order='C')).hexdigest() for row in pool]
    if episode_hashes is not None:require(hashes==episode_hashes,'Episode hashes differ')
    pool.flags.writeable=False
    return pool,hashes

def old_streams(args,records,regimes):
    streams=[]
    for reg in regimes:
        r=pick(records,'pool',regime=reg)
        sr=pick(records,'pool_summary',regime=reg)
        summary=read_json(within(args.artifacts,sr['transfer_relative_path']))
        streams.append(dict(regime=reg,path=within(args.artifacts,r['transfer_relative_path']),sha256=r['sha256'],episode_seeds=list(range(summary['seed'],summary['seed']+200))))
    return streams

def verify_new_release(args):
    require(args.allow_new12, 'NEW12 evaluation requires explicit --allow-new12')
    freeze=read_json(args.freeze_receipt,args.freeze_receipt_sha256)
    require(freeze.get('dataset_id')=='NEW12-v1' and freeze.get('approved') is True,'NEW12 not frozen/approved')
    mf=within(args.new12_root,'new12_manifest.json');doc=read_json(mf,freeze['manifest_sha256'])
    require(doc.get('dataset_id')=='NEW12-v1' and doc.get('regime_order')==REGIMES and doc.get('no_results_viewed_attestation') is True,'Invalid NEW12 release')
    require(doc.get('schema_version')==1 and doc.get('bit_generator')=='PCG64' and doc.get('calibration_seed')==1234567 and doc.get('calibration_sample_size')==200000,'Wrong generator protocol')
    bindings={'runtime_lock_sha256':args.runtime_lock_sha256,'job_matrix_sha256':digest(args.jobs),'OLD_manifest_sha256':digest(args.manifest),'adapter_sha256':digest(__file__),'plan_sha256':digest(args.plan),'generator_sha256':SOURCE_HASHES['src/ideaextra/kwallet_ideaextra_generator.py']}
    for key,value in bindings.items():require(doc.get(key)==value, f'NEW12 binding mismatch: {key}')
    overlap=read_json(within(args.new12_root,'overlap_report.json'),doc['overlap_report_sha256'])
    require(overlap.get('status')=='PASS' and overlap.get('collision_count')==0 and overlap.get('old_episode_count')==7700 and overlap.get('new_episode_count')==2400,'Overlap gate incomplete')
    g3=read_json(args.g3_receipt,args.g3_receipt_sha256)
    require(g3.get('approved') is True and g3.get('gate')=='G3_ADAPTER_ACCEPTED','G3 not accepted')
    for key in ['runtime_lock_sha256','job_matrix_sha256','adapter_sha256']:
        require(g3.get(key)==bindings[key],f'G3 binding mismatch: {key}')
    require(set(g3.get('validated_methods',[]))==set(MODULES),'G3 method coverage incomplete')
    require(set(g3.get('capacities',[]))=={800,1200},'G3 capacity coverage incomplete')
    verify_g3_evidence(g3, args)
    pools=doc.get('pools',[])
    require(len(pools)==12 and [p['regime'] for p in pools]==REGIMES,'NEW12 pool coverage mismatch')
    streams=[]
    for i,p in enumerate(pools):
        require(p['filename']==REGIMES[i]+'_NEW12_v1_T1000.npy','Wrong NEW12 filename')
        require(p['episode_count']==200 and p['T']==1000 and p['dtype']=='int32' and p['order']=='C','NEW12 dimensions/dtype mismatch')
        mask=within(args.new12_root,p['burst_mask_filename'])
        require(mask.name==REGIMES[i]+'_NEW12_v1_T1000_burst_mask.npy','Wrong burst mask filename')
        check_hash(mask,p['burst_mask_sha256'])
        import numpy as np
        audit_mask=np.load(mask,allow_pickle=False)
        require(audit_mask.shape==(200,1000) and audit_mask.dtype==np.int8 and np.isin(audit_mask,[0,1]).all(),'Invalid burst mask')
        require(within(args.new12_root,p['filename']).stat().st_size==p['size_bytes'],'NEW12 file size mismatch')
        require(p['episode_seeds']==list(range(710000001+100000*i,710000201+100000*i)),'Wrong NEW12 seed namespace')
        streams.append(dict(regime=p['regime'],path=within(args.new12_root,p['filename']),sha256=p['sha256'],episode_seeds=p['episode_seeds'],episode_hashes=p['episode_hashes']))
    # Recheck content disjointness from immutable original pools; never generate streams here.
    import numpy as np
    old_hashes=set()
    for r in csv.DictReader(Path(args.manifest).open()):
        if r['artifact_type']=='pool':
            old=np.load(within(args.artifacts,r['transfer_relative_path']),allow_pickle=False)
            old_hashes.update(hashlib.sha256(row.astype('<i4',copy=False).tobytes()).hexdigest() for row in old)
    seen=set()
    for stream in streams:
        _,hashes=load_pool(stream['path'],stream['sha256'],stream['episode_hashes'])
        require(hashlib.sha256(np.load(stream['path'],allow_pickle=False).astype('<i4',copy=False).tobytes()).hexdigest()==next(p['payload_sha256'] for p in pools if p['regime']==stream['regime']), 'NEW12 payload hash mismatch')
        require(len(set(hashes))==200 and not (set(hashes)&(seen|old_hashes)), 'NEW12 episode content collision')
        seen.update(hashes)
    return streams,digest(mf)

def verify_g3_evidence(g3, args):
    """Approval must reference actual complete OLD-only outputs for every method/C."""
    evidence = g3.get('old_receipts', [])
    require(len(evidence)==10, 'G3 needs ten OLD validation receipts')
    covered=set()
    jobs=load_jobs(args.jobs)
    for item in evidence:
        path=within(args.work_root,item['path'])
        receipt=read_json(path,item['sha256'])
        job=jobs[receipt['job_id']]
        require(receipt.get('status')=='complete' and receipt.get('dataset_id')=='OLD12_VALIDATION','G3 requires completed OLD-only evidence')
        for key in ['runtime_lock_sha256','adapter_sha256','job_matrix_sha256']:
            require(receipt.get(key)==g3[key], 'Stale G3 evidence')
        for filename in ['episodes.csv','result.json','parity.json']:
            check_hash(within(path.parent,filename),receipt['files'][filename])
        parity=read_json(path.parent/'parity.json')
        require(parity['status']=='PASS', 'Failed OLD parity')
        if job['method']!='BF-T0.5':
            require(parity['original_evaluator_called_directly'] and parity['current_runtime_exact_comparisons']>=200,'Missing original evaluator parity')
        covered.add((job['method'],int(job['C'])))
    require(covered=={(m,c) for m in MODULES for c in [800,1200]}, 'Missing method/capacity evidence')
    unit=read_json(within(args.work_root,g3['unit_test_receipt']['path']),g3['unit_test_receipt']['sha256'])
    require(unit.get('status')=='PASS' and unit.get('adapter_sha256')==g3['adapter_sha256'], 'Missing unit test evidence')
    check_hash(Path(__file__).with_name('test_adapter.py'),unit['test_source_sha256'])


def output_bundle(args,job,cfg,cp,cp_sha,info_sha,module,streams,dataset,manifest_sha):
    episode_rows=[];summaries={};means=[];raw_by_regime={};diagnostics=[]
    for stream in streams:
        pool,hashes=load_pool(stream['path'],stream['sha256'],stream.get('episode_hashes'))
        result=evaluate_one(module,job['method'],cfg,cp,pool,stream['regime'])
        check_hash(stream['path'],stream['sha256'])
        raw_by_regime[stream['regime']]=result
        summaries[stream['regime']]=result['summary'];means.append(result['summary']['eval_money']['mean'])
        for i,metric in enumerate(result['raw_results']):
            row=dict(schema_version=1,dataset_id=dataset,job_id=job['job_id'],family=job['family'],method=job['method'],C=int(job['C']),k=24,F=3,T=1000,training_seed=None if job['method']=='BF-T0.5' else int(job['seed']),condition_mode=job['condition_mode'],cohort='rule' if job['method']=='BF-T0.5' else ('stage2_new_current_runtime' if job['checkpoint_source'].startswith('receipt:') else 'historical_recovered'),regime=stream['regime'],episode_index=i,episode_seed=stream['episode_seeds'][i],episode_sha256=hashes[i],pool_sha256=stream['sha256'],checkpoint_sha256=cp_sha)
            row.update({k:v for k,v in metric.items() if not k.startswith('gate_')})
            gates={k:v for k,v in metric.items() if k.startswith('gate_')}
            if gates:diagnostics.append(dict(regime=stream['regime'],episode_index=i,**gates))
            row['money']=metric['eval_money'];episode_rows.append(row)
    require(len(episode_rows)==200*len(streams),'Incomplete episode coverage')
    require(len({(r['regime'],r['episode_index']) for r in episode_rows})==len(episode_rows),'Duplicate episode row')
    result=dict(schema_version=1,dataset_id=dataset,job=job,config=cfg,checkpoint_source=str(cp) if cp else None,checkpoint_sha256=cp_sha,run_info_sha256=info_sha,rule_spec_sha256=object_hash(RULE) if cp is None else None,pool_sha256={s['regime']:s['sha256'] for s in streams},runtime_lock_sha256=args.runtime_lock_sha256,stream_manifest_sha256=manifest_sha,adapter_sha256=digest(__file__),source_hashes=SOURCE_HASHES,artifact_manifest_sha256=digest(args.manifest),job_matrix_sha256=digest(args.jobs),git=git_state(args.repo),regime_summaries=summaries,row_count=len(episode_rows),macro12_money=statistics.mean(means) if len(streams)==12 else None,subset_mean_money=statistics.mean(means),validation_status='PASS')
    csvbuf=io.StringIO();fields=list(episode_rows[0]);require(all(list(r)==fields for r in episode_rows),'Inconsistent output schema')
    writer=csv.DictWriter(csvbuf,fieldnames=fields);writer.writeheader();writer.writerows(episode_rows)
    files={'episodes.csv':csvbuf.getvalue(),'result.json':json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'}
    if diagnostics:files['diagnostics.json']=json.dumps(diagnostics,sort_keys=True,indent=2,allow_nan=False)+'\n'
    return files,raw_by_regime

def archive_compare(records,args,job,raw):
    if job['method'] not in ['JA-PPO','IFAC']:
        return {'status':'NOT_A_GATE','note':'SC historical drift accepted by PI; no historical delta investigation. Zero/BF use current-runtime parity/spec tests.'}
    r=pick(records,'result_metadata',job['method'],job['C'],job['seed'])
    archive=read_json(within(args.artifacts,r['transfer_relative_path']))
    deltas={}
    for reg,got in raw.items():
        expected=archive['test_results'][reg]
        require(expected['num_episodes']==got['num_episodes'],'Historical coverage mismatch')
        for metric,stats in expected['summary'].items():
            require(isinstance(stats,dict), 'Unexpected historical summary schema')
            for name,value in stats.items():
                if name=='values':continue
                actual=got['summary'][metric][name]
                if isinstance(value,(int,float)):
                    delta=abs(actual-value);deltas[f'{reg}.{metric}.{name}']=delta
                    require(delta==0,'JA/IF bit-exact historical parity mismatch')
                else:require(actual==value,'Historical summary metadata mismatch')
    return {'status':'PASS','tolerance':0,'max_abs_difference':max(deltas.values(),default=0),'all_numeric_bit_equal':all(d==0 for d in deltas.values())}

def execute(args, old=False):
    jobs=load_jobs(args.jobs);require(args.job_id in jobs,'Unknown job')
    job=jobs[args.job_id];require(job['job_type']=='evaluation','This adapter never trains')
    if old:require(not job['checkpoint_source'].startswith('receipt:'),'OLD validation uses recovered checkpoints only')
    verify_sources(args.repo);lock=verify_runtime(args)
    records=artifact_inventory(args.manifest,args.artifacts)
    for k in ['XDG_CACHE_HOME','MPLCONFIGDIR']:
        p=Path(os.environ.get(k,'/')).resolve()
        require(not p.is_relative_to(Path(args.repo).resolve()) and not p.is_relative_to(Path(args.artifacts).resolve()),'Cache must be outside historical/repository roots')
    if old:
        destination=within(args.work_root,'old_validation/'+job['job_id'])
    else:
        destination=remap(job['output_path'],'/data/sijia/aamas2027_stage2',args.work_root)
        require(destination==within(args.work_root,'evaluations/'+job['job_id']), 'Unexpected job output path')
    safe_output(destination,[args.repo,args.artifacts]+([] if old else [args.new12_root]))
    if old:
        streams=old_streams(args,records,args.regimes);stream_hash=digest(args.manifest);dataset='OLD12_VALIDATION'
    else:
        streams,stream_hash=verify_new_release(args);dataset='NEW12-v1'
    cfg,cp,cp_sha,info_sha=resolve_model(args,job,records)
    module=import_original(args.repo,job['method'])
    start=time.time();files,raw=output_bundle(args,job,cfg,cp,cp_sha,info_sha,module,streams,dataset,stream_hash)
    result_doc=json.loads(files['result.json'])
    result_doc.update(runtime=lock['snapshot'], completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), elapsed_seconds=time.time()-start, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)))
    result_doc['episodes_sha256']=hashlib.sha256(files['episodes.csv'].encode()).hexdigest()
    files['result.json']=json.dumps(result_doc,sort_keys=True,indent=2,allow_nan=False)+'\n'
    parity=None
    if old:
        comparisons=0
        if job['method']!='BF-T0.5':
            for stream in streams:
                pool,_=load_pool(stream['path'],stream['sha256'])
                direct=learned_evaluation(module,job['method'],cfg,cp,pool,stream['regime'])
                validate_result(direct)
                # Exact current-runtime per-episode and summary parity. Never relax for SC.
                require(canonical(direct['raw_results'])==canonical(raw[stream['regime']]['raw_results']),'Current-runtime episode parity failed')
                require(canonical(direct['summary'])==canonical(raw[stream['regime']]['summary']),'Current-runtime summary parity failed')
                serialized=[r for r in csv.DictReader(io.StringIO(files['episodes.csv'])) if r['regime']==stream['regime']]
                require(len(serialized)==200, 'Serialized episode coverage mismatch')
                for emitted, original in zip(serialized,direct['raw_results']):
                    require(all(float(emitted[k])==v for k,v in original.items() if not k.startswith('gate_')), 'CSV normalization changes original metric')
                    require(float(emitted['money'])==original['eval_money'], 'CSV Money differs')
                comparisons+=len(direct['raw_results'])
        parity={'status':'PASS','current_runtime_exact_comparisons':comparisons,'original_evaluator_called_directly':job['method']!='BF-T0.5','historical':archive_compare(records,args,job,raw),'g3_approved':False}
        files['parity.json']=json.dumps(parity,sort_keys=True,indent=2)+'\n'
    # Recheck every immutable artifact/source/runtime before publishing success.
    verify_sources(args.repo);artifact_inventory(args.manifest,args.artifacts);verify_runtime(args)
    if cp:check_hash(cp,cp_sha)
    if not old:
        _, final_manifest_sha=verify_new_release(args)
        require(final_manifest_sha==stream_hash,'Release changed during evaluation')
    for stream in streams:check_hash(stream['path'],stream['sha256'])
    receipt={'schema_version':1,'status':'complete','job_id':job['job_id'],'dataset_id':dataset,'runtime_lock_sha256':args.runtime_lock_sha256,'adapter_sha256':digest(__file__),'source_hashes':SOURCE_HASHES,'job_matrix_sha256':digest(args.jobs),'stream_manifest_sha256':stream_hash,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),'elapsed_seconds':time.time()-start,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},'g3_approved':False if old else True}
    files['receipt.json']=json.dumps(receipt,sort_keys=True,indent=2)+'\n'
    publish(destination,files)
    print(json.dumps({'status':'complete','output':str(destination),'dataset':dataset}))

def g3_candidate(args):
    verify_runtime(args)
    jobs=load_jobs(args.jobs)
    unit=Path(args.unit_test_receipt).resolve()
    root=Path(args.work_root).resolve()
    require(unit.is_relative_to(root), 'Unit test receipt must be inside work root')
    evidence=[]
    for job in jobs.values():
        if job['job_type']=='evaluation' and job['seed'] in ['123','NA']:
            rel='old_validation/'+job['job_id']+'/receipt.json'
            path=within(root,rel)
            require(path.is_file(),'Missing OLD acceptance job: '+job['job_id'])
            evidence.append(dict(path=rel,sha256=digest(path)))
    doc=dict(schema_version=1,gate='G3_ADAPTER_ACCEPTED',approved=False,
        runtime_lock_sha256=args.runtime_lock_sha256,adapter_sha256=digest(__file__),
        job_matrix_sha256=digest(args.jobs),validated_methods=list(MODULES),capacities=[800,1200],
        old_receipts=evidence,unit_test_receipt=dict(path=str(unit.relative_to(root)),sha256=digest(unit)))
    verify_g3_evidence(doc,args)
    dst=safe_output(root/'g3_candidate',[args.repo,args.artifacts])
    publish(dst,{'g3_receipt.json':json.dumps(doc,sort_keys=True,indent=2)+'\n'})
    print('Complete OLD evidence; pending PI/authorized acceptance:',dst/'g3_receipt.json')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['capture-runtime','old-parity','g3-candidate','evaluate'])
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--jobs',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--stage1-report',type=Path)
    p.add_argument('--runtime-lock',type=Path)
    p.add_argument('--runtime-lock-sha256')
    p.add_argument('--job-id')
    p.add_argument('--unit-test-receipt',type=Path)
    p.add_argument('--regimes',nargs='+',choices=REGIMES,default=['US'])
    p.add_argument('--allow-new12',action='store_true')
    p.add_argument('--new12-root',type=Path)
    p.add_argument('--freeze-receipt',type=Path)
    p.add_argument('--freeze-receipt-sha256')
    p.add_argument('--g3-receipt',type=Path)
    p.add_argument('--g3-receipt-sha256')
    return p

def main():
    args=parser().parse_args()
    base=args.repo/'research/aamas2027'
    args.jobs=args.jobs or base/'STAGE2_JOB_MATRIX.csv'
    args.manifest=args.manifest or base/'artifact_handoff/AAMAS_ARTIFACT_MANIFEST.csv'
    args.plan=args.plan or base/'STAGE2_EXECUTION_PLAN.md'
    require(len(args.regimes)==len(set(args.regimes)),'Duplicate regimes')
    if args.command=='capture-runtime':
        require(args.stage1_report is not None,'--stage1-report required')
        artifact_inventory(args.manifest,args.artifacts);load_jobs(args.jobs)
        dst=safe_output(args.work_root/'runtime_capture',[args.repo,args.artifacts])
        doc=runtime_document(args)
        publish(dst,{'runtime_lock.json':json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n'})
        print('Candidate runtime lock:',dst/'runtime_lock.json')
        print('SHA256:',digest(dst/'runtime_lock.json'))
        print('PI must confirm this captures the already approved Stage 1 runtime before use.')
    else:
        require(args.runtime_lock is not None and args.runtime_lock_sha256,'Pinned --runtime-lock and --runtime-lock-sha256 required')
        if args.command=='g3-candidate':
            require(args.unit_test_receipt is not None,'--unit-test-receipt required')
            g3_candidate(args)
            return
        if args.command=='evaluate':
            require(all([args.new12_root,args.freeze_receipt,args.freeze_receipt_sha256,args.g3_receipt,args.g3_receipt_sha256]),'NEW12 release arguments required')
        execute(args,old=args.command=='old-parity')

if __name__=='__main__':
    try:
        main()
    except (Stop,KeyError,ValueError,TypeError,FileNotFoundError,ImportError) as exc:
        print('STOP: '+str(exc),file=sys.stderr)
        sys.exit(2)
