"""OLD-only contract tests; no training, NEW stream generation, or policy replay."""
import ast
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import adapter as a

REPO=Path(os.environ.get('AAMAS_REPO',Path(__file__).resolve().parents[2]))
JOBS=Path(os.environ.get('AAMAS_JOBS',REPO/'research/aamas2027/STAGE2_JOB_MATRIX.csv'))
ARTIFACTS=Path(os.environ.get('AAMAS_ARTIFACTS','/Users/zhouzhou/Desktop/kwallet-aamas-artifacts'))


def metric():
    return dict(settled=5000.,flushes=20,drops=500,oversize_drops=300,insufficient_drops=200,accepted_count=500,total_tx_count=1000,total_requested_value=10000.,value_accept_ratio=.5,count_accept_ratio=.5,utilization=.1,eval_money=4800.,eval_money_p=1.,eval_money_tau=10.)


def result():
    import numpy as np
    rows=[metric() for _ in range(200)]
    summary={}
    for key in rows[0]:
        vals=[r[key] for r in rows]
        summary[key]={name:float(fn(vals)) for name,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max),('median',np.median)]}
        summary[key]['values']=vals
    return dict(num_episodes=200,raw_results=rows,summary=summary)


class RuleTests(unittest.TestCase):
    def env(self,balances,tx,usable=None):
        usable=set(range(len(balances))) if usable is None else set(usable)
        return types.SimpleNamespace(k=len(balances),C=10*len(balances),wallets=balances,current_tx=tx,_usable=lambda i:i in usable)
    def action(self,balances,tx,usable=None):
        env=self.env(balances,tx,usable)
        return divmod(a.bf_action(env),env.k+1)
    def test_best_fit_and_index_ties(self):
        self.assertEqual(self.action([8,6,6,2],5),(1,3))
    def test_oversize_still_flushes(self):
        self.assertEqual(self.action([10,3,4],11),(3,1))
    def test_no_fit_still_flushes(self):
        self.assertEqual(self.action([3,2],8),(2,1))
    def test_half_threshold_is_strict(self):
        self.assertEqual(self.action([5,10],7),(1,2))
    def test_settlement_excluded(self):
        self.assertEqual(self.action([2,3],1),(0,1))
    def test_frozen_wallets_excluded(self):
        self.assertEqual(self.action([1,4,9],3,[1,2]),(1,3))
    def test_all_frozen_noop(self):
        self.assertEqual(self.action([1,2],1,[]),(2,2))
    def test_flush_tie_lowest_index(self):
        self.assertEqual(self.action([2,2,8],6),(2,0))
    def test_no_state_mutation(self):
        env=self.env([8,1,3],7); before=list(env.wallets)
        a.bf_action(env);self.assertEqual(env.wallets,before)


class ContractTests(unittest.TestCase):
    def test_finite_complete_metrics(self):a.validate_result(result())
    def test_bad_metrics_stop(self):
        for key,value in [('eval_money',4801),('flushes',.5),('drops',501),('total_tx_count',999),('value_accept_ratio',float('nan')),('settled',float('inf')),('eval_money_tau',9)]:
            with self.subTest(key=key):
                r=metric();r[key]=value
                with self.assertRaises(a.Stop):a.validate_episode(r)
    def test_missing_metric_stop(self):
        r=metric();del r['settled']
        with self.assertRaises(a.Stop):a.validate_episode(r)
    def test_incomplete_rows_stop(self):
        r=result();r['raw_results'].pop()
        with self.assertRaises(a.Stop):a.validate_result(r)
    def test_inconsistent_summary_stop(self):
        r=result();r['summary']['eval_money']['mean']+=1
        with self.assertRaises(a.Stop):a.validate_result(r)
    def test_hash_mutation_stop(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact';p.write_bytes(b'original');h=a.digest(p);a.check_hash(p,h)
            p.write_bytes(b'changed')
            with self.assertRaises(a.Stop):a.check_hash(p,h)
    def test_path_escape_stop(self):
        with tempfile.TemporaryDirectory() as d:
            for rel in ['../escape','/absolute']:
                with self.assertRaises(a.Stop):a.within(d,rel)
            (Path(d)/'escape').symlink_to('/tmp')
            with self.assertRaises(a.Stop):a.within(d,'escape/file')
    def test_atomic_publish_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)/'output';a.publish(dst,{'receipt.json':'{}','episodes.csv':'complete'})
            self.assertEqual((dst/'episodes.csv').read_text(),'complete')
            with self.assertRaises(a.Stop):a.publish(dst,{'receipt.json':'overwrite'})
    def test_protected_output(self):
        with self.assertRaises(a.Stop):a.safe_output(REPO/'some-output',[REPO])
    def test_new_evaluation_not_default(self):
        with self.assertRaises(a.Stop):a.verify_new_release(types.SimpleNamespace(allow_new12=False))
    def test_training_not_exposed(self):
        self.assertNotIn('train',a.parser()._actions[1].choices)
    def test_all_learned_dispatches_use_original_evaluator(self):
        from unittest.mock import Mock
        import sys
        for method,name in [('JA-PPO','BasicPPOAgent'),('IFAC','FactorizedPPOAgent'),('SC-FAC','ConditionalFactorizedPPOAgent'),('SC-FAC-zero','ConditionalFactorizedPPOAgent')]:
            with self.subTest(method=method):
                model=Mock();agent=types.SimpleNamespace(model=model)
                ctor=Mock(return_value=agent)
                evaluator=Mock(return_value={'original':'sentinel'})
                module=types.SimpleNamespace(make_env=Mock(return_value=types.SimpleNamespace(state_size=74,base_state_size=74,k=24)),evaluate_agent_on_array=evaluator)
                setattr(module,name,ctor)
                fake_torch=types.SimpleNamespace(load=Mock(return_value={'weights':'sentinel'}))
                cfg={'attention_context':{'window_size':50}};pool=object()
                with patch.dict(sys.modules,{'torch':fake_torch}):
                    got=a.learned_evaluation(module,method,cfg,'best_model.pth',pool,'US')
                self.assertEqual(got,{'original':'sentinel'})
                evaluator.assert_called_once_with(agent,cfg,pool,'US',200,1000)
                model.load_state_dict.assert_called_once_with({'weights':'sentinel'},strict=True)
                model.eval.assert_not_called()
                fake_torch.load.assert_called_once_with('best_model.pth',map_location='cpu',weights_only=True)
    def test_output_normalization_and_diagnostics(self):
        import numpy as np
        args=types.SimpleNamespace(runtime_lock_sha256='f'*64,manifest=JOBS,jobs=JOBS,repo=REPO)
        job=a.load_jobs(JOBS)['A-EVAL-SC-C800-S123']
        original=result()
        for r in original['raw_results']:r['gate_mean']=.25
        stream=dict(path=Path('/unused'),sha256='a'*64,regime='US',episode_seeds=list(range(200)))
        with patch.object(a,'load_pool',return_value=(np.ones((200,1000),dtype=np.int32),['b'*64]*200)),patch.object(a,'evaluate_one',return_value=original),patch.object(a,'check_hash'),patch.object(a,'git_state',return_value={}):
            files,raw=a.output_bundle(args,job,{},Path('checkpoint'),'c'*64,'d'*64,None,[stream],'OLD12_VALIDATION','e'*64)
        rows=list(csv.DictReader(io.StringIO(files['episodes.csv'])))
        self.assertEqual(len(rows),200);self.assertNotIn('gate_mean',rows[0])
        self.assertEqual(float(rows[0]['money']),original['raw_results'][0]['eval_money'])
        self.assertEqual(len(json.loads(files['diagnostics.json'])),200)
        self.assertIsNone(json.loads(files['result.json'])['macro12_money'])
    def test_source_pins(self):a.verify_sources(REPO)
    def test_matrix_exact_dependencies(self):
        jobs=a.load_jobs(JOBS);self.assertEqual(len(jobs),46)
        self.assertEqual(sum(j['job_type']=='training' for j in jobs.values()),4)
    def test_matrix_bad_training_gate_stop(self):
        rows=list(csv.DictReader(JOBS.open()));rows[0]['dependencies']+=';G2_NEW12_FROZEN'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            with self.assertRaises(a.Stop):a.load_jobs(p)
    def test_runtime_drift_stop(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'runtime.json'
            p.write_text(json.dumps(dict(schema_version=1,stage1_approved=True,runtime_confirmed=True,source_hashes=a.SOURCE_HASHES,artifact_manifest_sha256='0'*64,job_matrix_sha256='0'*64,adapter_sha256='0'*64,snapshot={'torch':'old'})))
            args=types.SimpleNamespace(runtime_lock=p,runtime_lock_sha256=a.digest(p),manifest=p,jobs=p)
            with patch.object(a,'check_hash'),patch.object(a,'runtime_snapshot',return_value={'torch':'changed'}):
                with self.assertRaises(a.Stop):a.verify_runtime(args)
    def test_runtime_unconfirmed_stop(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'runtime.json';p.write_text('{"schema_version":1,"stage1_approved":true,"runtime_confirmed":false}')
            with self.assertRaises(a.Stop):a.verify_runtime(types.SimpleNamespace(runtime_lock=p,runtime_lock_sha256=a.digest(p)))
    def test_real_handoff_hashes_configs_old_pools(self):
        records=a.artifact_inventory(REPO/'research/aamas2027/artifact_handoff/AAMAS_ARTIFACT_MANIFEST.csv',ARTIFACTS)
        jobs=a.load_jobs(JOBS);args=types.SimpleNamespace(artifacts=ARTIFACTS)
        for job in jobs.values():
            if job['job_type']=='evaluation' and not job['checkpoint_source'].startswith('receipt:'):
                a.resolve_model(args,job,records)
        streams=a.old_streams(args,records,a.REGIMES)
        self.assertEqual(len(streams),12)
        for s in streams:
            pool,hashes=a.load_pool(s['path'],s['sha256'])
            self.assertEqual(len(hashes),200);self.assertFalse(pool.flags.writeable)
    def test_sc_historical_drift_not_reopened(self):
        got=a.archive_compare([],None,{'method':'SC-FAC'},None)
        self.assertEqual(got['status'],'NOT_A_GATE')
    def test_no_training_calls_or_collaborator_imports(self):
        tree=ast.parse(Path(a.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,{'train_agent','backward','generate_regime_pool'})
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,{'src.kwallet','collab'})


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    run=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=os.environ.get('AAMAS_UNIT_RECEIPT')
    if receipt:
        p=Path(receipt);a.safe_output(p,[REPO,ARTIFACTS]);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:
            json.dump(dict(status='PASS' if run.wasSuccessful() else 'FAIL',tests_run=run.testsRun,adapter_sha256=a.digest(a.__file__),test_source_sha256=a.digest(__file__)),f,indent=2)
    raise SystemExit(0 if run.wasSuccessful() else 1)
