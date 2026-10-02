"""Independent numerical checks and frozen-data guard tests; no simulations."""
import csv,io,json,math,statistics,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_stage3_analysis as a
from scipy import stats
import numpy as np

class StatisticalTests(unittest.TestCase):
    def test_df4_t_critical(self):self.assertAlmostEqual(stats.t.ppf(.975,4),2.7764451051977987,places=13)
    def test_sample_sd_se_ci(self):
        s=a.summary5([1,2,3,4,5]);self.assertEqual(s['n'],5);self.assertEqual(s['df'],4)
        self.assertAlmostEqual(s['SD'],math.sqrt(2.5));self.assertAlmostEqual(s['SE'],math.sqrt(.5))
        self.assertAlmostEqual(s['ci95_low'],3-2.7764451051977987*math.sqrt(.5))
    def test_requires_five_seeds(self):
        with self.assertRaises(a.AcceptanceError):a.summary5([1,2,3,4])
    def test_independent_scipy_paired_test(self):
        left=np.array([10,23,36,40,55]);right=np.array([9,21,30,39,50]);s=a.contrast_stats((left-right).tolist());r=stats.ttest_rel(left,right)
        self.assertAlmostEqual(s['t_statistic'],r.statistic);self.assertAlmostEqual(s['raw_p'],r.pvalue)
    def test_df4_closed_form_tail(self):
        # Independent analytic df=4 survival identity: p=1-1.5u+0.5u^3, u=|t|/sqrt(t²+4).
        for tv in [0,.2,1,2.7764451051977987,5,20]:
            u=abs(tv)/math.sqrt(tv*tv+4);p=1-1.5*u+.5*u**3
            self.assertAlmostEqual(2*stats.t.sf(abs(tv),4),p,places=13)
    def test_bf_one_sample_constant_not_independent_seeds(self):
        xs=[100,110,120,130,150];bf=160;s=a.contrast_stats([x-bf for x in xs]);r=stats.ttest_1samp(xs,bf)
        self.assertAlmostEqual(s['raw_p'],r.pvalue);self.assertAlmostEqual(s['SD'],statistics.stdev(xs))
    def test_holm_across_full_family(self):
        rows=[dict(contrast_id=str(i),raw_p=p) for i,p in enumerate([.04,.001,.02,.2,.5,.06])];a.holm(rows)
        self.assertEqual([round(r['holm_p'],8) for r in rows],[.16,.006,.1,.4,.5,.18])
    def test_holm_ties(self):
        rows=[dict(contrast_id='b',raw_p=.03),dict(contrast_id='a',raw_p=.03)];a.holm(rows)
        self.assertEqual([r['holm_p'] for r in rows],[.06,.06])
    def test_degenerate_not_significant(self):
        for xs in [[0]*5,[2]*5]:
            s=a.contrast_stats(xs);self.assertIsNone(s['raw_p']);self.assertIsNone(s['d_z']);self.assertEqual(s['ci95_low'],s['ci95_high'])
    def test_undefined_preserves_holm_family_size(self):
        rows=[dict(contrast_id='a',raw_p=None),dict(contrast_id='b',raw_p=.03)];a.holm(rows)
        self.assertIsNone(rows[0]['holm_p']);self.assertEqual(rows[1]['holm_p'],.06)
    def test_wrong_raw_hash_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'wrong.tar.gz';raw.write_bytes(b'wrong')
            with self.assertRaises(a.AcceptanceError):a.extract_verified(raw,root/'out')
            self.assertFalse((root/'out').exists())

class FrozenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=Path(a.__file__).resolve().parents[1];cls.doc=a.jread(cls.out/'statistics.json');cls.t=cls.doc['tables']
    def test_complete_units(self):
        self.assertEqual(len(self.t['stage2b_job_summary']),42);self.assertEqual(len(self.t['seed_level_scores']),42);self.assertEqual(len(self.t['regime_level_scores']),504)
        self.assertEqual(sum(r['rows'] for r in self.t['stage2b_job_summary']),100800)
    def test_ledger_reconciliation(self):
        self.assertEqual(set(self.doc['acceptance']['preexisting_jobs']),a.MISSING_LEDGER);self.assertEqual(self.doc['acceptance']['ledger_completed'],38)
    def test_bf_no_fake_uncertainty(self):
        bf=[r for r in self.t['method_capacity_summary'] if r['method']=='BF-T0.5'];self.assertEqual(len(bf),2)
        for r in bf:
            self.assertEqual(r['n_training_seeds'],0)
            for k in ['SD','SE','ci95_low','ci95_high','df']:self.assertIsNone(r[k])
    def test_all_contrasts_against_scipy(self):
        for name in ['family_A_contrasts','family_B_conditioning_ablation','family_C_vs_bft05']:
            for r in self.t[name]:
                ps=[p for p in self.t['paired_seed_differences'] if p['contrast_id']==r['contrast_id']]
                direct=stats.ttest_1samp([p['delta'] for p in ps],0) if r['family']=='C' else stats.ttest_rel([p['left_score'] for p in ps],[p['right_score'] for p in ps])
                self.assertAlmostEqual(r['raw_p'],direct.pvalue,places=13)
                self.assertAlmostEqual(r['t_statistic'],direct.statistic,places=12)
    def test_seed_pairing_not_row_order(self):
        rows=self.t['seed_level_scores'];lookup={(r['method'],r['C'],r['training_seed']):r['macro12_money'] for r in reversed(rows)}
        for p in self.t['paired_seed_differences']:
            s=p['training_seed'];rightseed=None if p['family']=='C' else s
            self.assertEqual(p['delta'],lookup[p['left_method'],p['C'],s]-lookup[p['right_method'],p['C'],rightseed])
    def test_fixed_families(self):
        for f,n in [('A',6),('B',2),('C',2)]:self.assertEqual(len([r for r in self.t['holm_results'] if r['family']==f]),n)
    def test_regime_not_inferential_unit(self):
        for r in self.t['method_capacity_summary']:
            if r['method']!='BF-T0.5':self.assertEqual((r['n'],r['df']),(5,4))
        for r in self.t['regime_level_summary']:self.assertNotIn('raw_p',r)
    def test_raw_mutation_guard(self):
        rows=a.csvread(self.out/'raw_file_inventory.csv');p=self.out/'work/raw'/a.BUNDLE/rows[0]['path'];self.assertEqual(a.sha(p),rows[0]['sha256'])
    def test_independent_unequal_episode_weighting_fixture(self):
        # Illustrates why the unit hierarchy is explicit even with 200 episodes everywhere in this freeze.
        regimes=[[0],[10]*9];proper=statistics.mean([statistics.mean(r) for r in regimes]);pooled=statistics.mean(sum(regimes,[]))
        self.assertEqual(proper,5);self.assertEqual(pooled,9)
    def test_observed_mean_hierarchy(self):
        for r in self.t['seed_level_scores']:self.assertAlmostEqual(r['macro12_money'],statistics.mean(r['money_'+g] for g in a.REGIMES),places=11)
    def test_accounting_decomposition(self):
        for r in self.t['mechanism_decomposition']:self.assertAlmostEqual(r['money_delta'],r['settled_delta']-10*r['flushes_delta'],places=9)
    def test_byte_identical_regeneration_from_frozen_tarball(self):
        raw=self.out.parent/'stage2/raw'/f'{a.BUNDLE}.tar.gz'
        targets=[name+'.csv' for name in self.t]
        targets+=['statistics.json','raw_acceptance.json','raw_file_inventory.csv','STAGE3_RAW_ACCEPTANCE.md','STAGE3_ANALYSIS_REPORT.md']
        targets+=[str(p.relative_to(self.out)) for p in sorted((self.out/'figures').iterdir()) if p.is_file()]
        with tempfile.TemporaryDirectory(prefix='kwallet-stage3-reproduce-') as td:
            result=subprocess.run([sys.executable,str(Path(a.__file__).resolve()),'--raw',str(raw),'--out',td],text=True,capture_output=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            for name in targets:self.assertEqual(a.sha(self.out/name),a.sha(Path(td)/name),name+' did not reproduce byte-identically')
        type(self).regeneration_hashes={name:a.sha(self.out/name) for name in sorted(targets)}
    def test_all_plots_present(self):
        for name in ['figure1_macro12_money','figure2_structural_seed_contrasts','figure3_conditioning_ablation','figure4_regime_contrasts']:
            for ext in ['pdf','svg','png']:self.assertGreater((self.out/'figures'/f'{name}.{ext}').stat().st_size,1000)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(a.__file__).resolve().parents[1]
    a.jwrite(out/'verification.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,analysis_code_sha256=a.sha(a.__file__),test_code_sha256=a.sha(__file__),raw_sha256=a.RAW_SHA,byte_identical_regeneration=getattr(FrozenIntegrationTests,'regeneration_hashes',{})))
    raise SystemExit(0 if result.wasSuccessful() else 1)
