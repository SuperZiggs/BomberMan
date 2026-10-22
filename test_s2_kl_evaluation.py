"""Locked sampling, raw retention metrics, clustered inference and gate boundaries."""
from copy import deepcopy
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from test_s2_kl_support import (ROOT, S2_REL, S2_SHA, SAMPLE_REL, SAMPLE_SHA, assert_same,
    digest, function, gate_fixture, model_descriptor, read_json, register_case, rng_snapshot, sha, write_json)

CONDITIONS = ('DRIFT_REPRODUCED', 'DECISION_RETAINED', 'PROBABILITY_RETAINED',
              'ANCHOR_BENEFIT', 'RB_ACTIVE_RETAINED', 'RB_SUICIDE_RETAINED')


def condition(result, name):
    value = result['conditions'][name]
    return value['pass'] if isinstance(value, dict) else value


class SamplingTests(unittest.TestCase):
    def test_targets_accept_all_legal_finite_maxima_and_reject_noninformative_ties(self):
        target = function(self, 'decision_targets')
        scores = np.array([[5., 5., 4., 100., -np.inf, np.nan],
                           [3., 3., 3., 3., 3., 3.],
                           [1., 2., np.inf, 8., -np.inf, np.nan],
                           [0., 1., 2., 3., 4., 5.]], dtype=np.float32)
        masks = np.array([[1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=bool)
        targets, valid = target(scores, masks)
        np.testing.assert_array_equal(np.asarray(valid), [True, False, True, False])
        np.testing.assert_allclose(np.asarray(targets)[0], [.5, .5, 0, 0, 0, 0])
        np.testing.assert_allclose(np.asarray(targets)[2], [0, 1, 0, 0, 0, 0])
        self.assertTrue(np.isfinite(np.asarray(targets)).all())
        self.assertTrue((np.asarray(targets)[~np.asarray(valid)] == 0).all())
        with self.assertRaises((ValueError, TypeError)): target(scores[:, :5], masks)

    def test_real_manifest_loads_exact_locked_rows_and_excludes_original_heldout(self):
        load = function(self, 'load_samples')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-samples-') as tmp:
            inputs, output, reg = register_case(self, tmp)
            loaded = load(reg); manifest = read_json(inputs / SAMPLE_REL)
            self.assertEqual(set(loaded), {'decision', 'anchor'})
            for kind, expected_boards in (('decision', 722), ('anchor', 667)):
                data = loaded[kind]
                self.assertEqual(np.asarray(data['obs']).shape, (1024, 13, 9, 9))
                self.assertEqual(np.asarray(data['mask']).shape, (1024, 6))
                self.assertEqual(len(data['ids']), 1024); self.assertEqual(len(set(data['ids'])), 1024)
                self.assertEqual(len(set(map(int, data['boards']))), expected_boards)
                self.assertFalse(set(map(int, data['boards'])) & set(manifest['excluded_heldout_board_ids']))
                self.assertTrue(np.isfinite(np.asarray(data['obs'])).all())
                for index, selected in enumerate(manifest['samples'][kind]['selected_rows']):
                    relative = selected['file'].replace('\\', '/'); row = selected['row']
                    self.assertIn(relative, str(data['ids'][index]).replace('\\', '/'))
                    self.assertIn(str(row), str(data['ids'][index]))
                    with np.load(inputs / relative, allow_pickle=False) as shard:
                        np.testing.assert_array_equal(np.asarray(data['obs'])[index], shard[kind + '_obs'][row])
                        np.testing.assert_array_equal(np.asarray(data['mask'])[index], shard[kind + '_mask'][row])
                        self.assertEqual(int(data['boards'][index]), int(shard[kind + '_seed'][row]))
            self.assertEqual(np.asarray(loaded['decision']['scores']).shape, (1024, 6))
            self.assertEqual(np.asarray(loaded['decision']['targets']).shape, (1024, 6))
            self.assertEqual(sha(output / 'offline_sample_manifest.json'), SAMPLE_SHA)

    def test_changed_shard_bad_shape_count_row_or_heldout_cannot_trigger_resampling(self):
        load = function(self, 'load_samples')
        for change in ('shard', 'shape', 'count', 'row', 'heldout'):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-sample-drift-') as tmp:
                inputs, output, reg = register_case(self, tmp)
                manifest = read_json(inputs / SAMPLE_REL)
                relative = manifest['samples']['decision']['selected_rows'][0]['file']
                path = inputs / relative
                if change in ('shard', 'shape', 'heldout'):
                    with np.load(path, allow_pickle=False) as shard: data = {k: shard[k].copy() for k in shard.files}
                    if change == 'shard': data['decision_obs'][0, 0, 0, 0] += 1.
                    if change == 'shape': data['decision_obs'] = data['decision_obs'][:, :12]
                    if change == 'heldout':
                        row = manifest['samples']['decision']['selected_rows'][0]['row']
                        data['decision_seed'][row] = manifest['excluded_heldout_board_ids'][0]
                    np.savez_compressed(path, **data)
                else:
                    if change == 'count': manifest['samples']['decision']['selected_rows'].pop()
                    else: manifest['samples']['decision']['selected_rows'][0]['row'] = 10**9
                    write_json(inputs / SAMPLE_REL, manifest); write_json(output / 'offline_sample_manifest.json', manifest)
                with self.assertRaises(ValueError): load(reg)


class DiagnosticsTests(unittest.TestCase):
    def test_baseline_raw_rows_match_independent_forward_and_cached_summary(self):
        diagnose = function(self, 'diagnose_model'); load = function(self, 'load_samples')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-diag-') as tmp:
            inputs, _, reg = register_case(self, tmp)
            model = model_descriptor(checkpoint=inputs / S2_REL)
            before = rng_snapshot(); checkpoint = sha(inputs / S2_REL)
            result = diagnose(model, reg)
            assert_same(self, before, rng_snapshot(), 'diagnostic RNG')
            self.assertEqual(sha(inputs / S2_REL), checkpoint)
            self.assertEqual(result['format'], 's2-kl-diagnostics-v1')
            self.assertEqual(result['model_sha256'], S2_SHA)
            self.assertEqual(result['registration_digest'], reg['registration_digest'])
            self.assertEqual(result['sample_manifest_sha256'], SAMPLE_SHA)
            self.assertEqual(result['diagnostic_digest'], digest(result, 'diagnostic_digest'))
            data = load(reg)
            from train_actor_critic import ActorCriticNet
            with torch.random.fork_rng(devices=[]):
                net = ActorCriticNet(in_channels=13, board_size=9).eval()
                net.load_state_dict(torch.load(inputs / S2_REL, map_location='cpu', weights_only=True))
            for kind in ('decision', 'anchor'):
                rows = result['raw'][kind]
                self.assertEqual(len(rows), 1024)
                self.assertEqual([r['id'] for r in rows], list(data[kind]['ids']))
                self.assertEqual([r['board'] for r in rows], list(map(int, data[kind]['boards'])))
                with torch.no_grad():
                    logits, values = net(torch.as_tensor(np.asarray(data[kind]['obs'])))
                    lp = F.log_softmax(logits.masked_fill(~torch.as_tensor(np.asarray(data[kind]['mask']), dtype=torch.bool), -1e9), -1)
                    probs = lp.exp().numpy()
                for index, row in enumerate(rows):
                    for key in ('kl_s2_to_model', 'bomb_probability', 'value'):
                        self.assertTrue(math.isfinite(row[key]))
                    self.assertAlmostEqual(row['kl_s2_to_model'], 0., places=6)
                    self.assertAlmostEqual(row['bomb_probability'], float(probs[index, 5]), places=6)
                    self.assertAlmostEqual(row['value'], float(values.reshape(-1)[index]), places=5)
                    if kind == 'decision':
                        targets = np.asarray(data[kind]['targets'])[index]
                        good = targets > 0
                        self.assertEqual(row['hit'], int(good[probs[index].argmax()]))
                        self.assertAlmostEqual(row['mass'], float(probs[index, good].sum()), places=6)
                        self.assertAlmostEqual(row['ce'], float(-(targets * lp[index].numpy()).sum()), places=5)
                    else:
                        self.assertFalse({'hit', 'mass', 'ce'} & set(row))
            decision = result['raw']['decision']
            self.assertAlmostEqual(result['summary']['R'], 100*np.mean([r['hit'] for r in decision]))
            self.assertAlmostEqual(result['summary']['Q'], 100*np.mean([r['mass'] for r in decision]))


class ClusterTests(unittest.TestCase):
    def test_unequal_clusters_preserve_state_micro_weights_and_paired_board_resampling(self):
        compare = function(self, 'cluster_comparison')
        left, right = [], []
        sizes, differences = [1, 2, 4, 8], [0., .2, .8, 1.]
        for board, (size, delta) in enumerate(zip(sizes, differences)):
            for index in range(size):
                left.append(dict(id=f'{board}:{index}', board=board, value=delta))
                right.append(dict(id=f'{board}:{index}', board=board, value=0.))
        result = compare(left, right)
        self.assertAlmostEqual(result['difference_pp'], 100*sum(n*d for n,d in zip(sizes,differences))/sum(sizes))
        self.assertEqual((result['board_clusters'], result['samples'], result['resamples'], result['bootstrap_seed']),
                         (4, 15, 10000, 20261023))
        distribution = []
        for chosen in itertools.product(range(4), repeat=4):
            distribution.append(100*sum(sizes[i]*differences[i] for i in chosen)/sum(sizes[i] for i in chosen))
        # All 256 equally probable whole-board samples provide an RNG-independent oracle.
        for actual, lo, hi in zip(result['ci95_pp'], (1.0, 96.0), (4.0, 99.0)):
            self.assertGreaterEqual(actual, np.percentile(distribution, lo, method='linear') - 1e-12)
            self.assertLessEqual(actual, np.percentile(distribution, hi, method='linear') + 1e-12)
        self.assertEqual(result, compare(left, list(reversed(right))))
        self.assertEqual(result, compare(left, right))
        custom = compare(left, right, resamples=201, seed=123)
        self.assertEqual((custom['resamples'], custom['bootstrap_seed']), (201,123))

    def test_invalid_pairing_duplicates_board_identity_nonfinite_and_nonprobability_fail(self):
        compare = function(self, 'cluster_comparison')
        good = [dict(id='a', board=1, value=.2), dict(id='b', board=2, value=.5)]
        bad = [[], [good[0]], [good[0], good[0]],
               [dict(good[0], board=8), good[1]], [dict(good[0], id='z'), good[1]]]
        bad += [[dict(good[0], value=value),good[1]] for value in (-.01, 1.01, float('nan'), float('inf'))]
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises((ValueError,TypeError)): compare(good, rows)

    def test_binary_rb_corner_rows_are_paired_in_the_same_board_cluster(self):
        compare = function(self, 'cluster_comparison')
        left = [dict(id=f'{b}:{c}',board=b,value=int(b==0 or c==0)) for b in range(4) for c in (0,1)]
        right = [dict(r,value=0) for r in left]
        result = compare(left,right)
        self.assertEqual(result['board_clusters'],4); self.assertEqual(result['samples'],8)
        self.assertEqual(result['difference_pp'],62.5)


class GateTests(unittest.TestCase):
    def gate(self, values=None, **kwargs):
        return function(self, 'decide_gate')(*(values or gate_fixture()), complete=kwargs.pop('complete',True), **kwargs)

    def test_exact_thresholds_pass_but_one_ulp_outside_each_threshold_fails(self):
        self.assertEqual(self.gate()['status'],'RETENTION_PILOT_PASS')
        cases = [
            ('DECISION_RETAINED',2,'R',math.nextafter(90.,-math.inf),'RETENTION_NOT_PRESERVED'),
            ('PROBABILITY_RETAINED',2,'Q',math.nextafter(85.,-math.inf),'RETENTION_NOT_PRESERVED'),
            ('RB_ACTIVE_RETAINED',2,'AW',math.nextafter(78.,-math.inf),'RB_RETENTION_FAIL'),
            ('RB_SUICIDE_RETAINED',2,'SU',math.nextafter(4.,math.inf),'RB_RETENTION_FAIL'),
        ]
        for name,index,key,value,status in cases:
            values = gate_fixture(); values[index][key] = value
            with self.subTest(condition=name):
                result = self.gate(values)
                self.assertEqual(result['status'],status); self.assertFalse(condition(result,name))
        values = gate_fixture(); values[1]['R'] = 85.
        self.assertTrue(condition(self.gate(values),'DRIFT_REPRODUCED'))
        values[1]['R'] = math.nextafter(85.,math.inf)
        self.assertEqual(self.gate(values)['status'],'INCONCLUSIVE_NO_DRIFT')
        for lower in (0., math.nextafter(0.,-math.inf)):
            values = gate_fixture(); values[3]['anchor_minus_control']['R']['ci95_pp'][0] = lower
            self.assertEqual(self.gate(values)['status'],'NO_CLEAR_ANCHOR_BENEFIT')
        values = gate_fixture(); values[0]['R']=15.; values[1]['R']=0.; values[2]['R']=10.
        values[3]['anchor_minus_control']['R']['difference_pp']=10.
        self.assertTrue(condition(self.gate(values),'ANCHOR_BENEFIT'))
        values[2]['R']=math.nextafter(10.,-math.inf)
        values[3]['anchor_minus_control']['R']['difference_pp']=values[2]['R']
        self.assertFalse(condition(self.gate(values),'ANCHOR_BENEFIT'))

    def test_status_priority_and_all_six_conditions_are_always_emitted(self):
        values = gate_fixture(); values[1]['R']=94.; values[2].update(R=0.,Q=0.,AW=0.,SU=100.)
        for complete,expected in ((False,'INCOMPLETE'),(True,'INCONCLUSIVE_NO_DRIFT')):
            result=self.gate(values,complete=complete)
            self.assertEqual(result['status'],expected)
            self.assertEqual(set(result['conditions']),set(CONDITIONS))
        values[1]['R']=70.
        self.assertEqual(self.gate(values)['status'],'RB_RETENTION_FAIL')
        values[2].update(AW=80.,SU=3.)
        self.assertEqual(self.gate(values)['status'],'RETENTION_NOT_PRESERVED')
        values[2].update(R=90.,Q=85.)
        values[3]['anchor_minus_control']['R']['ci95_pp'][0]=0.
        self.assertEqual(self.gate(values)['status'],'NO_CLEAR_ANCHOR_BENEFIT')

    def test_smoke_is_never_scientific_pass_and_nonfinite_inputs_are_invalid(self):
        self.assertEqual(self.gate(smoke=True)['status'],'SMOKE_ONLY')
        for value in (float('nan'),float('inf')):
            values=gate_fixture();values[2]['R']=value
            with self.assertRaises((ValueError,TypeError)):self.gate(values)


class ReportTests(unittest.TestCase):
    def test_full_report_uses_only_500_even_when_intermediate_is_perfect_and_refuses_incomplete_raw(self):
        build_report=function(self,'build_report');load=function(self,'load_samples')
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-report-') as tmp:
            _,output,reg=register_case(self,tmp,smoke=False)
            samples=load(reg)
            models=[model_descriptor(checkpoint=Path(reg['input_root'])/S2_REL)]
            weights=torch.load(ROOT/S2_REL,map_location='cpu',weights_only=True)
            for index,(arm,ep) in enumerate(itertools.product(('control','anchor'),(100,250,500)),1):
                candidate={key:value.clone() for key,value in weights.items()}
                candidate['conv.0.weight'].view(-1)[0]+=index*.0001
                checkpoint=Path(tmp)/'synthetic-checkpoints'/f'{arm}-{ep}.pt'
                checkpoint.parent.mkdir(exist_ok=True);torch.save(candidate,checkpoint)
                models.append(model_descriptor(f'{arm}-{ep}',arm,ep,checkpoint,sha(checkpoint)))
            diagnostics={};rb_rows=[]
            for model in models:
                hits=973 if model['arm'] is None else 768 if model['arm']=='control' else 717 if model['episode']==500 else 1024
                mass=.4 if model['arm']=='anchor' and model['episode']==500 else .9
                raw={}
                for kind in ('decision','anchor'):
                    raw[kind]=[]
                    for index,(row_id,board) in enumerate(zip(samples[kind]['ids'],samples[kind]['boards'])):
                        row=dict(id=row_id,board=int(board),kl_s2_to_model=0. if model['arm'] is None else .1,
                                 bomb_probability=.2,value=0.)
                        if kind=='decision':row.update(hit=int(index<hits),mass=mass,ce=1.)
                        raw[kind].append(row)
                diagnostic=dict(format='s2-kl-diagnostics-v1',model_sha256=model['sha256'],
                    registration_digest=reg['registration_digest'],sample_manifest_sha256=SAMPLE_SHA,
                    raw=raw,summary=dict(R=100*hits/1024,Q=100*mass))
                diagnostic['diagnostic_digest']=digest(diagnostic)
                diagnostics[model['sha256']]=diagnostic
                for index in range(200):
                    win=int(index<160)
                    rb_rows.append(dict(learner_sha256=model['sha256'],opponent_id='RB',
                        opponent_sha256=reg['sources']['rule_based_bot.py'],board=720000+index//2,corner=index%2,
                        win=win,active_win=win,draw=1-win,passive_win=0,sudden_death_win=0,
                        suicide=0,sudden_death_death=0,kills=win,bombs=1,steps=20,
                        opponent_outcome='loss' if win else 'draw'))
            report=build_report(reg,models,diagnostics,rb_rows)
            self.assertTrue(report['complete'])
            self.assertEqual(report['status'],'RETENTION_NOT_PRESERVED')
            for arm in ('control','anchor'):
                endpoint=report['endpoints'][arm+'500']
                expected=next(m for m in models if m['arm']==arm and m['episode']==500)
                if isinstance(endpoint,dict):
                    self.assertEqual(endpoint['episode'],500);self.assertEqual(endpoint['sha256'],expected['sha256'])
                else:self.assertIn(endpoint,(expected['id'],expected['sha256']))
            def incomplete_or_invalid(selected,diags,rows):
                try:result=build_report(reg,selected,diags,rows)
                except ValueError:return
                self.assertFalse(result['complete']);self.assertEqual(result['status'],'INCOMPLETE')
            endpoint_sha=next(m['sha256'] for m in models if m['arm']=='anchor' and m['episode']==500)
            incomplete_or_invalid([m for m in models if m['sha256']!=endpoint_sha],diagnostics,rb_rows)
            incomplete_or_invalid(models,{k:v for k,v in diagnostics.items() if k!=endpoint_sha},rb_rows)
            incomplete_or_invalid(models,diagnostics,rb_rows[:-1])
            incomplete_or_invalid(models,diagnostics,rb_rows+[deepcopy(rb_rows[0])])
            bad=deepcopy(diagnostics);bad[endpoint_sha]['raw']['decision'].pop()
            bad[endpoint_sha]['diagnostic_digest']=digest(bad[endpoint_sha],'diagnostic_digest')
            incomplete_or_invalid(models,bad,rb_rows)
            bad=deepcopy(diagnostics);bad[endpoint_sha]['summary']['R']=100.
            bad[endpoint_sha]['diagnostic_digest']=digest(bad[endpoint_sha],'diagnostic_digest')
            with self.assertRaises(ValueError):build_report(reg,models,bad,rb_rows)

    def test_validation_tasks_are_fixed_endpoint_inventory_and_paired_rb_boards(self):
        tasks_for = function(self, 'validation_tasks')
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-tasks-') as tmp:
            _,_,reg=register_case(self,Path(tmp)/'full',smoke=False)
            models=[model_descriptor()]
            for arm in ('control','anchor'):
                for ep in (100,250,500):
                    models.append(model_descriptor(f'{arm}-{ep}',arm,ep,model_sha=digest([arm,ep])))
            tasks=tasks_for(reg,models)
            self.assertEqual(len(tasks),700)
            self.assertEqual(len({t['id'] for t in tasks}),700)
            self.assertEqual({t['board'] for t in tasks},set(range(720000,720100)))
            for task in tasks:
                self.assertEqual(task['corners'],[0,1]);self.assertEqual(task['max_steps'],500)
                self.assertEqual(task['opponent']['id'],'RB')
                self.assertEqual(task['registration_digest'],reg['registration_digest'])
            _,_,smoke_reg=register_case(self,Path(tmp)/'smoke')
            smoke_models=[models[0]]
            for arm in ('control','anchor'):
                for ep in (2,4):smoke_models.append(model_descriptor(f'{arm}-{ep}',arm,ep,model_sha=digest([arm,ep])))
            smoke=tasks_for(smoke_reg,smoke_models)
            self.assertEqual(len(smoke),10)
            self.assertEqual({t['board'] for t in smoke},{790002,790003})


if __name__=='__main__':unittest.main()
