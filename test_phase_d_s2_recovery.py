"""Additional transaction regressions; independent acceptance files remain frozen."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import phase_d_s2_protocol as protocol
from test_phase_d_s2_protocol import ROOT, copy_inputs, read_json


class RegistrationRecoveryTests(unittest.TestCase):
    def test_resume_reconciles_selection_projections_after_publish_interruption(self):
        from test_run_phase_d_s2 import ready_case,synthetic_result
        import phase_d_s2_ledger as ledger
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-projection-') as temporary:
            output=ready_case(self,temporary);original=ledger.atomic_json
            def interrupted(path,value,**kwargs):
                if Path(path).name=='validation.json':raise RuntimeError('projection interrupted')
                return original(path,value,**kwargs)
            with patch.object(ledger,'atomic_json',interrupted),self.assertRaisesRegex(RuntimeError,'projection interrupted'):
                ledger.evaluate_stage(output,'validation',evaluator=synthetic_result)
            ledger.evaluate_stage(output,'validation',resume=True,evaluator=synthetic_result)
            self.assertTrue((output/'validation.json').is_file())
            self.assertTrue((output/'decision_report.md').is_file())
            self.assertTrue(read_json(output/'report.json')['complete'])

    def test_package_refuses_corrupt_selection_instead_of_shipping_a_valid_status(self):
        from test_run_phase_d_s2 import ready_case,synthetic_result
        import phase_d_s2_ledger as ledger
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-package-') as temporary:
            output=ready_case(self,temporary)
            ledger.evaluate_stage(output,'validation',evaluator=synthetic_result)
            selection=read_json(output/'selected.json');selection['open_test']=False
            protocol.atomic_json(output/'selected.json',selection,overwrite=True)
            with self.assertRaisesRegex(ValueError,'Selection'):
                ledger.package_run(output)

    def test_registration_rebuilds_unpublished_partial_transaction_on_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-partial-') as temporary:
            inputs=copy_inputs(Path(temporary)/'inputs');output=Path(temporary)/'run'
            original=protocol.atomic_json
            def interrupted(path,payload,**kwargs):
                if Path(path).name=='runtime.json':raise RuntimeError('crash during prepare')
                return original(path,payload,**kwargs)
            with patch.object(protocol,'atomic_json',interrupted),self.assertRaisesRegex(RuntimeError,'crash during prepare'):
                protocol.register(output,inputs,device='cpu')
            self.assertFalse(output.exists())
            protocol.register(output,inputs,device='cpu',resume=True)
            protocol.verify_run(output)

    def test_reserved_unpublished_transaction_resumes_without_double_reserving(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-reserve-') as temporary:
            inputs=copy_inputs(Path(temporary)/'inputs');output=Path(temporary)/'run'
            original=protocol.os.replace
            def interrupted(source,destination):
                if Path(destination)==output:raise RuntimeError('crash before publish')
                return original(source,destination)
            with patch.object(protocol.os,'replace',interrupted),self.assertRaisesRegex(RuntimeError,'crash before publish'):
                protocol.register(output,inputs,device='cpu')
            self.assertFalse(output.exists())
            registry=inputs/'runs/phase_d_s2_seed_registry.json'
            self.assertEqual(next(iter(read_json(registry)['reservations'].values()))['status'],'PENDING')
            protocol.register(output,inputs,device='cpu',resume=True)
            protocol.verify_run(output)
            self.assertEqual(len(read_json(registry)['reservations']),1)

    def test_cpu_board_worker_applies_registered_deterministic_thread_profile(self):
        import torch
        from phase_d_s2_evaluation import evaluate_board
        from test_phase_d_s2_protocol import task_fixture, CHECKPOINTS
        task,_=task_fixture();task['max_steps']=2
        task['learner'].update(id='S2',checkpoint=str(ROOT/CHECKPOINTS['S2'][0]),sha256=CHECKPOINTS['S2'][1])
        task['opponent'].update(id='RB',checkpoint=None,sha256=protocol.sha(ROOT/'rule_based_bot.py'))
        torch.set_num_threads(2)
        result=evaluate_board(task)
        self.assertEqual(torch.get_num_threads(),1)
        self.assertEqual(result['worker_runtime']['threads'],1)
        self.assertEqual(result['worker_runtime']['device'],'cpu')
        self.assertTrue(result['worker_runtime']['deterministic'])

    def test_history_line_ending_conversion_is_audited_without_relaxing_new_run_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-history-') as temporary:
            root=Path(temporary);directory=root/'runs/legacy';source=directory/'source/run_sd74_probe.py'
            source.parent.mkdir(parents=True)
            text=b'MECH_SEED, MECH_GAMES, TEST_SEED = 6700, 100, 45000\n'
            source.write_bytes(text.replace(b'\n',b'\r\n'))
            protocol.atomic_json(directory/'registration.json',dict(mechanism_seed=6700,
                sources={'run_sd74_probe.py':hashlib.sha256(text).hexdigest()}))
            result=protocol.inventory_seeds(root)
            self.assertFalse(result['unresolved'])
            self.assertTrue(any(a==6700 and b==6749 for a,b,_ in result['ranges']))

    def test_published_pending_registration_finishes_only_on_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-recovery-') as temporary:
            inputs=copy_inputs(Path(temporary)/'inputs');output=Path(temporary)/'run'
            registry=inputs/'runs/phase_d_s2_seed_registry.json'
            original=protocol.atomic_json
            def interrupted(path,payload,**kwargs):
                if Path(path)==registry and any(e['status']=='COMPLETE' for e in payload['reservations'].values()):
                    raise RuntimeError('crash after publish')
                return original(path,payload,**kwargs)
            with patch.object(protocol,'atomic_json',interrupted),self.assertRaisesRegex(RuntimeError,'crash after publish'):
                protocol.register(output,inputs,device='cpu')
            before=(output/'registration.json').read_bytes()
            with self.assertRaises(ValueError):protocol.verify_run(output)
            protocol.register(output,inputs,device='cpu',resume=True)
            protocol.verify_run(output)
            self.assertEqual((output/'registration.json').read_bytes(),before)
            self.assertEqual(next(iter(read_json(registry)['reservations'].values()))['status'],'COMPLETE')

    def test_jsonl_history_cannot_reuse_held_out_board(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-history-') as temporary:
            inputs=copy_inputs(Path(temporary)/'inputs');path=inputs/'runs/legacy/raw.jsonl'
            path.parent.mkdir(parents=True);path.write_text('{"board":630001,"corner":1}\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'overlap'):
                protocol.register(Path(temporary)/'run',inputs,device='cpu')


if __name__=='__main__':unittest.main()
