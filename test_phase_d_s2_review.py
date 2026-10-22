"""Final-review regressions, in addition to the frozen independent acceptance suite."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phase_d_s2_protocol as protocol
import phase_d_s2_ledger as ledger
from test_phase_d_s2_protocol import ROOT,copy_inputs,read_json
from test_phase_d_s2_training import register_case
from test_run_phase_d_s2 import ready_case,synthetic_result


class ReviewRegressions(unittest.TestCase):
    def test_missing_state_after_committed_episode_is_not_restarted(self):
        import threading
        from phase_d_s2_training import train_arm
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-missing-state-') as temporary:
            _,out=register_case(self,temporary);stop=threading.Event()
            train_arm(out,'pool',stop_event=stop,on_episode=lambda row:stop.set())
            state=out/'arms/pool/training_state.pt';self.assertTrue(state.exists());state.unlink()
            with self.assertRaisesRegex(ValueError,'Committed artifacts'):
                train_arm(out,'pool',resume=True)

    def test_config_without_seed_mapping_cannot_register_full(self):
        for payload in ({'config':{}},{'config':{'phase':'D','episodes':2000}},
                        {'config':{'phase':'D','episodes':2000,'seed':20261022}}):
            with self.subTest(payload=payload),tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-unknown-') as temporary:
                inputs=copy_inputs(Path(temporary)/'inputs')
                protocol.atomic_json(inputs/'runs/unknown/registration.json',payload)
                with self.assertRaisesRegex(ValueError,'Unresolved'):
                    protocol.register(Path(temporary)/'output',inputs,device='cpu')

    def test_invalid_board_metadata_and_nonfinite_jsonl_are_unresolved(self):
        cases=[('registration.json',{'seed_manifest':{'ranges':{'x':{'start':640099,'end_inclusive':630000}}}}),
               ('raw.jsonl','{"board":null}\n'),('raw.jsonl','{"board":NaN}\n')]
        for name,payload in cases:
            with self.subTest(payload=payload),tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-bad-history-') as temporary:
                inputs=copy_inputs(Path(temporary)/'inputs');path=inputs/'runs/unknown'/name
                path.parent.mkdir(parents=True)
                if isinstance(payload,str):path.write_text(payload,encoding='utf-8')
                else:protocol.atomic_json(path,payload)
                with self.assertRaisesRegex(ValueError,'Unresolved'):
                    protocol.register(Path(temporary)/'output',inputs,device='cpu')

    def test_interrupted_source_copy_and_missing_early_identity_resume(self):
        for before_identity in (False,True):
            with self.subTest(before_identity=before_identity),tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-copy-') as temporary:
                inputs=copy_inputs(Path(temporary)/'inputs');out=Path(temporary)/'output'
                if before_identity:
                    out.with_name(out.name+'.registration-pending').mkdir()
                else:
                    def interrupt(source,destination):
                        Path(destination).write_bytes(b'incomplete source copy')
                        raise RuntimeError('copy interrupted')
                    with patch.object(protocol.shutil,'copy2',interrupt),self.assertRaisesRegex(RuntimeError,'copy interrupted'):
                        protocol.register(out,inputs,device='cpu')
                protocol.register(out,inputs,device='cpu',resume=True)
                reg=protocol.verify_run(out)
                self.assertTrue(all(protocol.sha(out/'sources'/name)==sha for name,sha in reg['sources'].items()))

    def test_registration_only_cannot_package_unearned_pass_or_completion(self):
        for status in ('PHASE_D_PASS','PILOT_PASS','VALIDATION_STOP'):
            with self.subTest(status=status),tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-false-report-') as temporary:
                _,out=register_case(self,temporary)
                report=ledger._incomplete('registered',0,0)
                report.update(complete=True,status=status)
                report['pilot_gate']['pass']=True;report['phase_d_gate']['pass']=True
                protocol.atomic_json(out/'report.json',report)
                with self.assertRaisesRegex(ValueError,'Report'):
                    ledger.package_run(out)
                self.assertFalse(out.with_name(out.name+'-send.zip').exists())
                self.assertFalse((out/'decision_report.md').exists())

    def test_validation_only_cannot_package_test_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-false-validation-') as temporary:
            out=ready_case(self,temporary)
            ledger.evaluate_stage(out,'validation',evaluator=synthetic_result)
            report=read_json(out/'report.json');report['status']='PHASE_D_PASS'
            report['pilot_gate']['pass']=True;report['phase_d_gate']['pass']=True
            protocol.atomic_json(out/'report.json',report,overwrite=True)
            with self.assertRaisesRegex(ValueError,'Report'):
                ledger.package_run(out)

    def test_test_marker_crash_before_first_result_packages_as_incomplete(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-before-test-') as temporary:
            out=ready_case(self,temporary)
            ledger.evaluate_stage(out,'validation',evaluator=synthetic_result)
            validation_report=(out/'report.json').read_bytes()
            with patch.object(ledger,'run_tasks',side_effect=RuntimeError('before any board')):
                with self.assertRaisesRegex(RuntimeError,'before any board'):
                    ledger.evaluate_stage(out,'test',evaluator=synthetic_result)
            # Simulate abrupt process death after the marker, before the exception handler report.
            (out/'report.json').write_bytes(validation_report)
            result=ledger.package_run(out)
            self.assertEqual(result['status'],'INCOMPLETE')
            report=read_json(out/'report.json')
            self.assertFalse(report['complete'])
            self.assertEqual(report['actual_games'],0)
            self.assertGreater(report['expected_games'],0)

    def test_train_resume_before_initial_state_commit_restarts_only_initialization(self):
        from phase_d_s2_training import train_arm,read_training_state
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='d-s2-initialize-') as temporary:
            _,out=register_case(self,temporary)
            def interrupt(event,episode):
                if event=='before_state_commit' and episode==0:raise RuntimeError('initialization interrupted')
            with self.assertRaisesRegex(RuntimeError,'initialization interrupted'):
                train_arm(out,'pool',commit_hook=interrupt)
            self.assertFalse((out/'arms/pool/training_state.pt').exists())
            train_arm(out,'pool',resume=True)
            self.assertEqual(read_training_state(out,'pool')['episode'],4)


if __name__=='__main__':unittest.main()
