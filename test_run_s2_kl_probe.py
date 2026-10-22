"""Black-box frozen CLI journey, raw ledgers and packaging without false PASS."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from test_s2_kl_support import (ROOT, STAGES, SAMPLE_SHA, cli, copy_inputs, digest, function,
    read_json, register_case, sha, write_json)


def board_payloads(output):
    result = {}
    for path in (Path(output) / 'validation/tasks').glob('*.json'):
        value = read_json(path)
        if isinstance(value,dict) and isinstance(value.get('records'),list): result[path]=value
    return result


class ContractTests(unittest.TestCase):
    def test_cli_frozen_stages_flags_and_no_test_or_budget_escape_hatch(self):
        result=cli(self,'--help');self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn(',test',result.stdout.replace(' ',''))
        for stage in STAGES:
            result=cli(self,stage,'--help');self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--output',result.stdout)
            for forbidden in ('--force','--ignore-hash','--episodes','--lr','--beta','--held-out'):
                self.assertNotIn(forbidden,result.stdout)
        register=cli(self,'register','--help')
        for flag in ('--input-root','--device','--workers','--smoke','--resume'):
            self.assertIn(flag,register.stdout)
        self.assertIn('--arm',cli(self,'train','--help').stdout)
        self.assertNotEqual(cli(self,'test','--help').returncode,0)

    def test_invalid_order_drift_and_keyboard_interrupt_have_documented_exit_codes(self):
        main=function(self,'main')
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-cli-order-') as tmp:
            output=Path(tmp)/'absent'
            for stage in ('train','diagnose','validate','package'):
                args=[stage,'--output',str(output)]
                if stage=='train':args+=['--arm','control']
                with self.subTest(stage=stage):self.assertEqual(main(args),1)
            _,output,_=register_case(self,Path(tmp)/'valid')
            reg=read_json(output/'registration.json');reg['device']='bad-device'
            write_json(output/'registration.json',reg)
            self.assertEqual(main(['package','--output',str(output)]),1)
        # Interrupt at the public register boundary; exercise actual CLI dispatch/catch.
        module=__import__('run_s2_kl_probe')
        with patch.object(module,'register',side_effect=KeyboardInterrupt):
            self.assertEqual(main(['register','--output','unused-acceptance-output','--input-root',str(ROOT),
                                   '--device','cpu','--smoke']),130)
        self.assertFalse((ROOT/'unused-acceptance-output').exists())

    def test_training_order_missing_arm_and_overrides_create_no_game_ledger(self):
        function(self,'main')
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-cli-bad-') as tmp:
            output=Path(tmp)/'output'
            for args in (('train',),('train','--arm','pool'),('train','--arm','anchor','--episodes','8'),
                         ('validate','--test'),('test',)):
                result=cli(self,*args,'--output',output)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(board_payloads(output))


class PackagingTests(unittest.TestCase):
    def test_incomplete_package_exports_only_available_output_without_doing_missing_work(self):
        package=function(self,'package_run')
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-incomplete-') as tmp:
            inputs,output,reg=register_case(self,tmp)
            before={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
            result=package(output);self.assertIsInstance(result,dict)
            report=read_json(output/'report.json')
            self.assertEqual(report['status'],'INCOMPLETE');self.assertFalse(report['complete'])
            self.assertEqual(report['registration_digest'],reg['registration_digest'])
            for name in ('endpoints','counts','summary','comparisons','conditions','limitations','mode','model_manifest_digest'):
                self.assertIn(name,report)
            self.assertFalse((output/'arms').exists());self.assertFalse((output/'diagnostics').exists())
            self.assertFalse(board_payloads(output));self.assertFalse((output/'selected.json').exists())
            for relative,expected in before.items():self.assertEqual(sha(output/relative),expected)
            archive=Path(str(output)+'-send.zip');self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as z:
                self.assertTrue(any(name.endswith('report.json') for name in z.namelist()))
                self.assertTrue(any(name.endswith('registration.json') for name in z.namelist()))
                for name in z.namelist():
                    self.assertNotIn('collection/',name.replace('\\','/'))
                    self.assertNotIn('sd8a-20261020/',name.replace('\\','/'))
                    self.assertNotIn('../',name.replace('\\','/'))
                    self.assertFalse(name.startswith(('/', '\\')))
            self.assertEqual(sha(inputs/'runs/phaseD-s2-20261022-analysis-20261023/offline_sample_manifest.json'),SAMPLE_SHA)


class JourneyTests(unittest.TestCase):
    def test_actual_cpu_cli_journey_snapshots_raw_ledgers_resume_and_projection_integrity(self):
        for name in ('main','build_report','diagnose_run','validate_run','candidates_for_run'):
            function(self,name)
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-journey-') as tmp:
            inputs=copy_inputs(Path(tmp)/'inputs');output=Path(tmp)/'output'
            stages=[('register','--input-root',inputs,'--device','cpu','--workers','1','--smoke'),
                    ('train','--arm','control'),('train','--arm','anchor'),('diagnose',),('validate',),('package',)]
            for args in stages:
                result=cli(self,*args,'--output',output)
                self.assertEqual(result.returncode,0,f'{args}: {result.stdout}\n{result.stderr}')
            reg=read_json(output/'registration.json');manifest=read_json(output/'model_manifest.json')
            report=read_json(output/'report.json')
            self.assertEqual(report['status'],'SMOKE_ONLY');self.assertTrue(report['complete'])
            self.assertEqual(report['mode'],'smoke')
            self.assertEqual(set(report['endpoints']),{'baseline','control500','anchor500'})
            self.assertFalse((output/'selected.json').exists());self.assertFalse((output/'test_opened.json').exists())
            model_digest=manifest.get('model_manifest_digest',manifest.get('digest'))
            self.assertRegex(model_digest,r'^[0-9a-f]{64}$')
            digest_key='model_manifest_digest' if 'model_manifest_digest' in manifest else 'digest'
            self.assertEqual(model_digest,digest(manifest,digest_key))
            self.assertEqual(report['model_manifest_digest'],model_digest)
            models=function(self,'candidates_for_run')(output,reg)
            diagnostics={p.stem:read_json(p) for p in (output/'diagnostics').glob('*.json')}
            self.assertEqual(set(diagnostics),{m['sha256'] for m in models})
            payloads=board_payloads(output)
            self.assertEqual(len(payloads),2*len(models))
            rows=[]
            for payload in payloads.values():
                records=payload['records'];rows+=records
                self.assertEqual(len(records),2)
                self.assertEqual(sorted(r['corner'] for r in records),[0,1])
                self.assertEqual(len({r['board'] for r in records}),1)
                self.assertIn(records[0]['board'],(790002,790003))
                runtime=payload['worker_runtime']
                self.assertEqual(runtime['device'],'cpu');self.assertEqual(runtime['threads'],1)
                self.assertTrue(runtime['deterministic'])
                self.assertEqual(Path(runtime['source_root']).resolve(),(output/'sources').resolve())
                self.assertEqual(payload['registration_digest'],reg['registration_digest'])
            self.assertEqual(len({(r['learner_sha256'],r['board'],r['corner']) for r in rows}),len(rows))
            reconstructed=function(self,'build_report')(reg,models,diagnostics,rows)
            for key in ('status','complete','endpoints','counts','summary','comparisons'):
                self.assertEqual(reconstructed[key],report[key],key)
            fingerprints={str(p.relative_to(output)):sha(p) for p in payloads}
            fingerprints.update({str(p.relative_to(output)):sha(p) for p in (output/'diagnostics').glob('*.json')})
            for stage in ('diagnose','validate'):
                result=cli(self,stage,'--resume','--output',output)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for relative,expected in fingerprints.items():self.assertEqual(sha(output/relative),expected)
            # A single-corner committed shard cannot be silently reused or count twice.
            task_path=next(iter(payloads));original=task_path.read_bytes();bad=deepcopy(payloads[task_path]);bad['records'].pop()
            write_json(task_path,bad)
            result=cli(self,'validate','--resume','--output',output)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            task_path.write_bytes(original)
            # Recomputed metadata digest must not legitimize a cached-summary/raw mismatch.
            bad_diags=deepcopy(diagnostics);first=next(iter(bad_diags));bad_diags[first]['summary']['R']+=1.
            bad_diags[first]['diagnostic_digest']=digest(bad_diags[first],'diagnostic_digest')
            with self.assertRaises(ValueError):function(self,'build_report')(reg,models,bad_diags,rows)
            # Recomputed digest also cannot replace one locked row identity.
            bad_diags=deepcopy(diagnostics);bad_diags[first]['raw']['decision'][0]['id']='not-a-locked-row'
            bad_diags[first]['diagnostic_digest']=digest(bad_diags[first],'diagnostic_digest')
            with self.assertRaises(ValueError):function(self,'build_report')(reg,models,bad_diags,rows)
            # Missing offline observations and incomplete RB data must not create scientific completion.
            self.assertFalse(function(self,'build_report')(reg,models,{},rows)['complete'])
            self.assertEqual(function(self,'build_report')(reg,models,diagnostics,rows[:-1])['status'],'INCOMPLETE')
            # The next worker/stage must reject modified frozen engine bytes.
            source=output/'sources/bomberman_env.py'
            with source.open('ab') as handle:handle.write(b'\n# acceptance source drift\n')
            result=cli(self,'package','--output',output)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
