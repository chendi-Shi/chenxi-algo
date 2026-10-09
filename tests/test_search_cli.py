"""CLI integration for optional search output and reproducible run identity."""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import engine
import run
from test_engine import AS_OF
from test_search import fixture


class SearchCLITests(unittest.TestCase):
    def invoke(self, search_args=None, space=None):
        rows, values, config = fixture()
        written = {}
        args = ['--as-of', AS_OF.isoformat(), '--statements', 'statements.csv',
                '--valuations', 'valuations.csv', '--output', 'mock-output']
        if search_args is None:
            search_args = ['--search-method', 'grid', '--search-market', 'A',
                '--search-sector', 'Industrials', '--search-target-size', '2',
                '--search-space', 'search-space.json']
        args.extend(search_args)
        space = {'candidate_score': [50, 55]} if space is None else space

        def read_input(path):
            return copy.deepcopy(rows if path.name == 'statements.csv' else values)

        def write_output(path, content, **kwargs):
            written[path.name] = content
            return len(content)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(run, 'config_from', return_value=config))
            stack.enter_context(mock.patch.object(run, 'load_csv', side_effect=read_input))
            stack.enter_context(mock.patch.object(Path, 'read_bytes', autospec=True,
                side_effect=lambda path: path.name.encode('utf-8')))
            stack.enter_context(mock.patch.object(Path, 'read_text', return_value=json.dumps(space)))
            stack.enter_context(mock.patch.object(Path, 'write_text', autospec=True, side_effect=write_output))
            stack.enter_context(mock.patch.object(Path, 'mkdir'))
            stack.enter_context(mock.patch.object(run, 'export_csv'))
            stack.enter_context(mock.patch.object(run, 'save_database'))
            stack.enter_context(mock.patch('sys.stdout', new_callable=io.StringIO))
            stderr = stack.enter_context(mock.patch('sys.stderr', new_callable=io.StringIO))
            status = run.main(args)
        return status, written, stderr.getvalue(), rows, values, config

    def test_search_is_exported_without_replacing_main_screen(self):
        status, written, error, rows, values, config = self.invoke()
        self.assertEqual(status, 0, error)
        result = json.loads(written['results.json'])
        search = json.loads(written['search.json'])
        self.assertEqual(search, result['configuration_search'])
        self.assertTrue(search['best']['target_met'])
        self.assertEqual(result['companies'], engine.screen(rows, values, AS_OF, config)['companies'])
        self.assertIn('筛选参数搜索', written['report.html'])
        manifest = json.loads(written['manifest.json'])
        self.assertEqual(manifest['configuration_search_request']['target_size'], 2)

    def test_search_parameters_and_space_are_part_of_run_identity(self):
        first = json.loads(self.invoke()[1]['manifest.json'])
        repeat = json.loads(self.invoke()[1]['manifest.json'])
        changed_space = json.loads(self.invoke(space={'candidate_score': [50, 60]})[1]['manifest.json'])
        changed_method = json.loads(self.invoke(search_args=[
            '--search-method', 'beam', '--search-market', 'A', '--search-sector', 'Industrials',
            '--search-target-size', '2', '--search-beam-width', '1', '--search-space', 'search-space.json'
        ])[1]['manifest.json'])
        self.assertEqual(first['run_id'], repeat['run_id'])
        self.assertNotEqual(first['run_id'], changed_space['run_id'])
        self.assertNotEqual(first['run_id'], changed_method['run_id'])

    def test_search_requires_explicit_cohort_and_method(self):
        for args in (['--search-method', 'grid'], ['--search-market', 'A'],
                     ['--search-target-size', '3'], ['--search-beam-width', '2']):
            with self.subTest(args=args):
                status, _, message, *_ = self.invoke(search_args=args)
                self.assertEqual(status, 2)
                self.assertIn('require', message)

    def test_no_search_run_overwrites_stale_search_artifact(self):
        status, written, message, *_ = self.invoke(search_args=[])
        self.assertEqual(status, 0, message)
        self.assertEqual(json.loads(written['search.json'])['status'], 'not_requested')
        self.assertNotIn('configuration_search', json.loads(written['results.json']))

    def test_search_space_rejects_nonfinite_json_and_nonobjects(self):
        for text in ('{"min_roe":[NaN]}', '{"min_roe":[1e999]}', '[]'):
            with self.subTest(text=text), mock.patch.object(Path, 'read_text', return_value=text):
                with self.assertRaises(ValueError):
                    run.load_search_space(Path('unused.json'))

    def test_report_escapes_search_sources_and_candidate_labels(self):
        _, written, _, *_ = self.invoke()
        result = json.loads(written['results.json'])
        result['configuration_search']['sector'] = '<script>alert(1)</script>'
        result['configuration_search']['best']['candidates'] = ['<img src=x onerror=alert(1)>']
        html = run.report_html(result)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('&lt;img src=x', html)


if __name__ == '__main__':
    unittest.main()
