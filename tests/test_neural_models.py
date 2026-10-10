"""Offline numerical and integrity contracts, with fake ONNX sessions/tokenizers."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
from urllib.error import HTTPError

import numpy as np

import neural_models as neural


class FakeTokenizer:
    instances = []

    def __init__(self):
        self.calls = []
        self.__class__.instances.append(self)

    @classmethod
    def from_file(cls, filename):
        return cls()

    def token_to_id(self, token):
        return 7 if token == '<pad>' else None

    def enable_truncation(self, **kwargs):
        self.truncation = kwargs

    def enable_padding(self, **kwargs):
        self.padding = kwargs

    def encode_batch(self, values, **kwargs):
        self.calls.append((values, kwargs))
        return [types.SimpleNamespace(ids=[1, 2, 7], attention_mask=[1, 1, 0], type_ids=[0, 0, 0],
                 overflowing=[object()] if 'TRUNCATED' in str(value) else []) for value in values]


class FakeSession:
    instances = []
    transform = None

    def __init__(self, path, **kwargs):
        self.role = 'embedding' if 'embedding' in str(path) else 'reranker'
        self.options = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_inputs(self):
        names = ['input_ids', 'attention_mask', 'token_type_ids'] if self.role == 'embedding' else ['input_ids', 'attention_mask']
        return [types.SimpleNamespace(name=name, type='tensor(int64)') for name in names]

    def get_outputs(self):
        return [types.SimpleNamespace(name='last_hidden_state' if self.role == 'embedding' else 'logits')]

    def run(self, names, inputs):
        self.calls.append((names, inputs))
        batch, length = inputs['input_ids'].shape
        if self.role == 'embedding':
            output = np.zeros((batch, length, 384), dtype=np.float32)
            output[:, 0, 0] = 3
            output[:, 1, 1] = 4
            output[:, 2, :] = 10000  # Padding must make no contribution to the pooled vector.
        else:
            output = np.arange(batch, dtype=np.float32).reshape(-1, 1) - 2.5
        if self.__class__.transform:
            output = self.__class__.transform(output)
        return [output]


class NeuralModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = neural.filesystem_path(self.tmp.name) / 'models'
        self.specs, self.contents = {}, {}
        for role in ('embedding', 'reranker'):
            files = {'model.onnx': b'fake-onnx-' + role.encode(), 'tokenizer.json': b'{}',
                     'tokenizer_config.json': b'{"pad_token":"<pad>"}'}
            self.contents[role] = files
            self.specs[role] = dict(repository='test/' + role, revision='a' * 40, model_file='model.onnx',
                files={name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in files.items()})
        self.patch = mock.patch.object(neural, 'MODEL_SPECS', self.specs)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.write_bundle()
        FakeTokenizer.instances, FakeSession.instances, FakeSession.transform = [], [], None
        runtime = types.SimpleNamespace(SessionOptions=types.SimpleNamespace,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL='sequential'), InferenceSession=FakeSession)
        self.deps = mock.patch.object(neural, '_dependencies', return_value=(np, runtime, FakeTokenizer)).start()
        self.addCleanup(mock.patch.stopall)

    def write_bundle(self):
        for role, files in self.contents.items():
            for name, raw in files.items():
                path = neural._artifact(self.root, role, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        (self.root / 'bundle.json').write_text(json.dumps(neural._marker()), encoding='utf-8')

    def test_masked_mean_l2_prefix_and_cpu_session_contract(self):
        model = neural.ModelBundle(self.root)
        embeddings = model.encode(['hello', 'different length'], kind='query')
        self.assertEqual(embeddings.shape, (2, 384))
        np.testing.assert_allclose(embeddings[:, :2], [[.6, .8], [.6, .8]], atol=1e-6)
        np.testing.assert_array_equal(embeddings[:, 2:], 0)
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1)
        tokenizer, session = FakeTokenizer.instances[0], FakeSession.instances[0]
        self.assertEqual(tokenizer.calls[0][0], ['query: hello', 'query: different length'])
        self.assertEqual(tokenizer.truncation['max_length'], 512)
        self.assertEqual(tokenizer.padding['pad_id'], 7)
        self.assertEqual(session.options['providers'], ['CPUExecutionProvider'])
        self.assertEqual(session.options['sess_options'].intra_op_num_threads, 2)
        self.assertTrue(session.fallback_disabled)
        self.assertEqual(session.calls[0][1]['input_ids'].dtype, np.int64)
        model.encode(['hello'])
        self.assertEqual(tokenizer.calls[1][0], ['passage: hello'])
        self.assertEqual(len(FakeSession.instances), 1)

    def test_reranker_uses_pairs_without_prefix_and_returns_raw_logits(self):
        model = neural.ModelBundle(self.root)
        scores = model.rerank('question', ['first passage', 'second passage'])
        np.testing.assert_array_equal(scores, [-2.5, -1.5])
        self.assertEqual(FakeTokenizer.instances[0].calls[0][0], [('question', 'first passage'), ('question', 'second passage')])
        self.assertEqual(set(FakeSession.instances[0].calls[0][1]), {'input_ids', 'attention_mask'})

    def test_batch_and_truncation_statistics_record_indices(self):
        model = neural.ModelBundle(self.root, batch_size=2)
        result = model.encode(['one', 'TRUNCATED', 'TRUNCATED last'])
        stats = model.statistics()['embedding']
        self.assertEqual(result.shape, (3, 384))
        self.assertEqual(stats['batches'], 2)
        self.assertEqual(stats['truncated_inputs'], 2)
        self.assertEqual(stats['last_truncated_indices'], [1, 2])
        self.assertEqual(stats['tokens_after_truncation'], 6)
        stats['inputs'] = 0
        self.assertEqual(model.statistics()['embedding']['inputs'], 3)

    def test_fingerprint_is_offline_and_does_not_load_runtime(self):
        with mock.patch.object(neural, 'urlopen', side_effect=AssertionError('network')):
            model = neural.ModelBundle(self.root)
            fingerprint = model.fingerprint()
        self.deps.assert_not_called()
        self.assertEqual(fingerprint['models'], self.specs)
        self.assertEqual(fingerprint['embedding']['dimension'], 384)
        self.assertEqual(fingerprint['reranker']['score'], 'raw_logit')
        self.assertNotIn(str(self.root), json.dumps(fingerprint))

    def test_missing_cache_fails_without_network_or_automatic_fallback(self):
        with mock.patch.object(neural, 'urlopen', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'bundle is missing'):
                neural.ModelBundle(self.root / 'absent').encode(['text'])
        self.deps.assert_not_called()

    def test_hash_mismatch_and_manifest_mismatch_reject_before_runtime(self):
        path = neural._artifact(self.root, 'embedding', 'model.onnx')
        path.write_bytes(b'x' * len(self.contents['embedding']['model.onnx']))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            neural.ModelBundle(self.root).encode(['text'])
        self.deps.assert_not_called()
        self.write_bundle()
        (self.root / 'bundle.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            neural.ModelBundle(self.root).fingerprint()

    def test_cache_mutation_after_load_cannot_silently_change_fingerprint(self):
        model = neural.ModelBundle(self.root)
        model.encode(['text'])
        neural._artifact(self.root, 'embedding', 'tokenizer.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'cache changed'):
            model.fingerprint()
        with self.assertRaisesRegex(ValueError, 'cache changed'):
            model.encode(['another text'])

    def test_bad_shape_nonfinite_and_zero_embedding_are_rejected(self):
        for transform, message in ((lambda x: x[:, :, :2], 'shape'),
                                   (lambda x: np.full_like(x, np.nan), 'nonfinite'),
                                   (lambda x: np.zeros_like(x), 'zero norm')):
            with self.subTest(message=message):
                FakeSession.transform = transform
                with self.assertRaisesRegex(ValueError, message):
                    neural.ModelBundle(self.root).encode(['text'])
        FakeSession.transform = lambda x: np.zeros((x.shape[0], 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'shape'):
            neural.ModelBundle(self.root).rerank('query', ['text'])

    def test_input_validation_and_empty_shapes(self):
        for kwargs in ({'threads': 0}, {'threads': True}, {'batch_size': 33}):
            with self.assertRaises(ValueError):
                neural.ModelBundle(self.root, **kwargs)
        model = neural.ModelBundle(self.root)
        for value in ('one string', [''], [None], ['x' * 32001]):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                model.encode(value)
        with self.assertRaisesRegex(ValueError, 'kind'):
            model.encode(['text'], kind='invalid')
        self.assertEqual(model.encode([]).shape, (0, 384))
        self.assertEqual(model.rerank('query', []).shape, (0,))

    def test_explicit_download_reuses_verified_cache_without_network(self):
        with mock.patch.object(neural, '_fetch', side_effect=AssertionError('network')):
            result = neural.download_models(self.root)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(all(row['acquisition'] == 'verified_cache' for row in result['files']))

    def test_fetch_rejects_corrupt_content_and_does_not_replace_existing_file(self):
        target = self.root / 'existing'
        target.write_bytes(b'old data')
        response = io.BytesIO(b'corrupt!')
        response.status, response.headers = 200, {'Content-Length': '8'}
        expected = {'bytes': 8, 'sha256': hashlib.sha256(b'correct!').hexdigest()}
        with mock.patch.object(neural, 'urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                neural._fetch('https://huggingface.co/test', target, expected)
        self.assertEqual(target.read_bytes(), b'old data')
        self.assertFalse(list(self.root.glob('.model-*')))

    def test_download_retries_transient_http_failure_but_not_hash_failure(self):
        path = neural._artifact(self.root, 'embedding', 'model.onnx')
        path.unlink()
        def succeed(url, target, expected):
            target.write_bytes(self.contents['embedding']['model.onnx'])
        transient = HTTPError('https://huggingface.co/test', 503, 'unavailable', {}, None)
        def fetch_once_then_succeed(url, target, expected, **kwargs):
            if fetch.call_count == 1:
                raise transient
            succeed(url, target, expected)
        with mock.patch.object(neural, '_fetch', side_effect=fetch_once_then_succeed) as fetch, mock.patch.object(neural.time, 'sleep'):
            result = neural.download_models(self.root)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(result['files'][0]['attempts'], 2)
            neural.ModelBundle(self.root).fingerprint()
        path.unlink()
        with mock.patch.object(neural, '_fetch', side_effect=ValueError('SHA256 mismatch')) as fetch:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                neural.download_models(self.root)
            self.assertEqual(fetch.call_count, 1)


if __name__ == '__main__':
    unittest.main()
