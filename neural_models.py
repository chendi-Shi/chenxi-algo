"""Pinned, local CPU ONNX inference; network access exists only in download_models.

The official E5 int8 graph is named for its export quantization preset. ONNX
Runtime chooses supported CPU kernels at runtime; no native AVX512 binary is
loaded. Quantized ranking quality still needs a task-specific evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ingest_theme import filesystem_path, output_lock, _atomic_write


MAX_TOKENS = 512
EMBEDDING_DIMENSION = 384
MAX_TEXT_CHARS = 32000
MODEL_SPECS = {
    'embedding': {
        'repository': 'intfloat/multilingual-e5-small',
        'revision': '614241f622f53c4eeff9890bdc4f31cfecc418b3',
        'license': 'MIT',
        'model_file': 'onnx/model_qint8_avx512_vnni.onnx',
        'files': {
            'onnx/model_qint8_avx512_vnni.onnx': {'bytes': 118346824, 'sha256': 'dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88'},
            'tokenizer.json': {'bytes': 17082730, 'sha256': '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39'},
            'config.json': {'bytes': 655, 'sha256': '69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959'},
            'tokenizer_config.json': {'bytes': 443, 'sha256': 'a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b'},
            'special_tokens_map.json': {'bytes': 167, 'sha256': 'd05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7'},
            '1_Pooling/config.json': {'bytes': 200, 'sha256': '987f7a67a38fa564c849bb5d277c52ab9088a84368fc0be31a354125aebb12a0'},
            'sentence_bert_config.json': {'bytes': 57, 'sha256': '948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529'},
        },
    },
    'reranker': {
        'repository': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        'revision': '1427fd652930e4ba29e8149678df786c240d8825',
        'license': 'Apache-2.0',
        'model_file': 'onnx/model_quint8_avx2.onnx',
        'files': {
            'onnx/model_quint8_avx2.onnx': {'bytes': 118620016, 'sha256': '6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b'},
            'tokenizer.json': {'bytes': 17082660, 'sha256': '62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626'},
            'config.json': {'bytes': 891, 'sha256': 'cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77'},
            'tokenizer_config.json': {'bytes': 435, 'sha256': 'e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e'},
            'special_tokens_map.json': {'bytes': 239, 'sha256': '378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d'},
        },
    },
}


def _digest(path):
    result = hashlib.sha256()
    with filesystem_path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _marker():
    return {'schema_version': 1, 'status': 'complete', 'models': MODEL_SPECS}


def _marker_path(root):
    path = root / 'bundle.json'
    if path.is_symlink() or not filesystem_path(path).is_relative_to(root):
        raise ValueError('Model bundle marker must not escape its cache directory')
    return filesystem_path(path)


def _artifact(root, role, name):
    path = root / role / MODEL_SPECS[role]['revision'] / name
    if path.is_symlink() or not filesystem_path(path).is_relative_to(root):
        raise ValueError('Model cache artifact must not escape its cache directory')
    return filesystem_path(path)


def _valid_artifact(path, expected):
    return path.is_file() and path.stat().st_size == expected['bytes'] and _digest(path) == expected['sha256']


def _fetch(url, target, expected, *, progress=None):
    """Fetch one fixed official artifact and expose it only after SHA256 verification."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.model-', delete=False) as handle:
            temporary = Path(handle.name)
            digest, size, began = hashlib.sha256(), 0, time.monotonic()
            request = Request(url, headers={'User-Agent': 'chenxi-algo-local-models/1', 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError('Model download did not return HTTP 200')
                declared = response.headers.get('Content-Length')
                if declared is not None and int(declared) != expected['bytes']:
                    raise ValueError('Model artifact Content-Length mismatch')
                while True:
                    if time.monotonic() - began > 600:
                        raise TimeoutError('Model download exceeded its time budget')
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > expected['bytes']:
                        raise ValueError('Model artifact exceeds pinned size')
                    digest.update(block)
                    handle.write(block)
                    if progress and (size % (8 * 1024 * 1024) == 0 or size == expected['bytes']):
                        progress(size)
            if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError('Model artifact size/SHA256 mismatch')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_models(cache_dir, *, progress=None):
    """Explicit network operation. Download only the pinned ONNX/tokenizer/config files."""
    root = filesystem_path(cache_dir)
    records = []
    with output_lock(root):
        root.mkdir(parents=True, exist_ok=True)
        marker = _marker_path(root)
        for role, spec in MODEL_SPECS.items():
            for name, expected in spec['files'].items():
                target = _artifact(root, role, name)
                cached = _valid_artifact(target, expected)
                attempts = 0
                if not cached:
                    url = f'https://huggingface.co/{spec["repository"]}/resolve/{spec["revision"]}/{name}'
                    for attempts in range(1, 4):
                        try:
                            transfer = (lambda size: progress({'role': role, 'file': name,
                                'status': 'receiving', 'attempt': attempts, 'bytes_received': size,
                                'expected_bytes': expected['bytes']})) if progress else None
                            _fetch(url, target, expected, progress=transfer)
                            break
                        except HTTPError as exc:
                            if attempts == 3 or exc.code not in (429, 500, 502, 503, 504):
                                raise
                            time.sleep(2 ** (attempts - 1))
                        except (URLError, TimeoutError, ConnectionError):
                            if attempts == 3:
                                raise
                            time.sleep(2 ** (attempts - 1))
                record = dict(role=role, file=name, sha256=expected['sha256'], bytes=expected['bytes'],
                              acquisition='verified_cache' if cached else 'https_download', attempts=attempts)
                records.append(record)
                if progress:
                    progress(record)
        _atomic_write(marker, (json.dumps(_marker(), sort_keys=True) + '\n').encode('utf-8'))
    return {'status': 'completed', 'files': records, 'total_bytes': sum(row['bytes'] for row in records)}


def _dependencies():
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError('Local semantic dependencies are missing; install requirements-semantic.txt into .venv-semantic') from exc
    return np, ort, Tokenizer


class ModelBundle:
    """Offline-only, lazily loaded pair of pinned models. Calls on one instance serialize."""

    def __init__(self, cache_dir, threads=2, batch_size=8):
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError('threads must be an integer from 1 to 8')
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError('batch_size must be an integer from 1 to 32')
        self.cache_dir = filesystem_path(cache_dir)
        self.threads, self.batch_size = threads, batch_size
        self._identities, self._sessions, self._tokenizers = {}, {}, {}
        self._lock = threading.RLock()
        self._stats = {role: dict(calls=0, inputs=0, batches=0, truncated_inputs=0, tokens_after_truncation=0,
                                 last_truncated_indices=[]) for role in MODEL_SPECS}

    def _verify(self):
        marker = _marker_path(self.cache_dir)
        if not marker.is_file():
            raise ValueError('Verified model bundle is missing; run the explicit neural_models.py download command')
        if json.loads(marker.read_text(encoding='utf-8')) != _marker():
            raise ValueError('Model bundle manifest differs from the pinned specification')
        identities = {}
        for role, spec in MODEL_SPECS.items():
            for name, expected in spec['files'].items():
                path = _artifact(self.cache_dir, role, name)
                if not path.is_file():
                    raise ValueError(f'Model artifact is missing: {role}/{name}')
                before = _identity(path)
                if not _valid_artifact(path, expected):
                    raise ValueError(f'Model artifact size/SHA256 mismatch: {role}/{name}')
                if _identity(path) != before:
                    raise ValueError('Model artifact changed while its SHA256 was being verified')
                identities[path] = before
        identities[marker] = _identity(marker)
        self._identities = identities

    def _check_files(self):
        if not self._identities:
            self._verify()
        for path, before in self._identities.items():
            if not path.is_file() or _identity(path) != before:
                raise ValueError('Model cache changed after verification; create a new verified ModelBundle')

    def _load(self, role):
        self._check_files()
        np, ort, tokenizer_class = _dependencies()
        if role not in self._sessions:
            spec = MODEL_SPECS[role]
            tokenizer = tokenizer_class.from_file(str(_artifact(self.cache_dir, role, 'tokenizer.json')))
            config = json.loads(_artifact(self.cache_dir, role, 'tokenizer_config.json').read_text(encoding='utf-8'))
            pad_token = config['pad_token']
            if isinstance(pad_token, dict):
                pad_token = pad_token['content']
            pad_id = tokenizer.token_to_id(pad_token)
            if pad_id is None:
                raise ValueError('Pinned tokenizer has no configured padding token')
            tokenizer.enable_truncation(max_length=MAX_TOKENS, strategy='longest_first', stride=0, direction='right')
            tokenizer.enable_padding(direction='right', pad_id=pad_id, pad_type_id=0, pad_token=pad_token)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = self.threads, 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(_artifact(self.cache_dir, role, spec['model_file'])),
                                           sess_options=options, providers=['CPUExecutionProvider'])
            session.disable_fallback()
            inputs = {item.name: item.type for item in session.get_inputs()}
            if not {'input_ids', 'attention_mask'} <= inputs.keys() or set(inputs) - {'input_ids', 'attention_mask', 'token_type_ids'}:
                raise ValueError('Unsupported ONNX input contract')
            if any(value != 'tensor(int64)' for value in inputs.values()):
                raise ValueError('ONNX input tensors must use int64')
            output_name = 'last_hidden_state' if role == 'embedding' else 'logits'
            if output_name not in {item.name for item in session.get_outputs()}:
                raise ValueError(f'ONNX model is missing output {output_name}')
            self._tokenizers[role], self._sessions[role] = tokenizer, session
        return np, self._tokenizers[role], self._sessions[role]

    @staticmethod
    def _texts(texts):
        if isinstance(texts, (str, bytes)):
            raise ValueError('texts must be an iterable of text strings, not one string')
        values = list(texts)
        if len(values) > 100000:
            raise ValueError('At most 100000 texts are accepted per call')
        if any(not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS for text in values):
            raise ValueError(f'Each text must contain 1 to {MAX_TEXT_CHARS} characters and cannot be blank')
        return values

    def _batch(self, role, values, offset, np, tokenizer, session):
        encoded = tokenizer.encode_batch(values, add_special_tokens=True)
        if len(encoded) != len(values):
            raise ValueError('Tokenizer returned an incorrect batch size')
        arrays = {'input_ids': np.asarray([x.ids for x in encoded], dtype=np.int64),
                  'attention_mask': np.asarray([x.attention_mask for x in encoded], dtype=np.int64),
                  'token_type_ids': np.asarray([x.type_ids for x in encoded], dtype=np.int64)}
        shape = arrays['input_ids'].shape
        if len(shape) != 2 or shape[0] != len(values) or not 1 <= shape[1] <= MAX_TOKENS:
            raise ValueError('Tokenizer output shape exceeds the 512-token contract')
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError('Tokenizer output tensor shapes differ')
        mask = arrays['attention_mask']
        if not np.isin(mask, (0, 1)).all() or np.any(mask.sum(axis=1) == 0):
            raise ValueError('Tokenizer produced an invalid attention mask')
        stats = self._stats[role]
        stats['batches'] += 1
        stats['inputs'] += len(values)
        stats['tokens_after_truncation'] += int(mask.sum())
        truncated = [offset + i for i, item in enumerate(encoded) if item.overflowing]
        stats['truncated_inputs'] += len(truncated)
        stats['last_truncated_indices'].extend(truncated)
        inputs = {item.name: arrays[item.name] for item in session.get_inputs()}
        output_name = 'last_hidden_state' if role == 'embedding' else 'logits'
        output = np.asarray(session.run([output_name], inputs)[0])
        if not np.isfinite(output).all():
            raise ValueError('ONNX model returned nonfinite values')
        return output, mask

    def encode(self, texts, kind='passage'):
        if kind not in ('passage', 'query'):
            raise ValueError('kind must be passage or query')
        values = self._texts(texts)
        with self._lock:
            np, tokenizer, session = self._load('embedding')
            self._stats['embedding']['calls'] += 1
            self._stats['embedding']['last_truncated_indices'] = []
            results = []
            for offset in range(0, len(values), self.batch_size):
                prepared = [f'{kind}: {text}' for text in values[offset:offset + self.batch_size]]
                hidden, mask = self._batch('embedding', prepared, offset, np, tokenizer, session)
                if hidden.shape != (*mask.shape, EMBEDDING_DIMENSION):
                    raise ValueError('Embedding output must have shape (batch, tokens, 384)')
                masked = np.where(mask[:, :, None].astype(bool), hidden, 0.0)
                pooled = masked.sum(axis=1, dtype=np.float64) / mask.sum(axis=1)[:, None]
                norm = np.linalg.norm(pooled, axis=1, keepdims=True)
                if not np.isfinite(norm).all() or np.any(norm <= 0):
                    raise ValueError('Embedding mean pooling produced an invalid or zero norm')
                results.append((pooled / norm).astype(np.float32))
            self._check_files()
            return np.concatenate(results, axis=0) if results else np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    def rerank(self, query, texts):
        self._texts([query])
        values = self._texts(texts)
        with self._lock:
            np, tokenizer, session = self._load('reranker')
            self._stats['reranker']['calls'] += 1
            self._stats['reranker']['last_truncated_indices'] = []
            results = []
            for offset in range(0, len(values), self.batch_size):
                pairs = [(query, text) for text in values[offset:offset + self.batch_size]]
                logits, _ = self._batch('reranker', pairs, offset, np, tokenizer, session)
                if logits.shape != (len(pairs), 1):
                    raise ValueError('Reranker output must have shape (batch, 1)')
                results.append(logits[:, 0].astype(np.float32))
            self._check_files()
            return np.concatenate(results) if results else np.empty((0,), dtype=np.float32)

    def statistics(self):
        with self._lock:
            return copy.deepcopy(self._stats)

    def fingerprint(self):
        with self._lock:
            self._check_files()
            versions = {}
            for name in ('numpy', 'onnxruntime', 'tokenizers'):
                try:
                    versions[name] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    versions[name] = None
            return {'schema_version': 1, 'models': copy.deepcopy(MODEL_SPECS),
                    'implementation_sha256': _digest(Path(__file__)),
                    'runtime': {'versions': versions, 'python': platform.python_version(), 'platform': platform.system(),
                                'machine': platform.machine(), 'provider': 'CPUExecutionProvider',
                                'intra_op_threads': self.threads, 'inter_op_threads': 1, 'execution': 'sequential'},
                    'tokenization': {'max_tokens': MAX_TOKENS, 'truncation': 'longest_first_right',
                                     'padding': 'longest_in_batch_right', 'add_special_tokens': True,
                                     'max_input_characters': MAX_TEXT_CHARS, 'batch_size': self.batch_size,
                                     'embedding_prefixes': {'query': 'query: ', 'passage': 'passage: '},
                                     'reranker_input': 'unprefixed_query_document_pair'},
                    'embedding': {'dimension': EMBEDDING_DIMENSION, 'pooling': 'attention_masked_mean', 'normalization': 'L2'},
                    'reranker': {'score': 'raw_logit', 'activation': 'identity', 'calibrated_probability': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'verify'))
    parser.add_argument('--cache-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = download_models(args.cache_dir, progress=lambda row: print(json.dumps(row, ensure_ascii=True), flush=True)) if args.command == 'download' else ModelBundle(args.cache_dir).fingerprint()
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
