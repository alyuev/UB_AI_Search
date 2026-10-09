"""ONNX embedding models: registry, loading, and query/passage embedding."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import MODELS_DIR, ROOT
from app.log import get_logger

logger = get_logger("app")

GGUF_DIR = MODELS_DIR / "gguf"
LLAMACPP_BIN_DIR = ROOT / "tools" / "llama.cpp" / "bin"

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

# GGUF variants runnable through a local llama-server (Vulkan backend on this
# machine: uses the GPU's *compute* queue, unlike onnxruntime-directml, which
# uses the 3D/graphics queue and visibly lags the desktop). `verified_cosine`
# is the measured cosine similarity between this GGUF's embeddings and the
# corresponding ONNX model's embeddings on a handful of real EN passages —
# only entries with verified_cosine >= 0.98 are considered safe substitutes
# for the ONNX path; the others are kept here for transparency/benchmarking
# but excluded from "auto" selection.
LLAMACPP_REGISTRY = {
    "multilingual-e5-small": {
        "gguf_file": "multilingual-e5-small-q8_0-v2.gguf",
        "pooling": "mean",
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "verified_cosine": 0.79,  # community BERT->GGUF conversion, unreliable
        "quality_ok": False,
        "not_recommended_reason": (
            "GGUF conversion quality failed the cosine>=0.98 gate (0.77-0.82 vs ONNX, weak "
            "query/passage discrimination too: 0.915 vs 0.899). A second community source "
            "(milimyname/...-Q8_0-GGUF) crashed at load entirely (unsupported ggml op types). "
            "Dropped from the subset benchmark; use the ONNX/CPU path if this model is needed."
        ),
    },
    "multilingual-e5-base": {
        "gguf_file": "multilingual-e5-base-q8_0.gguf",
        "pooling": "mean",
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        # Different conversion source (dinab/multilingual-e5-base-Q8_0-GGUF) than
        # the e5-small GGUF above (keisuke-miyako/...); this one is a good
        # conversion — verified cosine 0.983-0.988 vs ONNX on 3 sample texts.
        "verified_cosine": 0.983,
        "quality_ok": True,
    },
    "bge-m3": {
        "gguf_file": "bge-m3-q8_0.gguf",
        "pooling": "cls",
        "prefix_query": "",
        "prefix_passage": "",
        "verified_cosine": 0.985,
        "quality_ok": True,
    },
    "qwen3-embedding-0.6b": {
        "gguf_file": "qwen3-embedding-0.6b-q8_0.gguf",
        "pooling": "last",
        "prefix_query": QUERY_INSTRUCTION,
        "prefix_passage": "",
        "verified_cosine": 0.85,  # below the 0.98 bar; root cause not fully isolated
        "quality_ok": False,
        "not_recommended_reason": (
            "GGUF (official Qwen/Qwen3-Embedding-0.6B-GGUF Q8_0) vs our ONNX path cosine only "
            "0.78-0.90, below the 0.98 gate. Root cause not fully isolated (bge-m3 passing "
            "proves the test harness itself is sound, so this may be a subtlety in our custom "
            "ONNX decoder-forward workaround rather than llama.cpp being wrong). Also the "
            "slowest model on every backend. Dropped from the subset benchmark."
        ),
    },
}

# Models actually benchmarked/shipped via llama.cpp-Vulkan (passed the
# cosine>=0.98 quality gate). Kept separate from LLAMACPP_REGISTRY (which
# still documents every model we *tried*, including the dropped ones above)
# so callers have one place to ask "what should we actually compare".
BENCHMARK_CANDIDATES = ["bge-m3", "multilingual-e5-base"]

# Registry of models we support. `repo` is the HF repo holding ONNX exports
# (Xenova/* re-exports with tokenizer.json included), `onnx_file` is the
# quantized weights file we download, `dim` the embedding dimension.
MODEL_REGISTRY = {
    "multilingual-e5-small": {
        "repo": "Xenova/multilingual-e5-small",
        "onnx_file": "onnx/model_quantized.onnx",
        "dim": 384,
        "max_tokens": 512,
        "kind": "e5",
        "description": "intfloat multilingual-e5-small, 384d, quantized ONNX",
    },
    "multilingual-e5-base": {
        "repo": "Xenova/multilingual-e5-base",
        "onnx_file": "onnx/model_quantized.onnx",
        "dim": 768,
        "max_tokens": 512,
        "kind": "e5",
        "description": "intfloat multilingual-e5-base, 768d, quantized ONNX",
    },
    "bge-m3": {
        "repo": "Xenova/bge-m3",
        "onnx_file": "onnx/model_quantized.onnx",
        "dim": 1024,
        "max_tokens": 8192,
        "kind": "bge",
        "description": "BAAI/bge-m3, 1024d, quantized ONNX, long context",
    },
    "qwen3-embedding-0.6b": {
        "repo": "onnx-community/Qwen3-Embedding-0.6B-ONNX",
        "onnx_file": "onnx/model_int8.onnx",
        "dim": 1024,
        "max_tokens": 32768,
        "kind": "qwen3",
        "description": "Qwen3-Embedding-0.6B, 1024d (MRL-truncatable), int8 ONNX, instruct-prefixed queries",
    },
}

def _model_dir(name: str) -> Path:
    return MODELS_DIR / name


def _model_files_present(name: str) -> bool:
    d = _model_dir(name)
    return (d / "model.onnx").exists() and (d / "tokenizer.json").exists()


DEVICE_INFO = {
    "cpu": {
        "label": "CPU",
        "description": "Runs on CPU cores only. Default and recommended for indexing and search.",
    },
    "gpu": {
        "label": "GPU (DirectML, shared with desktop)",
        "description": (
            "Uses the integrated Radeon 890M via DirectML. This iGPU is shared with the "
            "Windows desktop compositor, and onnxruntime's DirectML backend schedules work "
            "on the GPU's 3D/graphics queue (not a separate compute queue), so heavy use can "
            "make the desktop UI feel laggy while it runs. Opt in only if you accept that "
            "trade-off; CPU is usually fast enough for this corpus size."
        ),
    },
    "npu": {
        "label": "NPU (not available)",
        "description": (
            "Not supported in this build: onnxruntime-directml's DirectML EP does not accept "
            "device_filter='npu' here, and real XDNA2 NPU offload needs the separate AMD "
            "Ryzen AI / Vitis AI SDK, which is not installed."
        ),
    },
}


def list_runtimes() -> list[dict]:
    """List embedding runtimes and whether each is actually usable right now
    on this machine, for a settings UI. `name` values match what
    `create_embedder(model, runtime=...)` and `build_index(..., runtime=...)`
    accept, except "onnx-cpu"/"onnx-dml" both map to runtime="onnx" with
    device="cpu"/"gpu" respectively (ONNX has one runtime, two devices).
    """
    try:  # not shipped in dist (llama.cpp/Vulkan is the runtime there)
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        onnx_ok = True
    except ImportError:
        available_providers, onnx_ok = [], False
    llama_bin_present = (LLAMACPP_BIN_DIR / "llama-server.exe").exists()

    return [
        {
            "name": "onnx-cpu",
            "runtime": "onnx",
            "device": "cpu",
            "available": onnx_ok,
            "description": "CPU (onnxruntime). Default and recommended for indexing and search.",
        },
        {
            "name": "onnx-dml",
            "runtime": "onnx",
            "device": "gpu",
            "available": "DmlExecutionProvider" in available_providers,
            "description": (
                "GPU (DirectML, нагружает 3D/интерфейс) — использует общий iGPU через очередь "
                "3D/графики, что может приводить к подтормаживанию рабочего стола при активной "
                "индексации. Явный выбор, не используется автоматически."
            ),
        },
        {
            "name": "llamacpp-vulkan",
            "runtime": "llamacpp-vulkan",
            "device": "gpu",
            "available": llama_bin_present,
            "description": (
                "GPU Vulkan (Compute) — использует очередь Compute (не 3D), не мешает "
                "интерфейсу; в 3-4 раза быстрее CPU в наших замерах. Требует "
                "tools/llama.cpp/bin/llama-server.exe и модель в models/gguf/."
            ),
        },
    ]


def model_runtime_compatibility(model_name: str) -> list[dict]:
    """For a given model, which runtimes can it actually run on, and are
    they quality-verified? ONNX is always available (if the model files are
    downloaded) and always considered correctness-verified (it's the
    reference implementation everything else was checked against).
    llamacpp-vulkan is only listed as quality_ok=True for models whose GGUF
    conversion passed the cosine>=0.98 gate (see LLAMACPP_REGISTRY).
    """
    out = [
        {
            "runtime": "onnx",
            "quality_ok": True,
            "reason": "reference implementation; always correctness-verified",
        }
    ]
    entry = LLAMACPP_REGISTRY.get(model_name)
    if entry is not None:
        out.append(
            {
                "runtime": "llamacpp-vulkan",
                "quality_ok": bool(entry.get("quality_ok")),
                "reason": entry.get("not_recommended_reason")
                or f"verified cosine {entry.get('verified_cosine')} vs ONNX (>=0.98 required)",
            }
        )
    return out


def list_available_models() -> list[dict]:
    out = []
    for name, spec in MODEL_REGISTRY.items():
        d = _model_dir(name)
        out.append(
            {
                "name": name,
                "dim": spec["dim"],
                "max_tokens": spec["max_tokens"],
                "downloaded": _model_files_present(name),
                "path": str(d),
                "description": spec["description"],
            }
        )
    return out


def download_model(name: str, force: bool = False) -> Path:
    """Download tokenizer.json + quantized ONNX weights for `name` into
    models/<name>/. Requires network access; call this from a separate
    script (bench/download_models.py), not from the offline runtime path.
    """
    from huggingface_hub import hf_hub_download

    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}")
    spec = MODEL_REGISTRY[name]
    dest = _model_dir(name)
    dest.mkdir(parents=True, exist_ok=True)

    if _model_files_present(name) and not force:
        logger.info("model %s already present at %s", name, dest)
        return dest

    logger.info("downloading %s from %s", name, spec["repo"])
    tok_path = hf_hub_download(spec["repo"], "tokenizer.json")
    onnx_path = hf_hub_download(spec["repo"], spec["onnx_file"])

    import shutil

    shutil.copyfile(tok_path, dest / "tokenizer.json")
    shutil.copyfile(onnx_path, dest / "model.onnx")

    data_file = spec.get("onnx_data_file")
    if data_file:
        data_path = hf_hub_download(spec["repo"], data_file)
        shutil.copyfile(data_path, dest / "model.onnx_data")

    # config.json is convenient but optional
    try:
        cfg_path = hf_hub_download(spec["repo"], "config.json")
        shutil.copyfile(cfg_path, dest / "config.json")
    except Exception:
        pass

    logger.info("model %s downloaded to %s", name, dest)
    return dest


def _mean_pool(last_hidden: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
    mask = attn_mask[:, :, None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def _last_token_pool(last_hidden: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
    """Take the embedding of the last non-padding token of each sequence
    (works regardless of left/right padding, by using the attention mask
    to find each sequence's true last position).
    """
    seq_lens = attn_mask.sum(axis=1)  # right-padding: last real token index = seq_len - 1
    # If padding were on the left, the last token is always at position -1;
    # detect that case by checking whether the first column is ever masked out.
    left_padded = bool(attn_mask[:, 0].min() == 0) if attn_mask.shape[1] > 0 else False
    if left_padded:
        return last_hidden[:, -1, :]
    idx = np.clip(seq_lens - 1, 0, last_hidden.shape[1] - 1).astype(np.int64)
    return last_hidden[np.arange(last_hidden.shape[0]), idx, :]


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vecs / norms


class Embedder:
    """Loads one ONNX embedding model and exposes embed_query/embed_passages."""

    def __init__(self, model_name: str, device: str = "auto", intra_op_threads: Optional[int] = None):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}")
        self.model_name = model_name
        self.runtime = "onnx"
        self.spec = MODEL_REGISTRY[model_name]
        self.dim = self.spec["dim"]
        self.max_tokens = self.spec["max_tokens"]
        self.kind = self.spec["kind"]
        self.status = "loading"

        model_dir = _model_dir(model_name)
        onnx_path = model_dir / "model.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not onnx_path.exists() or not tok_path.exists():
            self.status = "missing"
            raise FileNotFoundError(
                f"model files for {model_name!r} not found in {model_dir}; "
                f"run bench/download_models.py first"
            )

        # Ensure fully offline at runtime (downloads happen only via
        # download_model()/bench/download_models.py).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        # Qwen3 has a much larger nominal context (32k) than we need; cap it to
        # keep benchmark/index runs fast, still far above our paragraph lengths.
        # Cap very long contexts (bge-m3: 8192, qwen3: 32768) for benchmark/index
        # speed; our paragraphs are short and never need the full context.
        if self.kind in ("bge", "qwen3"):
            effective_max = min(self.max_tokens, 1024)
        else:
            effective_max = self.max_tokens
        self.tokenizer.enable_truncation(max_length=effective_max)
        pad_token = "<|endoftext|>" if self.kind == "qwen3" else "[PAD]"
        try:
            pad_id = self.tokenizer.token_to_id(pad_token) or 0
        except Exception:
            pad_id = 0
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        self.session, self.provider = self._load_session(onnx_path, device, intra_op_threads)
        self._kv_cache_spec = self._detect_kv_cache_spec()
        self.status = f"ready ({self.provider})"

    def _detect_kv_cache_spec(self) -> Optional[tuple[int, int]]:
        """If this ONNX graph is a decoder export requiring past_key_values
        inputs (e.g. Qwen3), return (n_kv_heads, head_dim) read from the
        first such input's static shape dims; else None.
        """
        for inp in self.session.get_inputs():
            if inp.name.startswith("past_key_values."):
                shape = inp.shape
                # shape: [batch_size, n_kv_heads, past_sequence_length, head_dim]
                n_kv_heads = shape[1] if isinstance(shape[1], int) else 8
                head_dim = shape[3] if isinstance(shape[3], int) else 128
                return (n_kv_heads, head_dim)
        return None

    def _load_session(self, onnx_path: Path, device: str, intra_op_threads: Optional[int] = None):
        import onnxruntime as ort

        available = ort.get_available_providers()

        # Device policy: "auto" is deliberately CPU-only. onnxruntime-directml's
        # DML EP creates a D3D12 *direct* (3D/graphics) command queue on this
        # iGPU (Radeon 890M) rather than a compute queue, so any DML work shows
        # up on the shared "3D" engine in Task Manager and visibly lags the
        # desktop UI. "gpu" remains available as an explicit, informed opt-in
        # (see MODEL_REGISTRY-adjacent docs / list_available_models callers for
        # the UI-facing warning). "npu" is NOT offered: this onnxruntime-directml
        # build's DML EP rejects device_filter="npu" outright
        # ("Invalid Filter provided for DirectML EP device selection" — only
        # "gpu"/"any"-style enum values compiled in are accepted), and the
        # AMD Vitis AI / Ryzen AI EP needed for real NPU offload requires a
        # separate, large SDK install we were told not to pull in silently.
        if device == "cpu" or device == "auto":
            providers = ["CPUExecutionProvider"]
        elif device == "gpu":
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif device == "npu":
            raise ValueError(
                "device='npu' is not supported: this onnxruntime-directml build's "
                "DmlExecutionProvider rejects device_filter='npu', and the Vitis AI / "
                "Ryzen AI EP for real NPU offload requires a separate SDK that was not installed"
            )
        else:
            raise ValueError(f"unknown device {device!r}")
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            physical_cores = os.cpu_count() // 2 if os.cpu_count() else 4
            physical_cores = max(1, physical_cores or 4)
            if intra_op_threads is not None:
                threads = max(1, intra_op_threads)
            else:
                # Leave headroom for the desktop UI during indexing by default.
                threads = max(1, physical_cores - 2)
            so.intra_op_num_threads = threads
        except Exception:
            pass
        try:
            session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
            used = session.get_providers()[0]
        except Exception as e:
            logger.warning("failed to load with providers %s (%s); falling back to CPU", providers, e)
            session = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            used = "CPUExecutionProvider"
        return session, used

    def _prefix(self, texts: list[str], is_query: bool) -> list[str]:
        if self.kind == "e5":
            p = "query: " if is_query else "passage: "
            return [p + t for t in texts]
        if self.kind == "qwen3" and is_query:
            return [QUERY_INSTRUCTION + t for t in texts]
        return list(texts)

    def _run(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        inputs = {"input_ids": ids, "attention_mask": mask}
        input_names = {i.name for i in self.session.get_inputs()}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(ids)
        if "position_ids" in input_names:
            inputs["position_ids"] = np.clip(np.cumsum(mask, axis=1) - 1, 0, None).astype(np.int64)

        # Decoder-style export (Qwen3): needs an (empty, on the first pass)
        # KV cache per layer even though we only want the encoder-like forward
        # pass over the full prompt in one shot.
        kv_spec = self._kv_cache_spec
        if kv_spec is not None:
            batch = ids.shape[0]
            n_kv_heads, head_dim = kv_spec
            empty = np.zeros((batch, n_kv_heads, 0, head_dim), dtype=np.float32)
            for name in input_names:
                if name.startswith("past_key_values."):
                    inputs[name] = empty

        inputs = {k: v for k, v in inputs.items() if k in input_names}

        outputs = self.session.run(None, inputs)
        output_names = [o.name for o in self.session.get_outputs()]
        last_hidden = outputs[output_names.index("last_hidden_state")] if "last_hidden_state" in output_names else outputs[0]

        if self.kind == "bge":
            pooled = last_hidden[:, 0, :]  # CLS pooling
        elif self.kind == "qwen3":
            pooled = _last_token_pool(last_hidden, mask)
        else:
            pooled = _mean_pool(last_hidden, mask)

        return _normalize(pooled.astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        texts = self._prefix([text], is_query=True)
        vecs = self._run(texts)
        return vecs[0]

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        prefixed = self._prefix(texts, is_query=False)
        if not prefixed:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Sort by (approximate) length before batching so each batch pads to
        # a shorter common length, then restore the original order. This is
        # a pure perf optimization; results are identical to unsorted batching.
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
        sorted_texts = [prefixed[i] for i in order]

        chunks = []
        for i in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[i : i + batch_size]
            chunks.append(self._run(batch))
        stacked = np.concatenate(chunks, axis=0)

        out = np.empty_like(stacked)
        out[order] = stacked
        return out


class LlamaCppEmbedder:
    """Runs a GGUF embedding model through a local `llama-server` subprocess
    (Vulkan backend on this machine), talking to its OpenAI-compatible
    /v1/embeddings endpoint. Exposes the same interface as Embedder
    (.dim, .status, .embed_query, .embed_passages) so search.py/indexer.py
    don't need to care which runtime produced the vectors.

    Vulkan matters here specifically because onnxruntime-directml's DirectML
    EP creates a D3D12 *3D/graphics* command queue on this shared iGPU
    (Radeon 890M), which visibly lags the desktop compositor; llama.cpp's
    Vulkan backend instead dispatches through a *compute* queue (confirmed
    via `Get-Counter '\\GPU Engine(*)\\Utilization Percentage'`, which shows
    load on `engtype_Compute 0`, not `engtype_3D`, while this class runs).
    """

    def __init__(self, model_name: str, port: Optional[int] = None, startup_timeout: float = 60.0):
        import socket
        import subprocess

        if model_name not in LLAMACPP_REGISTRY:
            raise ValueError(f"no llama.cpp/GGUF entry for {model_name!r}")
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}")

        self.model_name = model_name
        self.runtime = "llamacpp-vulkan"
        self.spec = LLAMACPP_REGISTRY[model_name]
        self.dim = MODEL_REGISTRY[model_name]["dim"]
        self.status = "loading"
        self._client = None  # persistent keep-alive httpx.Client, created on first use
        self._tokenizer = None  # lazy: HF tokenizer, used to truncate to this model's true limit
        self.max_tokens = MODEL_REGISTRY[model_name]["max_tokens"]

        gguf_path = GGUF_DIR / self.spec["gguf_file"]
        server_bin = LLAMACPP_BIN_DIR / "llama-server.exe"
        if not gguf_path.exists():
            self.status = "missing"
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        if not server_bin.exists():
            self.status = "missing"
            raise FileNotFoundError(
                f"llama-server binary not found: {server_bin}; "
                "download the Vulkan release into tools/llama.cpp/bin/"
            )

        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"

        cmd = [
            str(server_bin), "-m", str(gguf_path),
            "--embedding", "--pooling", self.spec["pooling"],
            "-ngl", "99", "-ub", "2048", "-b", "2048", "-c", "8192", "-np", "1",
            "--port", str(port), "--host", "127.0.0.1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT)
        )
        try:
            self._wait_ready(startup_timeout)
        except Exception:
            self.close()
            raise
        self.status = "ready (llamacpp-vulkan)"

        import atexit

        atexit.register(self.close)

    def _wait_ready(self, timeout: float) -> None:
        import time as _time

        import httpx

        deadline = _time.time() + timeout
        last_err = None
        while _time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self._proc.returncode})")
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            _time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become ready within {timeout}s: {last_err}")

    def _get_client(self):
        if self._client is None:
            import httpx

            # A persistent client with keep-alive matters here: the app
            # keeps this server warm for the whole session and embeds one
            # query at a time, so avoiding a fresh TCP/HTTP handshake per
            # query search meaningfully cuts per-query latency.
            self._client = httpx.Client(timeout=120)
        return self._client

    def _get_tokenizer(self):
        """Lazily load this model's HF tokenizer (already present in
        models/<name>/tokenizer.json from the ONNX download) so we can
        truncate text to the model's true token limit before sending it to
        llama-server — cheaper and more accurate than guessing a character
        cap, and avoids most 400 (context overflow) errors before they
        happen rather than only reacting to them.
        """
        if self._tokenizer is False:
            return None
        if self._tokenizer is None:
            tok_path = _model_dir(self.model_name) / "tokenizer.json"
            if not tok_path.exists():
                self._tokenizer = False
                return None
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(tok_path))
            except Exception as e:
                logger.warning("could not load tokenizer for %s truncation safeguard: %s", self.model_name, e)
                self._tokenizer = False
                return None
        return self._tokenizer

    def _truncate_to_limit(self, text: str, margin: int = 16, max_limit: Optional[int] = None) -> str:
        """Truncate `text` to at most (max_tokens - margin) tokens using the
        model's own tokenizer, if available; otherwise fall back to a
        conservative character-based cap (~3 chars/token is safe for both
        Latin and Cyrillic BPE-style tokenizers). `max_limit`, if given,
        additionally caps the result below the model's own max_tokens —
        used by the last-resort single-item retry, since llama-server's
        -ub (physical batch token budget) can be smaller than a long
        model's own context (e.g. bge-m3's 8192) and a single oversized
        sequence needs to fit within it on its own.
        """
        tok = self._get_tokenizer()
        limit = max(8, self.max_tokens - margin)
        if max_limit is not None:
            limit = min(limit, max_limit)
        if tok is not None:
            ids = tok.encode(text).ids
            if len(ids) <= limit:
                return text
            truncated_ids = ids[:limit]
            return tok.decode(truncated_ids)
        char_limit = limit * 3
        return text[:char_limit]

    def _embed_batch(self, texts: list[str], _retrying: bool = False) -> np.ndarray:
        client = self._get_client()
        try:
            r = client.post(f"{self.base_url}/v1/embeddings", json={"input": texts})
            r.raise_for_status()
            data = r.json()["data"]
            vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
            return _normalize(vecs)
        except Exception as e:
            # llama-server returns 400 for some context/length problems and
            # 500 for others (e.g. cumulative batch tokens > -ub) — treat
            # both as "this batch/text was too big" and apply the same
            # bisect-then-truncate safeguard.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in (400, 500):
                raise
            # Safeguard: either one text exceeded the model's context, or the
            # whole batch's cumulative token count exceeded llama-server's
            # -ub (a single request's token budget is shared across all
            # texts in it, not just per-text). Handle both:
            if len(texts) > 1:
                # Bisect and retry each half — narrows down to (and isolates)
                # whichever text(s) actually caused the overflow, and also
                # fixes pure cumulative-batch-size overflows on their own
                # (each half needs less of the shared token budget).
                logger.warning(
                    "llama-server 400 for %s (batch of %d) — bisecting and retrying",
                    self.model_name, len(texts),
                )
                mid = len(texts) // 2
                left = self._embed_batch(texts[:mid])
                right = self._embed_batch(texts[mid:])
                return np.concatenate([left, right], axis=0)
            if not _retrying:
                # Single text still too long even alone: truncate to this
                # model's real token limit (via its own tokenizer) and retry
                # once, rather than failing the whole indexing run over a
                # handful of unusually long paragraphs.
                logger.warning(
                    "llama-server 400 for %s on a single text — truncating and retrying",
                    self.model_name,
                )
                return self._embed_batch(
                    [self._truncate_to_limit(texts[0], margin=64, max_limit=480)], _retrying=True
                )
            raise

    def embed_query(self, text: str) -> np.ndarray:
        prefixed = (self.spec.get("prefix_query") or "") + self._truncate_to_limit(text)
        return self._embed_batch([prefixed])[0]

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.spec.get("prefix_passage") or ""
        prefixed = [prefix + self._truncate_to_limit(t) for t in texts]
        chunks = []
        for i in range(0, len(prefixed), batch_size):
            chunks.append(self._embed_batch(prefixed[i : i + batch_size]))
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_embedder(model_name: str, runtime: str = "auto", device: str = "auto", **kwargs):
    """Factory: runtime in {"auto", "onnx", "llamacpp-vulkan"}.

    "auto" picks llamacpp-vulkan only for models whose GGUF conversion has
    been quality-verified against the ONNX model (cosine >= 0.98); otherwise
    it falls back to the ONNX runtime, which is always correctness-verified.
    Explicitly passing runtime="llamacpp-vulkan" bypasses the quality gate
    (useful for benchmarking) but logs a warning if the model isn't verified.
    """
    if runtime == "auto":
        entry = LLAMACPP_REGISTRY.get(model_name)
        if entry and entry.get("quality_ok"):
            try:
                return LlamaCppEmbedder(model_name)
            except Exception as e:
                logger.warning("llamacpp-vulkan unavailable for %s (%s); falling back to ONNX", model_name, e)
        return Embedder(model_name, device=device, **kwargs)

    if runtime == "onnx":
        return Embedder(model_name, device=device, **kwargs)

    if runtime == "llamacpp-vulkan":
        entry = LLAMACPP_REGISTRY.get(model_name)
        if entry and not entry.get("quality_ok"):
            logger.warning(
                "llamacpp-vulkan for %s has NOT passed the cosine>=0.98 quality gate "
                "(measured %.2f) — use with caution", model_name, entry.get("verified_cosine") or -1,
            )
        try:
            return LlamaCppEmbedder(model_name)
        except Exception as e:
            # Automatic fallback: Vulkan may be unavailable on this machine
            # (no GPU, driver missing, binary not downloaded, port conflict...).
            # Correctness of search results matters more than a hard failure.
            logger.warning(
                "llamacpp-vulkan requested for %s but unavailable (%s); "
                "falling back to ONNX/%s", model_name, e, device,
            )
            return Embedder(model_name, device=device, **kwargs)

    raise ValueError(f"unknown runtime {runtime!r}")
