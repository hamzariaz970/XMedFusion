# XMedFusion Backend — Test Suite

## Structure

```
tests/
├── conftest.py            # Shared fixtures (image paths, KG dicts, annotation data)
├── test_unit.py           # Unit tests — individual components
├── test_integration.py    # Integration tests — components working together
├── test_system.py         # System/E2E tests — live FastAPI endpoints
└── test_nonfunctional.py  # Performance, robustness, security tests
```

---

## a) Functional Testing

### Unit Tests (`test_unit.py`) — 30 tests
No API server or LLM required.

| Class | Module | What's Tested |
|---|---|---|
| `TestValidators` | `validators.py` | KG contradiction detection, section parsing, label extraction |
| `TestExplain` | `explain.py` | Zone parsing, clinical heuristics (Pneumothorax→top, Effusion→bottom), overlay generation |
| `TestDraftRetrieval` | `draft.py` | `RetrievalAgent` init, `truncate_report`, empty-dict edge case |
| `TestVisionEncoder` | `vision.py` | BioMedCLIP image/text encoding, L2 normalization, X-ray vs cat similarity |
| `TestXRayFilter` | `xray_filter.py` | Real X-ray passes, non-X-ray rejected, corrupt path returns `(False, 0.0)` |
| `TestConfig` | `config.py` | Required fields exist, temperature in [0, 2], BASE_URL is localhost |

```bash
python -m pytest tests/test_unit.py -v
```

---

### Integration Tests (`test_integration.py`) — 10 tests
No LLM required. BioMedCLIP model must be loaded.

| Class | What's Tested |
|---|---|
| `TestVisionToExplainPipeline` | Full chain: image → BioMedCLIP features → findings → bounding box overlay |
| `TestFilterGateSynthesis` | Non-X-ray blocked before LLM runs; real X-ray proceeds to `parallel_start` |
| `TestValidatorIntegration` | KG contradiction found → repair call triggered |
| `TestAnnotationIntegrity` | All 4138 annotation entries have reports + image paths that exist on disk |

```bash
python -m pytest tests/test_integration.py -v
```

---

### System Tests (`test_system.py`) — 12 tests
> **Requires the API server to be running** for most tests. Tests auto-skip if server is offline.

```bash
# Terminal 1 — start server
cd backend
python -muvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2 — run tests
python -m pytest tests/test_system.py -v
```

| Class | Endpoint | What's Tested |
|---|---|---|
| `TestHealthEndpoint` | `GET /api/health` | 200 status, all fields present, GPU flag is bool, response < 2s |
| `TestSynthesizeValidXray` | `POST /api/synthesize-report` | Accepts PNG, streams NDJSON, `complete` chunk has `final_report` + `knowledge_graph` |
| `TestSynthesizeInvalidImage` | `POST /api/synthesize-report` | Non-X-ray returns `error` status, no `complete` chunk emitted |
| `TestAPIWithTestClient` | Both | No server needed — mocked heavy models via FastAPI `TestClient` |

---

## b) Non-Functional Testing (`test_nonfunctional.py`) — 20 tests

```bash
# Most tests run without a server; server-dependent ones auto-skip
python -m pytest tests/test_nonfunctional.py -v
```

### Performance Benchmarks
| Test | Limit |
|---|---|
| BioMedCLIP image encode | < 500 ms |
| Batch text encode (64 prompts) | < 2 s |
| X-ray filter (bouncer head) | < 500 ms |
| Top-5 retrieval from 4 138 reports | < 10 s |
| Explainability overlay generation | < 2 s |
| `GET /api/health` response | < 1 s |
| GPU memory leak (5× filter calls) | < 50 MB growth |

### Robustness / Boundary
- 1×1 pixel image, 4K image, grayscale, RGBA format
- Empty string and 500-word text encoding
- `k` larger than dataset size
- All-black image through hybrid scoring
- KG with unrecognised anatomy zone

### Security
| Test | Expected Behaviour |
|---|---|
| Upload HTML file | No 500 crash |
| Upload empty file | No 500 crash |
| Missing `file` field | `422 Unprocessable Entity` |
| CORS check | `Access-Control-Allow-Origin` header present |
| Path traversal filename (`../../etc/passwd`) | No 500 crash |
| SQL injection string in KG | Validator handles gracefully |
| `<script>` tag in report text | Validator handles gracefully |

---

## Run Everything

```bash
cd /home/rapids/Desktop/XMedFusion/XMedFusion/backend

# All tests (skips server-dependent ones if server offline)
python -m pytest tests/ -v --tb=short

# Quick smoke-test (unit only, ~2 min)
python -m pytest tests/test_unit.py -v

# With coverage report
python -m pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=term-missing -v
```

## Dependencies

```bash
pip install pytest pytest-asyncio httpx
```
