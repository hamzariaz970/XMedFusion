Deploy this folder as the source for your Hugging Face Inference Endpoint if you want remote CT inference to mirror the local 4-bit MedGemma path more closely.

Required endpoint settings:

- Custom handler entrypoint using `handler.py`
- Environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Environment variable `HF_CT_BASE_MODEL_ID=google/medgemma-4b-it`
- Optional environment variable `HF_CT_ADAPTER_DIR=<adapter folder>` if serving a LoRA adapter
- Concurrency set to `1`

This handler expects JSON shaped like:

```json
{
  "inputs": {
    "image_urls": ["data:image/jpeg;base64,..."],
    "prompt": "..."
  },
  "parameters": {
    "max_new_tokens": 256
  }
}
```
