# Historical CPU source snapshot

These five Python files are the unchanged TinyLM sources from local Git commit
`2caec47`, before CUDA support was added. Their SHA256 hashes match
`reports/tiny_lm/verified_seed7.json` exactly. This snapshot preserves the
provenance of the measured CPU reference, not the current GPU implementation.

From the repository root, audit the historical record with:

```bash
python3 labs/evaluate_measured.py \
  --report reports/tiny_lm/verified_seed7.json \
  --source-dir reports/tiny_lm/cpu_reference_source
```

Current training sources are in `labs/tiny_lm/`; current CUDA evidence is in
`reports/tiny_lm/gpu_seed7.json`. Do not replace historical hashes with hashes
of the new code or relabel CPU measurements as GPU measurements.
