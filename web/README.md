# SafePostTrain learning website

This directory is a standalone, buildless static website. Serve `dist/` over HTTP.

The curriculum sources are `course.mjs`, `deep-dive.mjs` and `applied-practice.mjs`. After edits, run `node build.mjs` to regenerate `dist/content/course.json`. In the full SafePostTrain-Lab project, run `node scripts/build.mjs` from the project root to also update the GitHub-readable course files and downloadable teaching materials.

No dependencies, API keys, server-side storage, analytics or remote fonts are required. Learning progress is stored in the user's browser and can be exported from “我的学习”.

The training workflow requires a CUDA GPU: only cuda or cuda:N is accepted, and FP32 is the initial precision. The browser does not launch training. The practice page reads `environment.device` from `content/tiny-lm-report.json` and labels the actual hardware, precision, timing and memory scope. The page validates the report device before caching or rendering it; unsupported device reports are rejected. The primary report is generated from `reports/tiny_lm/gpu_seed7.json`, measured on an NVIDIA GeForce RTX 5070 with PyTorch 2.9.1+cu128 and FP32. Training fails when the requested CUDA device is unavailable.

Install a CUDA build using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then run `python -m labs.tiny_lm.run --check-device` from the project root. Use `labs/tiny_lm/requirements-gpu.txt` or the GPU lock file for dependencies. GPU-specific course and UI source edits require a fresh build before distribution and a separate authorized deployment before they appear on the hosted site.

`.openai/hosting.json` preserves the private Site identity. The deployment source repository is separate from the user's GitHub project repository.
