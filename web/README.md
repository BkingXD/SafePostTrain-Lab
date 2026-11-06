# SafePostTrain learning website

This directory is a standalone, buildless static website. Serve `dist/` over HTTP.

The curriculum source is `course.mjs`. After edits, run `node build.mjs` to regenerate `dist/content/course.json`. In the full SafePostTrain-Lab project, run `node scripts/build.mjs` from the project root to also update the GitHub-readable course files and downloadable teaching materials.

No dependencies, API keys, server-side storage, analytics or remote fonts are required. Learning progress is stored in the user's browser and can be exported from “我的学习”.

`.openai/hosting.json` preserves the private Site identity. The deployment source repository is separate from the user's GitHub project repository.
