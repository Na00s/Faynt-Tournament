# Faynt Tournament

Run local Melee games with Faynt, MIMIC, Slippi-AI, or the game's level 9 CPU. The runtime keeps each policy's observation, controller decoding, and delay contract, coordinates both controller ports at every frame, and retains replay and controller evidence.

The tournament scheduler runs mirrored MIMIC versus Slippi-AI Fox matches. The companion [Faynt Benchmarks](https://github.com/Na00s/Faynt-Benchmarks) repository provides multi-character Faynt schedules and cloud execution. The source manifest records the research revision used for this runtime.

## Local inputs

Use Python 3.12. The historical environments are described by `requirements-e001.lock` for native MIMIC mirrors and `requirements-e010.lock` for Slippi-AI and Faynt matches. Prepare those environments separately as `.e001-env` and `.e010-env`. Exact environment identities form part of the recorded result.

Configure [configs/integration.toml](configs/integration.toml) before running a game. Paths are local input locations. Supply:

- Your Melee NTSC 1.02 image, selected with `--iso-path` or `MELEE_ISO_PATH`.
- The configured frame-synchronized Slippi Dolphin application. The configuration pins the original macOS build, source revision, executable, and application-tree identities. The source patch is retained under `patches/`.
- MIMIC's pinned source checkout and complete native checkpoint bundle when selecting MIMIC. The [bundle manifest](src/melee_policy/integration/mimic_native_bundles.v1.json) lists bundle identities and asset hashes.
- Slippi-AI's pinned source checkout and the selected released checkpoint when selecting Slippi-AI. Faynt also uses its pinned game-state parser.
- A supported Faynt checkpoint. The exact model, controller codec, and tensor batching source files are bundled under `sources/faynt/`, with their original revision and SHA-256 manifest. The loader verifies those hashes before importing them.
- The pinned raw replay canary for the cross-runtime tournament input check.

Upstream source trees, weights, normalization assets, raw replays, game images, and emulator binaries are supplied separately. Runtime code performs no asset downloads or dependency installation. The configurations retain exact provenance values so changing an input remains visible to the integrity checks.

## Compare a policy with Faynt

From this directory, inspect the existing match interface:

```bash
./scripts/play --help
```

`faynt` is a public alias for the historical `frisson-ai` runtime. For example, with the released 10M Base checkpoint and matching local prerequisites:

```bash
./scripts/play \
  --config configs/integration.toml \
  --iso-path /path/to/melee-ntsc-1.02.iso \
  --p1 faynt --p2 mimic \
  --p1-checkpoint /path/to/Faynt-10M-Base/checkpoint.pt \
  --p1-character FOX --p2-character FOX \
  --stage FINAL_DESTINATION --seed 0 \
  --require-natural-end --save-slp \
  --artifact-label faynt_10m_base_vs_mimic_fox
```

The equivalent 75M Base comparison uses `Faynt-75M-Base/checkpoint.pt`. Use a new artifact label for each game. Faynt occupies P1 in the existing Faynt match runners. Select `--p2 slippi-ai` for the pinned Slippi-AI release, or `--p2 cpu --cpu-level 9` for CPU evaluation. The same named flags expose the existing single-game interfaces; each runner validates its supported characters and checkpoint contract before gameplay.

The Faynt family has three stages: Base is pretrained, Expert adds supervised curricula and 10M distillation, and Arena adds reinforcement learning. Checkpoint formats determine the loader. Base and Expert `.pt` files use historical training envelopes. The final Arena expanded schedules use recorded benchmark envelopes and native-checkpoint attestations. Their `d0` names describe Faynt's zero added delay; those public Slippi-AI opponents retain 21- or 24-frame queues. The companion benchmark repository's `evaluation/` runtime loads the native `checkpoint.pt` files for all six Base, Expert, and Arena models. Transformers `model.safetensors` packages use the model-card inference interface. Follow the companion Arena queue's envelope preparation when reproducing its historical schedule.

The local match runtime uses a continuous 256-frame ring cache, FP32, temperature 1, next-frame action alignment, and zero added Faynt delay. It resets the cache at the start of each game. The model's trained sequence length is also 256; the original actor's 128-frame trajectory-context setting is separate from the cache capacity. These settings form part of the checkpoint and benchmark contracts.

## Baseline tournament

The standard tournament scheduler registers MIMIC's Fox checkpoint and Slippi-AI `medium-v2` for Fox mirrors. Faynt comparisons use the single-game and benchmark runners described above. Integrating a new policy requires a source adapter and its identity, controller, state-reset, and timing checks. Its default schedule runs the matchup in both controller-port assignments:

```bash
./scripts/tournament \
  --config configs/integration.toml \
  --iso-path /path/to/melee-ntsc-1.02.iso \
  --seed 42 --stage BATTLEFIELD --games-per-block 2 \
  --order-seed 20260824 \
  --report artifacts/integration/tournament/local-smoke.json
```

Repeat `--seed` and `--stage` to expand the schedule. `--games-per-block` is even. The runner pairs policy-sampling seeds across ports, randomizes execution order, and records uncontrolled game RNG. Running the same command revalidates eligible existing evidence before resuming on the same host.

Every scheduled game runs in a fresh process. Accepted results require the declared source, checkpoint, environment, game, controller, and replay identities. A conclusive draw earns half a point. Failed or rejected attempts invalidate the report. Reports include per-port results, matched blocks and descriptive Wilson intervals. The software does not yet implement the planned paired statistical analysis, so `result_claim_ready` remains false.

## Validation

```bash
PYTHONPATH=src .e010-env/bin/python -m pytest -q tests/integration
```

The included tests use synthetic game states and temporary files. Local-checkpoint and raw-replay tests skip when their external inputs are absent. The release validation records test totals, source identities, exclusions, and command parsing separately. A full emulator evaluation requires the declared local inputs and its own run evidence.

## Companion suite

The [Faynt Benchmarks](https://github.com/Na00s/Faynt-Benchmarks) repository includes the initial and expanded schedules, cloud queue, frozen result records, and later Phillip and private zero-delay execution runtime. Its evaluation runtime loads all six published native Faynt checkpoints.

## License

Faynt-owned code and accompanying documentation are available under the [MIT License](LICENSE), copyright 2026 Frisson Labs. Third-party portions retain their original notices. The Slippi Dolphin source patch is licensed under GPL-2.0-or-later. See the [component licensing and runtime notices](THIRD_PARTY_NOTICES.md) for the scope and retained license texts.
