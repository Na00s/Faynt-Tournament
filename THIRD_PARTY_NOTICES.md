# Third-party source and runtime notices

## License scope

The repository's [MIT License](LICENSE) applies to Faynt-owned code and accompanying documentation, copyright 2026 Frisson Labs. Third-party portions retain the notices and licenses below.

| Component | License and notice |
|---|---|
| Faynt-owned source, scripts and documentation | [MIT](LICENSE) |
| Slippi-AI-derived or adapted portions | MIT, retaining [Vlad Firoiu's copyright and permission notice](licenses/slippi-ai-MIT.txt) |
| MIMIC-derived or adapted portions | MIT, retaining [Erick Martinez's copyright and permission notice](licenses/MIMIC-MIT.txt) |
| `patches/slippi-dolphin-two-pipe-frame-sync.patch`, including its modifications | GPL-2.0-or-later, with the [GNU GPL version 2 text](licenses/Slippi-Dolphin-GPL-2.0.txt) and upstream notices below |
| Separately supplied dependency source, model weights, datasets, game images and emulator binaries | The respective providers' terms apply to those inputs |

## Slippi Dolphin frame synchronization patch

`patches/slippi-dolphin-two-pipe-frame-sync.patch` modifies Slippi Dolphin's controller-pipe synchronization. It retains upstream source context from [project-slippi/Ishiiruka](https://github.com/project-slippi/Ishiiruka) at commit `e7711b104b339a99385f2bb12b472d46140a7bc7`.

The affected upstream files carry these notices:

- `Source/Core/InputCommon/ControllerInterface/ControllerInterface.cpp`: Copyright 2010 Dolphin Emulator Project.
- `Source/Core/InputCommon/ControllerInterface/Pipes/Pipes.cpp`: Copyright 2015 Dolphin Emulator Project.
- `Source/Core/InputCommon/ControllerInterface/Pipes/Pipes.h`: Copyright 2015 Dolphin Emulator Project.

All three files declare GPLv2+. The corresponding [GPL version 2 license text](licenses/Slippi-Dolphin-GPL-2.0.txt) is included verbatim from the pinned upstream repository. This GPLv2-or-later notice applies to the source patch and its modifications. The patch adds a two-controller frame synchronization transaction while preserving the pinned upstream input interface. Its recorded SHA-256 is `bb0e8885b33e6f3bb5a43e5ef936fce4a0b79459cde926a97d02aed9a7fd388e`.

This export contains source for the patch. A matching emulator binary is a separately supplied local input.

## External policy dependencies

The runtime imports user-provided source checkouts for:

- [MIMIC](https://github.com/erickfm/MIMIC), whose pinned source includes an MIT license with Copyright 2025 Erick Martinez.
- [Slippi-AI](https://github.com/vladfi1/slippi-ai), whose pinned source includes an MIT license with Copyright 2020 Vlad Firoiu.

Their source trees, checkpoints, and normalization assets are supplied separately. Source licenses and model or dataset terms should be retained with those inputs. Dependency versions and exact source revisions are recorded in the configuration and runtime manifests.

## Faynt source bundle

`sources/faynt/268031e7bddebb4e8c7a40026cd0b95f824d9d0d/` contains the original Faynt model, controller codec, and tensor batching files used by the historical benchmark loader. The adjacent manifest records the source revision, runtime revision, and exact per-file hashes. These files preserve their original bytes; the upstream-inspired portions retain the notices described below.

The Faynt state representation and controller codec follow Slippi-AI at its pinned revision. The complete MIT copyright and permission notices are retained in `licenses/slippi-ai-MIT.txt` and `licenses/MIMIC-MIT.txt`.
