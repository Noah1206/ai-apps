# TASK-005 reproducibility contract

TASK-005 recovery is **checksum-complete**. The slim GPU bundle was imported under
`recovered/task-005-recovery-gate2-slim-20260808/`; all 125 indexed files and all required
artifacts in `task-005-reproducibility-spec.json` passed SHA-256 verification. The Mac still
does not provide the NVIDIA runtime required to execute training or inference.

The run identity is:

```text
base model: nvidia/nemotron-3.5-asr-streaming-0.6b
model revision: f3d333391852ba876df169dcc9ba902d25b6ab0b
NeMo revision: 6c57e73e83de967eed4d334c493ac313b9afd147
train ZIP SHA-256: fcef88360c12d4149a290fa0ff93546fd2cb1bf7bd9d6f9c177557c27e22e20d
validation ZIP SHA-256: 8253af4979ce5f9e20aed9bbb9023ee9f03f3797278e53beabdb90468f825f0d
benchmark ZIP SHA-256: 151c1e28804627bea69bbd7f6632f4d3558ebf076147e42c1d168d508467233c
selection: minimum validation loss before opening benchmark
selected checkpoint: epoch 5, global step 97, val_loss 42.027042
best checkpoint SHA-256: a05b3fcc919d18638d87f69700e1e7fba9b2f5f8b2d20396226b2aae9d13343e
best exported .nemo SHA-256: 580124ff0ea5c9e2f5546e9186c93c3c2d9e16641a749d02b219e2d06029f950
best adapter tensor-content SHA-256: f2c17a1c2bfdb8cf9f24ad79f28b6c3379c48ce1f3067312bb09e4b0eb41ff36
recovery archive SHA-256: f41cf5aca71efb2bb1c261ce8ab41ceae5e6795368d1713e82600ecdb30b7f0a
```

Re-execution requires all of the following together:

1. the pinned pretrained model ID/revision above (the redundant base `.nemo` was omitted
   from the slim bundle; its expected SHA-256 remains pinned);
2. the checksum-pinned Train and Validation ZIPs;
3. the exact `hparams.yaml`, launcher, patched NeMo entry point and patch;
4. the recorded Python/PyTorch/CUDA/NeMo/Lightning/Hydra environment;
5. the same authorization gate and independent Validation manifest;
6. the selected `.ckpt`, its exported best `.nemo`, and selected adapter-only state for
   result verification;
7. TensorBoard, console, command, timing, and GPU telemetry logs.

`task-005-reproducibility-spec.json` is the source of truth. Verify it from the project root:

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python -m busan_lab.gate2 verify-artifacts \
  --root . \
  --spec artifacts/gate-2/reproducibility/task-005-reproducibility-spec.json \
  --output artifacts/gate-2/reproducibility/current-verification.json
```

The selected `.ckpt`, exported best `.nemo`, embedded 96-tensor adapter state, hparams,
launcher, patched training entry point, inference runner, environment snapshot, dependency
lock, TensorBoard events and both console logs now pass. The adapter tensor-content SHA-256
is `f2c17a1c2bfdb8cf9f24ad79f28b6c3379c48ce1f3067312bb09e4b0eb41ff36`.

Verify the entire imported bundle, including archive safety and its 125-file index:

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python -m busan_lab.gate2 verify-recovery-bundle \
  --bundle-root artifacts/gate-2/reproducibility/recovered/task-005-recovery-gate2-slim-20260808 \
  --source-archive /path/to/task-005-recovery-gate2-slim-20260808.tar.gz \
  --output artifacts/gate-2/reproducibility/recovery-bundle-verification.json
```

`current-verification.json` and `recovery-bundle-verification.json` are the current reports.
The base `.nemo` can be restored only from the pinned provider revision and must match
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74` before retraining.
