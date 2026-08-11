# GPU PC recovery checklist

Copy files; do not move, rename in place, retrain, or overwrite the GPU originals.

Required source files are listed with their exact GPU paths in
`task-005-reproducibility-spec.json`. Recover:

- pretrained `.nemo`;
- epoch-5/global-step-97 best `.ckpt`;
- exported best `.nemo`;
- adapter-only state exported from that selected checkpoint, not the final step-400 state;
- `hparams.yaml`;
- exact training launcher and benchmark inference runner;
- exact patched `train_asr_adapter.py`;
- all TensorBoard event files;
- initial and resumed run-control logs (`command.txt`, `console.log`, `gpu.csv`,
  `timing.txt` and related logs);
- an `environment.json` containing OS, Python, PyTorch, CUDA build/runtime/driver, cuDNN,
  NeMo commit, Lightning, Hydra, OmegaConf, GPU and VRAM versions.

Before sending the bundle to the Mac:

1. calculate SHA-256 for every file;
2. record those hashes in a separate `SHA256SUMS.txt`;
3. confirm the known hashes in the spec match;
4. zip the files without credentials, home-directory dumps, raw AI Hub data, or unrelated
   checkpoints;
5. keep the original GPU paths and files unchanged.

After copying to `artifacts/gate-2/reproducibility/recovered/`, add newly known expected
hashes to the spec and run the verifier in `REPRODUCIBILITY.md`. Do not mark
`reproducibility_passed=true` until it reports `passed: true`.
