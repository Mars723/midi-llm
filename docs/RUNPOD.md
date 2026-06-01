# Runpod GPU Host Setup

Use a Runpod Pod only after reviewing its billed hourly rate in the console.
The score-first pilot is designed for a single `A100 PCIe/SXM 80GB` or
`H100 PCIe 80GB` GPU.

## Create The Pod

1. Add the local public SSH key from `~/.ssh/id_ed25519.pub` to the Runpod
   account settings.
2. Create a Pod using an official Runpod PyTorch template.
3. Select one GPU: `A100 PCIe/SXM 80GB` is the default cost-conscious choice.
4. Allocate at least `50GB` container disk and `50GB` persistent volume.
5. Enable Full SSH with a public IP and TCP port `22`.
6. Copy the `SSH over exposed TCP` command from the Pod's Connect tab.

Runpod's basic proxied SSH is insufficient for this workflow because it does
not support `scp`. The required command resembles:

```bash
ssh root@POD_IP -p SSH_PORT -i ~/.ssh/id_ed25519
```

## Deploy

From the local MIDI-LLM workspace, pass the public-IP SSH target and port to
the checked-in deploy helper:

```bash
bash scripts/deploy_score_first_gpu.sh root@POD_IP SSH_PORT
```

The helper fast-forwards the score-first branch, uploads the verified dataset
bundle, checks SHA-256 digests before extraction, installs training
dependencies, and writes a remote dry-run specification.

## Run

Run the upstream-native parity gate before additional ScoreDSL training. This
uses the original checkpoint, prompt framing, MIDI BOS token, and Anticipation
MIDI representation. The resulting `native.mid` files are the authoritative
musical-content candidates:

```bash
export MIDI_LLM_NATIVE_OUTPUT_ROOT=/workspace/midllm-backups/generated_native_backbone
export MIDI_LLM_RCLONE_REMOTE='gdrive:MIDI-LLM/runpod/score-first-intermediate-v2'
bash scripts/run_native_backbone_pilot.sh
```

Inspect the native MIDI candidates against the upstream Live Demo before
training a notation editor. Converted score galleries are draft-only previews.
See [`ARCHITECTURE_CORRECTION.md`](ARCHITECTURE_CORRECTION.md).

The direct ScoreDSL adapter commands below remain available for research
comparison. They are not the default quality path.

Run the one-step real-sample check before the longer pilot:

```bash
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/run_score_first_stages.sh smoke'
```

After smoke succeeds, run the adapter-producing pilot:

```bash
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/run_score_first_stages.sh pilot'
```

Run the structural and complete-piece focus phases before producing the
reviewable sample:

```bash
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/run_score_first_stages.sh structure'
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/run_score_first_stages.sh whole-piece-focus'
```

After deploying the v3 materialization, run the notation-diversity refinement:

```bash
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/run_score_first_stages.sh diversity'
```

Generate the first checkpoint-backed complete-piece sample:

```bash
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/generate_score_first_pilot_sample.sh'
scp -P SSH_PORT \
  root@POD_IP:midi-llm/generated_score_first/checkpoint_sample_001.tar.gz \
  generated_score_first/
```

Terminate the billed Pod after collecting the adapter and generated sample.

## Persist And Back Up Training Runs

Keep the repository and all training outputs on the persistent volume, such
as `/workspace/midi-llm`. The virtual environment can stay on the faster
container disk because it is disposable.

For a long stage, save resumable checkpoints more frequently and run the
backup watcher:

```bash
export MIDI_LLM_RUN_ROOT=/workspace/midi-llm/training_runs/score_first_intermediate_v2
export MIDI_LLM_BACKUP_ROOT=/workspace/midillm-backups
export MIDI_LLM_RECOVERY_ROOT=/workspace/midillm-recovery
export MIDI_LLM_SAVE_STEPS=25
export MIDI_LLM_BACKUP_MIRROR_RUN_ROOT=0
export MIDI_LLM_CREATE_ADAPTER_ARCHIVE=0

mkdir -p "$MIDI_LLM_BACKUP_ROOT"
bash scripts/run_score_first_stages.sh two-staff-refinement > "$MIDI_LLM_BACKUP_ROOT/stage09.log" 2>&1 &
TRAINING_PID=$!
bash scripts/watch_score_first_backup.sh "$TRAINING_PID" > "$MIDI_LLM_BACKUP_ROOT/backup-watch.log" 2>&1 &
bash scripts/run_score_first_post_training.sh "$TRAINING_PID" > "$MIDI_LLM_BACKUP_ROOT/post-training.log" 2>&1 &
```

On ordinary disks, the watcher mirrors checkpoints, frozen log snapshots,
recovery bundles, and a final adapter archive into `MIDI_LLM_BACKUP_ROOT`.
For a Runpod network-mounted persistent volume, disable the redundant same-disk
run mirror and adapter archive as shown above. Live logs remain in `logs/`;
stable Drive copies are written from `log-snapshots/latest/` so an actively
appended log cannot cause a self-referential sync loop. The post-training
watcher also tries several seeds and mirrors the first validated complete-piece
sample.

If a Pod stops after a numbered Trainer checkpoint has been written, resume
the exact optimizer step instead of restarting the stage:

```bash
export MIDI_LLM_RESUME_CHECKPOINT_DIR=/workspace/midi-llm/training_runs/score_first_intermediate_v2/09_two_staff_refinement/checkpoint-25
bash scripts/run_score_first_stages.sh two-staff-refinement
```

For the focused stage 10 diversity repair pass, set the backup artifact stage
before starting the same watcher pattern:

```bash
export MIDI_LLM_BACKUP_STAGE=10_variation_repair_refinement
export MIDI_LLM_VARIATION_REPAIR_RESUME_ADAPTER_DIR=/root/midllm-local/09_two_staff_refinement-adapter
bash scripts/run_score_first_stages.sh variation-repair-refinement
```

Copy the stage 09 adapter to the disposable container disk before setting this
override when the network-mounted persistent volume stalls on large adapter
reads. Training checkpoints and final outputs must still use
`MIDI_LLM_RUN_ROOT` on the persistent volume.

Use the same local-read pattern for long checkpoint-backed generation. Keep
streamed raw DSL on the disposable container disk and mirror only stable
section partials while sampling:

```bash
export MIDI_LLM_SAMPLE_ADAPTER_DIR=/root/midllm-local/10_variation_repair_refinement-adapter
OUTPUT_ROOT=/root/midllm-local/generated_score_first
export MIDI_LLM_SAMPLE_OUTPUT="$OUTPUT_ROOT/checkpoint_sample_001"
mkdir -p "$OUTPUT_ROOT"

bash scripts/generate_score_first_pilot_sample.sh > "$MIDI_LLM_BACKUP_ROOT/sample.log" 2>&1 &
GENERATION_PID=$!
bash scripts/watch_score_first_generation_backup.sh "$GENERATION_PID" "$OUTPUT_ROOT" \
  > "$MIDI_LLM_BACKUP_ROOT/generation-backup-watch.log" 2>&1 &
```

The generation watcher copies completed `partial_after_<section>` ScoreIR files
and failure reports into the persistent backup root and configured Drive
remote. It intentionally skips raw DSL files because they are updated while
tokens stream.

To add offsite Google Drive backup, create an ephemeral `rclone`
configuration on the container disk:

```bash
rclone config --config /root/.config/rclone/rclone.conf
export RCLONE_CONFIG=/root/.config/rclone/rclone.conf
export MIDI_LLM_RCLONE_REMOTE='gdrive:MIDI-LLM/runpod'
```

Set those two exports before starting the watcher. Each backup pass will then
sync the training source directory and persistent backup metadata to Google
Drive. The watcher explicitly excludes `rclone.conf` from mirrored recovery
bundles so OAuth credentials do not enter the persistent backup or the Drive
mirror. Recreate the ephemeral configuration after rebuilding a Pod.

## Official References

- [Runpod Pod CLI](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
- [Runpod Full SSH setup](https://docs.runpod.io/pods/configuration/use-ssh)
- [Runpod pricing](https://www.runpod.io/pricing)
