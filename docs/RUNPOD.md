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

mkdir -p "$MIDI_LLM_BACKUP_ROOT"
bash scripts/run_score_first_stages.sh two-staff-refinement > "$MIDI_LLM_BACKUP_ROOT/stage09.log" 2>&1 &
TRAINING_PID=$!
bash scripts/watch_score_first_backup.sh "$TRAINING_PID" > "$MIDI_LLM_BACKUP_ROOT/backup-watch.log" 2>&1 &
bash scripts/run_score_first_post_training.sh "$TRAINING_PID" > "$MIDI_LLM_BACKUP_ROOT/post-training.log" 2>&1 &
```

The watcher continuously mirrors checkpoints, logs, recovery bundles, and a
final adapter archive into `MIDI_LLM_BACKUP_ROOT`. The post-training watcher
also tries several seeds and mirrors the first validated complete-piece
sample.

To add offsite Google Drive backup, create an ephemeral `rclone`
configuration on the container disk:

```bash
rclone config --config /root/.config/rclone/rclone.conf
export RCLONE_CONFIG=/root/.config/rclone/rclone.conf
export MIDI_LLM_RCLONE_REMOTE='gdrive:MIDI-LLM/runpod'
```

Set those two exports before starting the watcher. Each backup pass will then
sync the persistent backup directory to Google Drive as well. The watcher
explicitly excludes `rclone.conf` from mirrored recovery bundles so OAuth
credentials do not enter the persistent backup or the Drive mirror. Recreate
the ephemeral configuration after rebuilding a Pod.

## Official References

- [Runpod Pod CLI](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
- [Runpod Full SSH setup](https://docs.runpod.io/pods/configuration/use-ssh)
- [Runpod pricing](https://www.runpod.io/pricing)
