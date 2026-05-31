# Runpod GPU Host Setup

Use a Runpod Pod only after reviewing its billed hourly rate in the console.
The score-first pilot is designed for a single `A100 PCIe 80GB` or
`H100 PCIe 80GB` GPU.

## Create The Pod

1. Add the local public SSH key from `~/.ssh/id_ed25519.pub` to the Runpod
   account settings.
2. Create a Pod using an official Runpod PyTorch template.
3. Select one GPU: `A100 PCIe 80GB` is the default cost-conscious choice.
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

Generate the first checkpoint-backed complete-piece sample:

```bash
ssh root@POD_IP -p SSH_PORT \
  'cd midi-llm && bash scripts/generate_score_first_pilot_sample.sh'
scp -P SSH_PORT \
  root@POD_IP:midi-llm/generated_score_first/checkpoint_sample_001.tar.gz \
  generated_score_first/
```

Terminate the billed Pod after collecting the adapter and generated sample.

## Official References

- [Runpod Pod CLI](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
- [Runpod Full SSH setup](https://docs.runpod.io/pods/configuration/use-ssh)
- [Runpod pricing](https://www.runpod.io/pricing)
