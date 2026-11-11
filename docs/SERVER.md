# Running training on the lab GPU servers

The lab servers (2 × RTX Pro 6000 96 GB, 384–512 GB RAM, Ubuntu 24.04) are for training, evaluation and
pretrained-feature extraction. Isaac Sim data collection stays on the Windows workstation.

Use **4DGen** by default (more CPU cores and RAM for window building and large feature files). Files are not
shared between the two servers. Book time on the server's shared calendar first; runs left outside the booked
slot may be terminated, so always run inside `tmux` and rely on the resumable queues.

## One-time setup

```bash
ssh <user>@4dgen.ucdenver.pvt              # campus network or VPN
git clone <repo> ~/WorldModel4Sidewalk     # code in home; data and outputs on /home/storage
bash ~/WorldModel4Sidewalk/scripts/server/setup_env.sh /home/storage/$USER/go2wm
```

From the workstation, copy the datasets (episodes, RGB, split manifests, `derived/` features):

```powershell
powershell -File scripts\server\sync_data.ps1 -User <user>
```

## Each session

```bash
tmux new-session -s go2wm
source /home/storage/$USER/go2wm/venv/bin/activate
export GO2WM_ROOT=/home/storage/$USER/go2wm
cd ~/WorldModel4Sidewalk && python -m pytest -q
bash scripts/server/train_queue.sh --data data/go2_e4c_train --prefix e5 --seeds "0 1 2" --gpus "0 1" \
    dinov2s_gru dinov2s_grid_gru
```

Detach with `ctrl+b d`; reattach with `tmux attach -t go2wm`. Bring results back with
`scp -r <user>@4dgen.ucdenver.pvt:/home/storage/<user>/go2wm/outputs/<run> D:\go2wm\outputs\`.

## Notes

- `configs/paths.yaml` points at `D:/go2wm`; on Linux `GO2WM_ROOT` overrides it, so no config edits are needed.
- Split manifests live inside each dataset (`splits/*.json`) and are copied with it, so splits are identical on
  both machines. The `all_test` manifest is frozen on first use; copy it rather than regenerating it.
- Hugging Face models download to `~/.cache/huggingface` on first use (DINOv2-small ~90 MB; ViT-L/g encoders are
  larger; set `HF_HOME=/home/storage/$USER/hf` to keep them off the home directory).
