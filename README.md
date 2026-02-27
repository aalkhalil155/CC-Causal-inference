# causal-dragonnet-dummy

Minimal DragonNet-style continuous treatment dummy project.

## Run locally
```bash
pip install -r requirements.txt
python train.py
```

## Run in Google Colab
1. Open a new Colab notebook.
2. (Optional) Enable GPU: **Runtime -> Change runtime type -> T4 GPU**.
3. Run the following cell (replace with your repo URL):

```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
!pip install -r requirements.txt
!python train.py
```

### Optional: quick Colab-sized run
If you want a faster smoke run in Colab, reduce training size/epochs before running `train.py`:

```python
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["device"] = "cpu"  # use "cuda" if GPU runtime enabled
cfg["data"]["n_train"] = 5000
cfg["train"]["epochs"] = 3

with open("config.yaml", "w") as f:
    yaml.safe_dump(cfg, f)
```

## Colab troubleshooting
If you see an import error like `attempted relative import with no known parent package`:

- Make sure you are inside the repo folder before running (`%cd <your-repo>`).
- Confirm you are in the project folder with `%pwd` before training.
- Run the script form, not package form: `!python train.py`.
- Restart runtime and re-run cells to avoid stale code from old clones.

## Merge conflict note (for this PR)
If GitHub asks between **Accept current change** and **Accept incoming change** for `train.py`/`README.md` in this fix flow, choose:

- **Accept current change** to keep the base branch version (as you requested).
- Then manually verify `train.py` still runs with `python train.py` and `README.md` still includes the Colab steps above.
