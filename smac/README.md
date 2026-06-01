# MACCA — SMAC (StarCraft II)

SMAC experiments for **MACCA**, built on [PyMARL2](https://github.com/hijkzzz/pymarl2).

## 1. Install StarCraft II
```bash
bash install/install_sc2.sh        # downloads SC2 4.10 + the required maps
```

## 2. Offline datasets
Download the offline SMAC datasets and decompress the `.h5` files into `offline_datasets/`:
- [sc2datasets (Google Drive)](https://drive.google.com/file/d/1nIRwMrbIy6oJuuIM0okVm6DzDaWpNIqJ/view?usp=share_link)

Files are named `<map>_<quality>.h5`, quality in `{expert, medium, medium_replay}`.

## 3. Train MACCA
```bash
export LIBGL_ALWAYS_SOFTWARE=1
xvfb-run -a -s "-screen 0 1024x768x24" python3 src/main.py \
  --config=macca_omar --env-config=sc2 \
  with env_args.map_name=2s3z h5file_suffix=expert t_max=2000000 use_offline=True
```
- `--config`: `macca_omar` | `macca_icq` | `macca_cql`
- `env_args.map_name`: `2s3z` | `5m_vs_6m` | `6h_vs_8z` | `MMM2`
- `h5file_suffix`: `expert` | `medium` | `medium_replay`
