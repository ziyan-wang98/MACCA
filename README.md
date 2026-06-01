<h1 align="center">MACCA: Offline Multi-Agent Reinforcement Learning with Causal Credit Assignment</h1>

<p align="center">
  <a href="https://openreview.net/forum?id=gwUOzI4DuV"><img src="https://img.shields.io/badge/TMLR-2025-b31b1b.svg" alt="TMLR"></a>
  <a href="https://arxiv.org/abs/2312.03644"><img src="https://img.shields.io/badge/arXiv-2312.03644-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

<p align="center">
  <b><a href="https://openreview.net/forum?id=gwUOzI4DuV">Published in Transactions on Machine Learning Research (TMLR), 2025</a></b><br>
  Ziyan Wang, Yali Du, Yudi Zhang, Meng Fang, Biwei Huang
</p>

> **TL;DR** — In offline cooperative MARL, agents only see a shared *team* reward, so it is unclear which agent
> deserves credit for it. **MACCA** models the data-generating process as a **Dynamic Bayesian Network** over
> states, actions and rewards, learns the **causal structure** and an **individual reward function** from the
> offline data, and uses the recovered per-agent rewards to drive policy learning. It plugs into any offline
> MARL backbone — we provide **MACCA-OMAR**, **MACCA-CQL** and **MACCA-ICQ**.

<p align="center">
  <img src="assets/framework.png" width="900" alt="MACCA framework"/>
</p>
<p align="center"><i>MACCA learns the causal structure and an individual reward predictor ψ<sub>r</sub> from the offline
team-reward dataset, then trains each agent's policy with the recovered individual rewards.</i></p>

<p align="center">
  <img src="gifs/simple_spread.gif" width="240" alt="Cooperative Navigation"/>
  &nbsp;<img src="gifs/simple_tag.gif" width="240" alt="Predator-Prey"/>
  &nbsp;<img src="gifs/simple_world.gif" width="240" alt="Simple-World"/>
</p>
<p align="center"><i>MACCA policies on MPE — Cooperative Navigation · Predator-Prey · Simple-World</i></p>

---

## Method

Offline MARL with only a team reward `R_t` hides each agent's individual contribution. MACCA addresses this in
two stages:

1. **Causal model learning.** We treat the environment as a Dynamic Bayesian Network and learn (a) a *dynamic
   causal structure* `ψ_g(s_t, a_t, i)` — which state/action dimensions of which agents causally influence the
   reward — and (b) an *individual reward predictor* `ψ_r` such that the predicted per-agent rewards sum to the
   observed team reward. Under the offline-dataset setting these are shown to be **identifiable**.
2. **Credit-aware policy learning.** The recovered individual rewards replace the team reward in any offline
   MARL optimizer, yielding accurate, interpretable credit assignment without fitting an extra advantage/value
   estimator.

This repository provides three instantiations: **MACCA-OMAR**, **MACCA-CQL**, and **MACCA-ICQ**.

## Installation

```bash
# MPE (Cooperative Navigation / Predator-Prey / Simple-World)
conda create -n macca python=3.7 -y && conda activate macca
cd mpe
pip install -r requirements.txt                      # torch 1.12 (CUDA sm<=86 / A100), gym 0.9.4
pip install -e multiagent-particle-envs              # the MPE environments

# SMAC (StarCraft II) — separate env:
conda env create -f environment.yml && conda activate smac
cd smac
bash install/install_sc2.sh                          # installs StarCraft II 4.10 + maps
```

## Datasets

Four offline data qualities per environment — **Random**, **Medium-Replay**, **Medium**, **Expert** — where the
individual rewards are hidden and only their sum (the team reward) is stored.

- **MPE**: place under `mpe/datasets/<env>/<quality>/seed_<k>_data/` (PP/World also need
  `pretrained_adv_model.pt`, the prey policy used at evaluation).
- **SMAC**: place the `.h5` files under `smac/offline_datasets/`
  (download link in `smac/README.md`).

## Usage

Ready-to-run examples (MACCA only) live in [`examples/`](examples):

```bash
bash examples/mpe_cooperative_navigation.sh   # MACCA-OMAR on CN  (simple_spread)
bash examples/mpe_predator_prey.sh            # MACCA-OMAR on PP  (simple_tag)
bash examples/mpe_world.sh                    # MACCA-OMAR on WORLD (simple_world)
bash examples/smac_2s3z.sh                    # MACCA-OMAR on SMAC 2s3z
bash examples/smac_5m_vs_6m.sh                # MACCA-ICQ  on SMAC 5m_vs_6m
bash examples/smac_6h_vs_8z.sh                # MACCA-OMAR on SMAC 6h_vs_8z
```

Switch algorithm / quality with `--algorithm_name {causal_omar,causal_cql}` (MPE) or
`--config={macca_omar,macca_icq,macca_cql}` and `h5file_suffix={expert,medium,medium_replay,random}` (SMAC).
Render a showcase GIF of a trained MPE policy by adding `--save_gif`.

## Citation

```bibtex
@article{wang2025macca,
  title   = {{MACCA}: Offline Multi-agent Reinforcement Learning with Causal Credit Assignment},
  author  = {Wang, Ziyan and Du, Yali and Zhang, Yudi and Fang, Meng and Huang, Biwei},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/forum?id=gwUOzI4DuV}
}
```

## Acknowledgements

The offline MPE datasets and OMAR backbone build on [OMAR](https://github.com/ling-pan/OMAR) /
[CFCQL](https://github.com/thu-rllab/CFCQL); the SMAC pipeline builds on
[PyMARL2](https://github.com/hijkzzz/pymarl2). We thank the authors for releasing their code.
