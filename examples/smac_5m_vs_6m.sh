#!/bin/bash
# MACCA on SMAC 5m_vs_6m (Hard). Reproduces Table 2 5m_vs_6m (paper MACCA-ICQ Expert 0.88 / MACCA-OMAR 0.73).
# Algorithm config: macca_omar | macca_icq | macca_cql.  Quality: expert | medium | medium_replay.
# seed=1 pins the run; this hard map benefits from longer training — raise t_max if not yet converged.
cd "$(dirname "$0")/../smac"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
xvfb-run -a -s "-screen 0 1024x768x24" python3 src/main.py \
  --config=macca_icq --env-config=sc2 \
  with env_args.map_name=5m_vs_6m h5file_suffix=expert t_max=2000000 use_offline=True seed=1
