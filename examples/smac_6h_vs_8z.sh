#!/bin/bash
# MACCA on SMAC 6h_vs_8z (Super Hard). Reproduces Table 2 6h_vs_8z (paper MACCA-OMAR Expert 0.75).
# This super-hard map needs longer training (5-10M env steps) to converge.
# Algorithm config: macca_omar | macca_icq | macca_cql.  Quality: expert | medium | medium_replay.
cd "$(dirname "$0")/../smac"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
xvfb-run -a -s "-screen 0 1024x768x24" python3 src/main.py \
  --config=macca_omar --env-config=sc2 \
  with env_args.map_name=6h_vs_8z h5file_suffix=expert t_max=5000000 use_offline=True
