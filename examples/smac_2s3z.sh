#!/bin/bash
# MACCA on SMAC 2s3z (Easy). Reproduces Table 2 2s3z (paper MACCA-OMAR Expert 0.99 / Medium 0.55).
# Algorithm config: macca_omar | macca_icq | macca_cql.  Quality: expert | medium | medium_replay.
# Headless SC2 needs xvfb. Requires SC2 4.10 + offline_datasets/2s3z_<quality>.h5.
cd "$(dirname "$0")/../smac"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
xvfb-run -a -s "-screen 0 1024x768x24" python3 src/main.py \
  --config=macca_omar --env-config=sc2 \
  with env_args.map_name=2s3z h5file_suffix=expert t_max=2000000 use_offline=True
