#! /bin/bash
singularity exec /data/containers/test/xfce3.img xvfb-run -a -e /dev/stdout -s '-screen 0 640x480x16' ./launchClient.sh -port 10000 -env