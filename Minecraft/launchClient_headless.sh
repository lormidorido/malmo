#! /bin/bash
xvfb-run -a -e /dev/stdout -s '-screen 0 640x480x16' ./launchClient.sh -port $1 -env > ../out.txt 2>&1