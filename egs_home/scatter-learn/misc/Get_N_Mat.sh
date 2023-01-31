#!/bin/sh

#grep "header = " GenerateBinaryScatterMasks.py | grep -v "^\s*#" | sed 's/[^,]//g' | wc -c
grep "header = " generate_scatter_masks.py | grep -v "^\s*#" | sed 's/[^,]//g' | wc -c
