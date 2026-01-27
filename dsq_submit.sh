#!/bin/sh
module load dSQ;
filename=$1;
# dSQ --jobfile ${filename}.txt --mem=16G -t 01:00:00 -n 1 -p pi_krishnaswamy,scavenge -J ${filename}
dSQ --jobfile ${filename}.txt --mem=32G -t 01:00:00 -n 1 -p scavenge,pi_krishnaswamy -J ${filename}
# dSQ --jobfile ${filename}.txt --mem 128G -t 1-01:00:00 -n 1 -p  pi_krishnaswamy,scavenge,general,interactive -J ${filename}
sbatch dsq-${filename}-$(date +%Y-%m-%d).sh # > pid_dsq-${filename}-$(date +%Y-%m-%d-%T-%s).txt
# cat pid_dsq-${filename}-$(date +%Y-%m-%d-%T-%s).txt
