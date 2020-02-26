#!/bin/bash
cd ~/RemoteDir/OPEN_AI_repositories/baselines
for i in {1..20}
do
   python baselines/deepq/experiments/train_mountaincar.py
done
