#!/bin/bash
dataset_names='N1-most-common-neighbours-type N2-least-common-neighbours-type N3-parity-neighbours-type S1-multipartite-easy S2-multipartite-random S3-count-triangles-balanced'
model_names='MLP GCN GraphSAGE PNA GATv2'

for d_name in $dataset_names; do
  for m_name in $model_names; do
    python run.py --config-file config/$m_name/$d_name.yaml --results-dir $1/$m_name/ --num-workers $2;
  done
done