# Investigating the Interplay between Features and Structures in Graph Learning
This is the repo of the paper:

[Investigating the Interplay between Features and Structures in Graph Learning](https://arxiv.org/abs/2308.09570), Daniele Castellana and Federico Errica, *20th MLG workshop @ ECML-PKDD* 

## Run the experiments
In our experiments, we assess 5 models on 6 synthetic datasets.

**Models**:
- MLP
- GCN
- GATv2
- GraphSAGE
- PNA

**Datasets**:
- N1-most-common-neighbours-type
- N2-least-common-neighbours-type
- N3-parity-neighbours-type
- S1-multipartite-easy
- S2-multipartite-random
- S3-count-triangles-balanced

To execute a single experiment it is enough to run the following command:
`python run.py --config-file config/MODEL_NAME/DATASET_NAME.yaml --results-dir YOUR_RESULTS_DIR --num-workers NUM_WORKERS`

Note that this command executes the models selection in a parallel way. To execute it sequentially, you should set `NUM_WORKERS=1`. See the config file to check the hyperparameters values validated.

## Run all the experiments
The script *run_all.sh* allows to execute all the experiments of our paper (all the models on all the datasets). The script is executed by the following command:

`.\run_all.sh YOUR_RESULTS_DIR NUM_WORKERS`

The two arguments are mandatory.