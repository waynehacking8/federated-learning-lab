# Three-way algorithm comparison

## Runs

| Run | Algorithm | Partition | Rounds | Final acc | Best acc | Round to 0.90 |
|---|---|---|---|---|---|---|
| fedavg_iid | fedavg | iid | 15 | 0.9863 | 0.9863 | 1 |
| fedavg_dirichlet_a0.1 | fedavg | dirichlet (alpha=0.1) | 25 | 0.9769 | 0.9782 | 4 |
| fedprox_dirichlet_a0.1_mu0.01 | fedprox | dirichlet (alpha=0.1) | 25 | 0.9766 | 0.9778 | 4 |
| scaffold_dirichlet_a0.1 | scaffold | dirichlet (alpha=0.1) | 25 | 0.9669 | 0.9741 | 4 |

![comparison](three_way_comparison.png)

## Observations

- For Non-IID partitions (Dirichlet alpha=0.1), FedAvg drifts; FedProx
  anchors via the proximal term (mu); SCAFFOLD corrects drift via
  control variates.
- Plot accuracy vs *round*, not wall-clock: SCAFFOLD pays ~2x
  communication per round in return for fewer rounds to a target.
