# meta-optim-public
Code for paper
*Understanding Short-Horizon Bias in Stochastic Meta-Optimization* [[arxiv](https://arxiv.org/abs/1803.02021)]

## Dependencies
* matplotlib
* numpy
* pandas
* python 2/3
* tensorflow 1.3+
* tensorflow-forward-ad
* tqdm

Our code is tested on Ubuntu 14.04 and 16.04.

## Offline Meta-Optimization Experiments
### Meta-Optimization Loss Surface
```
# Do random search of hyperparameters.
python offline.py --run surface

# Train a network using the best selected hyperparameters.
python offline.py --run best
```
* For commandline configurations, please see `offline.py`.

### Gradient-based Meta-Optimization
```
python offline.py --run smd
```
* For commandline configurations, please see `offline.py`.

### Online Meta-Optimization Experiments
```
python online.py    [--dataset {DATASET}]                   \
                    [--num_meta_steps {NUM_META_STEPS}]     \
                    [--steps_per_update {STEPS_PER_UPDATE}]
```
* Possible `{DATASET}` options are `mnist`, `cifar-10`, default `mnist`.
* Replace `{NUM_META_STEPS}` with number of meta-optimization steps per update, default 10.
* Replace `{STEPS_PER_UPDATE}` with number of regular training steps per meta updates, default 100.
* Use larger `{NUM_META_STEPS}` and smaller `{STEPS_PER_UPDATE}` to observe stronger effect of short-horizon bias (also slower to run).

## Citation
If you use our code, please consider cite the following:
* Yuhuai Wu, Mengye Ren, Renjie Liao and Roger B. Grosse.
Understanding Short-Horizon Bias in Stochastic Meta-Optimization. 
In *Proceedings of 6th International Conference on Learning Representations (ICLR)*, 2018.

```
@inproceeding{wu18shorthorizon,
  author   = {Yuhuai Wu and 
              Mengye Ren and 
              Renjie Liao and 
              Roger B. Grosse},
  title    = {Understanding Short-Horizon Bias in Stochastic Meta-Optimization},
  booktitle= {Proceedings of 6th International Conference on Learning Representations {ICLR}},
  year     = {2018},
}
```
