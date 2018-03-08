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
* For command-line configurations, please see `offline.py`.

### Gradient-based Meta-Optimization
```
python offline.py --run smd
```
* For command-line configurations, please see `offline.py`.

## Online Meta-Optimization Experiments
```
python online.py    [--dataset {DATASET}]                   \
                    [--num_meta_steps {NUM_META_STEPS}]     \
                    [--steps_per_update {STEPS_PER_UPDATE}]
```
* Possible `{DATASET}` options are `mnist`, `cifar-10`, default `mnist`.
* Replace `{NUM_META_STEPS}` with number of meta-optimization steps per update, default 10.
* Replace `{STEPS_PER_UPDATE}` with number of regular training steps per meta updates, default 100.
* Use larger `{NUM_META_STEPS}` and smaller `{STEPS_PER_UPDATE}` to observe stronger effect of short-horizon bias (also slower to run).

## How to Write Forward-Mode AutoDiff for Other Optimizers

We include standard `GradientDescentOptimizer`, `MomentumOptimizer`, `MomentumInvDecayOptimizer`,
and `AdamOptimizer`. It is very easy to take gradients on the hyperparameters of other optimizers
that we haven't defined. Follow the instruction below on how to perform meta-optimization on other
optimizers.

1. You need to write a custom optimizer class. The constructor pass in a dictionary of
   hyperparameters that needs gradients. For example:
```python
class GradientDescentOptimizer(Optimizer):
    def __init__(self, lr, dtype=tf.float32):
        """Gradient descent optimizer.

        Args:
            lr: Float. Learning rate.
            dtype: Data type, default tf.float32.
        """
        super(GradientDescentOptimizer, self).__init__({'lr': lr}, dtype=dtype)
```
You can then access the value of the hyperparameter by accessing `self.hyperparams`, e.g.:
```python
lr = self.hyperparams['lr']
```

2. Implement the following two functions. 
```python
    def apply_gradients(self, grads_and_vars, global_step=None):
        """Applies gradients

        Args:
            grads_and_vars: List of tuples of the gradients and the variables.
            global_step: Tensor that records the global step. Optional.

        Returns:
            train_op: A TensorFlow op that applies the gradients to the variables.
        """
        raise NotImplemented()

    def minimize(self, cost, var_list=None, global_step=None,
                 gate_gradients=1):
        """Minimizes a cost function.

        Args:
            cost: Cost function to minimize.
            var_list: List of trainable variables, default tf.trainable_variables().
            global_step: Global step counter.
            gate_gradients: Whether to allow concurrency in calculating the gradients.

        Returns:
            train_op: A TensorFlow op that applies the gradients to the variables.
        """
        raise NotImplemented()
```

3. A tensor is called an accumulator if the recursive update formula relies on the value of the
   tensor of the previous time step. For example, the weight parameters are accumulators; velocities
   in SGD with momentum are also accumulators. Note that you need to keep a reference of the old
   accumulators and new accumulators in order to compute derivatives efficiently. For example, in `apply_gradients`, we keep a reference like this:
```python
self._accumulators = {'w': var_list}
self._new_accumulators = {'w': var_list_new}
```


4. Then, using the look ahead function defined in `look_ahead.py`, we can compute the forward
   gradients ops.
```python
hp_dict = {'lr': init_lr, mom_name: momentum}
hp_names = hp_dict.keys()
hyperparams = dict([(hp_name, model.optimizer.hyperparams[hp_name]) for hp_name in hp_names])
grads = model.optimizer.grads
accumulators = model.optimizer.accumulators
new_accumulators = model.optimizer.new_accumulators
loss = model.cost

# Build look ahead graph.
look_ahead_ops, hp_grad_ops, zero_out_ops = look_ahead_grads(
        hyperparams, grads, accumulators, new_accumulators, loss)
```
`look_ahead_ops` is called during regular training;
`hp_grad_ops` is called in the end where we compute the training loss of a certain set of objective examples;
`zero_out_ops` is called if we want to do this process over again, to clean up all the intermediate gradients variables. Below is an example of using these ops.

```python
# Running `look_ahead_ops` during regular training.
for ii, (xd, yd) in enumerate(data_list):
    fdict = {model.x: xd, model.y: yd}
    sess.run(look_ahead_ops, feed_dict=fdict)
    sess.run(model.train_op, feed_dict=fdict)

# Final round. Running evaluation on the objective training loss.
fdict = {model.x: x_eval, model.y: y_eval}

# Get hyperparam gradients by running hp_grad_ops
hp_grads = sess.run(hp_grad_ops, feed_dict=fdict)
```

5. Please see `optimizer.py`, `offline.py`, and `train.py` for more details.


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
