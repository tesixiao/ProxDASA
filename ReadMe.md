# Prox-DASA-(GT): Decentralized Proximal Averaged Stochastic Approximation
---

In this file we provide instructions on experimental results of ["A One-Sample Decentralized Proximal Algorithm for Non-Convex Stochastic Composite Optimization"](https://arxiv.org/abs/2302.09766). We used the implementation of SPPDM, ProxGT-SR and DEEPSTORMv2 in https://github.com/gmancino/DEEPSTORM, and based on their framework we implemented Prox-DASA and Prox-DASA-GT.

## Experiment set-up

All experiments were conducted on a laptop with Intel Core i7-11370H Processor and Windows 11 operating system. All code is written in Python version 3.9.15, using `PyTorch` version 1.13.1. and `mpi4py` version 3.1.4. See [here](https://mpi4py.readthedocs.io/en/stable/install.html) for instructions on how to install this package.

A complete list of packages necessary for completing the experiments is located in the [requirements.txt](requirements.txt) file.


## Running experiments

For sake of example, we will use `proxdasa.py` as our test method here; see the `methods` folder for the other methods utilized in the experiments

1. Move desired method out of `methods` folder

For sake of organization, the actual methods utilized in the experiments are delegated to the `methods` folder. Move them into the root directory

2. Run experiments using the parameters listed below

| Dataset | Parameters |
| :--- | :--- |
| a9a | `--data='a9a' --updates=10001 --report=100 --l1=1e-4` |
| MNIST | `--data='mnist' --updates=3001 --report=100 --l1=1e-4` |


For a full list of required parameters specific to each method, refer to the following table:

| Method | Dataset | Parameters |
| :--- | :--- | :--- |
| `deepstormv2.py` | a9a | `--init_batch=1 --mini_batch=4 --comm_pattern='ring' --step_type='diminishing' --k0=5 --beta=0.031 --lr=10.0` |
|  | MNIST | `--init_batch=1 --mini_batch=32 --comm_pattern='random' --step_type='diminishing'  --k0=3 --beta=0.0228 --lr=5.0` |
| `proxgtsr.py` | a9a | `--mini_batch=4 --full_grad=1000 --comm_pattern='ring' --lr=5e-3` |
|  | MNIST | `--mini_batch=32 --full_grad=32 --comm_pattern='random' --lr=0.1` |
| `sppdm.py` | a9a | `--mini_batch=4 --comm_pattern='ring' --c=1.0 --kappa=0.1 --gamma=3 --beta=0.9 --mom='nesterov' --alpha=0.01` |
|  | MNIST | `--mini_batch=32 --comm_pattern='random' --c=1.0 --kappa=0.1 --gamma=3 --beta=0.9 --mom='nesterov' --alpha=0.1` |
| `proxdasa.py` | a9a | `--mini_batch=4 --comm_pattern='ring' --step_type='diminishing' --alpha_base=0.3 --lr=10.0` |
|  | MNIST | `--mini_batch=32 --comm_pattern='random' --step_type='diminishing'  --alpha_base=3.0 --lr=1.0` |
| `proxdasagt.py` | a9a | `--mini_batch=4 --comm_pattern='ring' --step_type='diminishing' --alpha_base=0.3 --lr=10.0` |
|  | MNIST | `--mini_batch=32 --comm_pattern='random' --step_type='diminishing'  --alpha_base=3.0 --lr=5.0` |

So a full example for running `proxdasa.py` on the a9a dataset over the ring communication graph is given by:
```
mpiexec -np 8 python proxdasa.py --data=a9a --updates=10001 --report=100 --l1=1e-4 --comm_pattern=ring --mini_batch=4 --lr=10
```

1. Reproduce the results from the paper by varying over all initializations

Append `--trial=i` for `i=1,2,...,10` to the above code to record outputs across different initializations.

4. The results will be saved in the `results` folder

## Summary

We have the following directories and files:
```
helpers/
init_weights/
mixing_matrices/
methods/
models/
results/
requirements.txt
```

#### data
The `data` folder contains subfolders housing the training/testing data used in the experiments. To add a new dataset, include the folders here and modify lines containing `torch.utils.data.DataLoader` in the corresponding methods file

#### helpers
The `helpers` folder contains helper functions utilized by all methods. These are:

1. `replace_weights.py`: a custom PyTorch optimizer that simply _replaces_ model parameters from a LIST of PyTorch tensors; since all methods update many different variables, this is a straightforward way to minimize conflicts when updating model parameters. The `.step()` method requires two parameters:
    - `weights`: a LIST of PyTorch tensors of length `k` where `k` represents the number of model parameters
    - `device`: the torch device (i.e. GPU) the model is saved on

2. `l1_regularizer.py`: performs soft-thresholding of the weights in a LIST of PyTorch tensors via the `.forward()` method. Additionally, with the `.number_non_zeros()` method, we count the number of non-zero elements (up to error tolerance `1e-6`) in a LIST of PyTorch tensors

3. `custom_data_loader.py`: loads the a9a dataset from their tensor files in the proper format

#### mixing_matrices
The `mixing_matrices` folder contains `Numpy` arrays of size `N x N` where each `(i,j)` entry corresponds to agent `i`'s weighting of agent `j`'s information

To run experiments with different weight matrices/communication patterns or sizes, save the corresponding mixing matrix in this directory

#### init_weights
The `init_weights` directory contains the initial parameters for each agent, for each random seed, used in the experiments. These are only uploaded for sake of completeness; to remove the need to save initial weights for future experiments, comment lines containing
```
[torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]
```
and replace with
```
[p.data.detach().clone() for p in self.model.parameters()]
```

#### models
The `models` folder contains the neural network architectures used in the experiments

#### methods
The `methods` folder contains all of the different decentralized optimization methods compared in this work; see the paper for a list of those compared here

## Citation

Please cite the following papers if you use this code in your work:

```
@inproceedings{mancino2023proximal,
  title={Proximal Stochastic Recursive Momentum Methods for Nonconvex Composite Decentralized Optimization},
  author={Mancino-ball, Gabriel and Miao, Shengnan and Xu, Yangyang and Chen, Jie},
  booktitle={AAAI Conference on Artificial Intelligence},
  publisher={{AAAI} Press},
  year={2023}
}

@inproceedings{xiao2023one,
  title={A One-Sample Decentralized Proximal Algorithm for Non-Convex Stochastic Composite Optimization},
  author={Xiao, Tesi and Chen, Xuxing and Balasubramanian, Krishnakumar and Ghadimi, Saeed},
  booktitle={Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  publisher = {PMLR},
  year={2023}  
}
```