# Gaussian Kernel based Mixture of Experts
Gaussian Kernel based Mixture of Experts are an extension to the standard case of Mixture of Experts, proposed by Rumelhart. It is an ensemble model composed by several neural networks that uses a "divide and conquer" strategy, in which each network/expert focus on a specific region in data space. The distribution of samples among experts are managed by a gating network, which is also a neural network. 

In summary, the gating network is responsible for generating weights for each sample and expert, in order to define the importance factor that correlates each sample and expert. These importance factors are adjusted through iterative processes such as Expectation-Maximization. The Gaussian Kernel eliminates the necessity of using such a costly algorithm and uses an unsupervised learning technique to define which samples goes to each expert. 

More details can be found in the scientific work (check How to cite) and in the example file.

## How to cite
Please consider citing this model:
```
@mastersthesis{giuliani2022mekg,
    author = {Giuliani, Henrique L. V.},
    title = {{Comitês de Máquinas Aplicados a Interfaces Cérebro-Computador}},
    year = {2022},
    school = {UFABC}
}
```