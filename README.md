[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# XSPNFlow: Exchangeability-Aware Sum-Product Networks

This repository contains the official implementation of [Exchangeability-Aware Sum-Product Network](https://arxiv.org/abs/2110.05165) (XSPNs), introduced in the paper by Stefan Lüdtke, Christian Bartelt, and Heiner Stuckenschmidt, published at IJCAI 2022. 
The implementation is based on [SPFlow](https://github.com/SPFlow/SPFlow). For an introduction to SPFlow, see their [original Readme file](README_SPFLOW.md). 

## Example

This repository adds two new SPN leave types: Fully exchangeable, discrete distributions and Exchangeable Variable Models. We call an SPN with such leaves an “Exchangeability-Aware SPN” (XSPN). 

### Manually constructing XSPNs
XSPNs can be constructed like this: 

```python
from spn.algorithms.Exchangeability import Exchangeable
from spn.structure.Base import Sum, Product
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Inference import likelihood
import pandas as pd
import numpy as np

#For an exchangeable leaf, parameters w_1, ..., w_{|T|} as specified in the
#paper need to be provided. w_t is the probability of an assignment with t ones.
#In this example, the exchangeable node ranges over two variables.
#with p(0,0) = 0.2; p(0,1) = p(1,0) = 0.25; p(1,1) = 0.3.
params = pd.DataFrame([0.2,0.25,0.3])
expart = Exchangeable(params = params,scope=[0,1])

#The overall spn has the form p(x0,x1,x2) = p(x0,x1) * p(x2),
#where p(x0,x1) is exchangeable
xspn = Product(children=[expart,Categorical(p=[0.6,0.4],scope=2)])
```
Likelihoods for samples can be computed as usual:

```python
data = np.array([
[0,0,0],
[0,0,1],
[1,0,0],
[0,1,0]
])

ll = likelihood(xspn, data)
```

### XSPN Structure Learning
It is also possible to learn XSPNs via the LearnXSPN algorithm. Let us first load the senate dataset, containing all roll call votes of 98 senators in the Senate of the 116th United States Congress.

```python
from spn.structure.leaves.parametric.Parametric import Bernoulli, Categorical
from spn.structure.Base import Context

from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.Exchangeability import *
from spn.structure.leaves.evm.EVMLeaf import create_evm_leaf, EVM

dataset = "senate116"

data_train = np.loadtxt(open("spn/data/relational_real/senate116.ts.data","rb"),dtype=int,delimiter=",")
data_test = np.loadtxt(open("spn/data/relational_real/senate116.test.data","rb"),dtype=int,delimiter=",")

```

Next, we specify the parameters of the learning algorithm:
* All variables are Bernoulli variables. 
* As exchangeability test, we use the pairwise chi squared test described in the paper. 
* We use Exchageable Variable Models as multivariate leaves in the fallback case. This allows somewhat more general models than described in the paper, where only fully exchangeable distributions are discussed. Setting numComponents=1 makes the model identical to the XSPNs described in the paper. 

```python
#all variables are Bernoullis
ds_context = Context(parametric_types=[Bernoulli]*data_train.shape[1]).add_domains(data_train)

#specify the exchangeability test: pairwise chi squared test with significance 0.2
extest = lambda data:isExchangeable_viaChiSquared_pairwise(data,significance=0.05)

#EVM leaves as fallback case
create_leaf = lambda local_data,ds_context,scope,alpha: create_evm_leaf(local_data, ds_context, 
                                                          scope,alpha,numComponents=1)
```

Now, we are ready to call the actual learning algorithm. We use Laplace smoothng with alpha=0.1, G-Test for splitting columns, and GMM clustering for splitting rows. Depending on your machine, this call might take a few seconds.

```python
xspn = learn_parametric(data_train, ds_context, min_instances_slice=5,alpha=0.1,
                   alpha_exchangeable=0.1,isExchangeableTest=extest,threshold = 5,
                   multivariate_leaf=True, leaves=create_leaf,
                   cols ="gtest",rows="gmm")
```

For comparison, we can also learn a vanilla SPN with univariate leaves but otherwise identical parameters.

```python
spn = learn_parametric(data_train, ds_context, min_instances_slice=5,
                       doTestExchangeability=False,alpha=0.1,threshold = 5,
                       cols = "gtest",rows="gmm")

```

Finally, we can compute the test log likelihood of both models. We should get ll_xspn = -19.66 and ll_spn = -21.22.

```python
ll_xspn = np.mean(log_likelihood(xspn,data_test)) 
ll_spn = np.mean(log_likelihood(spn,data_test)) 
```

## Datasets

This repository also contains the [synthetic](src/spn/data/relational_synthetic/) and [real](src/spn/data/relational_real/) relational datasets used in the paper.

## Citation

If you find this repository useful, please consider citing the following paper:
```
@inproceedings{luedtke2022exchangeability,
  author = {Stefan Lüdtke and Christian Bartelt and Heiner Stuckenschmidt},
  title = {Exchangeability-Aware Sum-Product Networks},
  booktitle={Proceedings of the 31st International Joint Conference on Artificial Intelligence},
  year = {2022}
}
```

## Authors

* Stefan Lüdtke, University of Mannheim
* For the authors and contributors of SPFlow, see their [readme file](README_SPFLOW.md).

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
