# pyROGER pacakage
[![Documentation Status](https://readthedocs.org/projects/pyroger/badge/?version=latest)](https://pyroger.readthedocs.io/en/latest/)

Welcome to the pyROGER package, where you will be able to make an automatic dynamical classification
of galaxies around clusters.

## Installation

You can install pyROGER by doing:

`pip install pyROGER`

Or install the last version downloading the source:

`git clone https://github.com/Martindelosrios/pyROGER.git`

And then, in the pyROGER directory just doing:

`pip install .`

### Requirements

For running pyroger you will need:

 * numpy
 * scikit-learn (If you want to use the pre-loaded models, you will need to install scikit-learn==1.3.0)
 * mlxtend
 * pandas
 * joblib

## Quick Start

First you can import the roger library and the roger models...

```python 
from pyROGER import roger
from pyROGER import models

```
Then you can check the available models...

```python 
models.list_saved_models()
```

For uploading a model just give the path to the saved .joblib file

```python 
models.HighMassRoger1.train(path_to_saved_model= [PATH_TO_MODEL/HighMassRoger1_KNN.joblib',
                                                  PATH_TO_MODEL/HighMassRoger1_RF.joblib',
                                                  PATH_TO_MODEL/HighMassRoger1_SVM.joblib'])
```

To check that the model is ready to be used you can do:

```python 
models.HighMassRoger1
```
and you should see 'Ready to use' after some comments about the model.

Finally, for using the model you just do:

```python 
# For predicting the classes:
pred_class = models.HighMassRoger1.predict_class(data[:,(0,1)], 0)

# For predicting probabilities:
pred_prob = models.HighMassRoger1.predict_prob(data[:,(0,1)], 0)

```
 
## Tutorials

In [Examples](https://github.com/Martindelosrios/pyROGER/tree/dev/EXAMPLES) you can find some examples to start playing with pyROGER

## Citation

If you use pyROGER in your paper please cite [https://arxiv.org/abs/2010.11959](https://arxiv.org/abs/2010.11959) 

```latex 
@ARTICLE{2021MNRAS.500.1784D,
       author = {{de los Rios}, Mart{\'\i}n and {Mart{\'\i}nez}, H{\'e}ctor J. and {Coenda}, Valeria and {Muriel}, Hern{\'a}n and {Ruiz}, Andr{\'e}s N. and {Vega-Mart{\'\i}nez}, Cristian A. and {Cora}, Sof{\'\i}a A.},
        title = "{ROGER: Reconstructing orbits of galaxies in extreme regions using machine learning techniques}",
      journal = {\mnras},
     keywords = {methods: analytical, methods: numerical, galaxies: clusters: general, galaxies: kinematics and dynamics, Astrophysics - Astrophysics of Galaxies},
         year = 2021,
        month = jan,
       volume = {500},
       number = {2},
        pages = {1784-1794},
          doi = {10.1093/mnras/staa3339},
archivePrefix = {arXiv},
       eprint = {2010.11959},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.1784D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

<span style="background-color: #d4edda; color: #155724; padding: 10px; border: 1px solid #c3e6cb; border-radius: 5px; display: inline-block;">
A lot of people is already using pyROGER, you can check the [Papers here!](https://scixplorer.org/abs/2021MNRAS.500.1784D/citations?p=1)
</span>
