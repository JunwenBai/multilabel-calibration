# Calibrating Multi-label Predictors with Gradient Boosted Trees

This code is designed to implement a Python version of calibrating the multi-label predictors with Gradient Boosted Trees.

### Requirements

- Python 3.7+
- numpy 1.17.3
- sklearn 0.22.1
- [scikit-multilearn](http://scikit.ml/api/skmultilearn.html)

Older versions might work as well.

### Usage

Simply run

```python ./gb_cal.py```

Datasets are already collected in the `pickles` folder. The default running dataset is the `bibtex` dataset.

### References

Li, Cheng, et al. "Learning to calibrate and rerank multi-label predictions." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Cham, 2019.

[Java implementation](https://github.com/cheng-li/pyramid/wiki/BR-rerank)