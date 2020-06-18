This repository develops and evaluates a feature selection pipeline for the identification of toxicity targets. 

Existing models for incorporating target knowledge are limited by high dimensionality of feature space while ignoring dependency between structure properties and target binding. To address the problem, we will relate structure properties to targets using a regression model trained on ligand-target interaction data, then apply MultiSURF to identify targets predictive of organ toxicity. A benchmark study showed that MultiSURF outperforms other feature selection methods in detecting genotype-phenotype associations. To evaluate the identified targets, we will use them to build models for predicting organ toxicity and compare the performance to existing models.

Detailed documentation about the code can be found [here](src/README.md). Detailed documentation about the generated dataset can be found [here](data/README.md). Detailed documentation about the generated figure can be found [here](plot/README.md).

## References

+ Urbanowicz RJ, Olson RS, Schmitt P, Meeker M, Moore JH. Benchmarking relief-based feature selection methods for bioinformatics data mining. Journal of biomedical informatics. 2018 Sep 1;85:168-88.

+ Urbanowicz RJ, Meeker M, La Cava W, Olson RS, Moore JH. Relief-based feature selection: Introduction and review. Journal of biomedical informatics. 2018 Sep 1;85:189-203.
