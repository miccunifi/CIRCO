# CIRCO Dataset

This is the **official repository** of the **C**omposed **I**mage **R**etrieval on **C**ommon **O**bjects in context (CIRCO) dataset.

For more details please see our [**paper**](https://github.com/miccunifi/CIRCO) "*Zero-shot Composed Image Retrieval With Textual Inversion*".

>You are currently viewing the dataset repository. If you are looking for more information about our method SEARLE see the [repository](https://github.com/miccunifi/SEARLE).

## Overview
**CIRCO** (**C**omposed **I**mage **R**etrieval on **C**ommon **O**bjects in context) is an open-domain benchmarking dataset for Composed Image Retrieval (CIR) based on real-world images from [COCO 2017 unlabeled set](https://cocodataset.org/#home). It is the first CIR dataset with multiple ground truths and aims to address the problem of false negatives in existing datasets. CIRCO comprises a total of 1020 queries, randomly divided into 220 and 800 for the validation and test set, respectively, with an average of 4.53 ground truths per query. We evaluate the performance on CIRCO using mAP@K.

![](assets/circo.jpg "Examples of CIRCO")

## TODO
- [ ] Dataloader and evaluation code
- [ ] Validation set annotations
- [ ] Test set evaluation server


## Authors

* [**Alberto Baldrati**](https://scholar.google.com/citations?hl=en&user=I1jaZecAAAAJ)**\***
* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)**\***
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

**\*** Equal contribution. Author ordering was determined by coin flip.

## Acknowledgements
This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.
