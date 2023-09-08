# Reliable Semi-Supervised Ensemble Learning 

This repository contains the code for the "Reliable Semi-Supervised Ensemble Learning" (RESSEL) method.

This method was proposed in the following paper:

"A reliable ensemble based approach to semi-supervised learning" by Sjoerd de Vries & Dirk Thierens - Knowledge Based Systems

DOI: https://doi.org/10.1016/j.knosys.2021.106738

### Included

This repository contains the code for the RESSEL method itself, to allow for it to be used in other projects. 

### Installation

To install the code as a locally editable package, activate the environment you want to install the package into, then run the following command in the root directory of this repository:

``` pip install -e . ```

### Usage

The RESSEL classifier can be used as follows:

import the RESSEL classifier:

``` from ressel.ressel_classifier import ResselClassifier ```

Initialize an instance of the RESSEL classifier:

``` clf = ResselClassifier() ```

After initializing the classifier, it can be used as any scikit-learn classifier. For example, to train the classifier on a dataset X and labels y:

``` clf.fit(X, y) ```

where the labels in y should be specified as -1 when they are unknown.

To predict the labels for new data X:

``` clf.predict(X) ```