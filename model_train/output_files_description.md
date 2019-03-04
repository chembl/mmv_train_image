# Output files description



## model.json

<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/modeljson.png" width="300" ></a>

json file consisting of a dict with 14 keys. This file **contains ONLY information** related to informative descriptors. Dumped using [json_tricks](https://pypi.org/project/json_tricks/) library.

- alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing). Defaults to 1 as PP/KNIME models are using it like this
- binariser: Continuous variables are splitted in 10 bins using scikit-learn [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
    - bin_edges_: Number of bins per feature. Features are identified by an integer id and are sorted in the same order than model_configs.json file. (ex: for model3 feature 0 is alogp)
    - categories: Unique values per feature, needed to configure the [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) required by the [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
- class_count: Number of samples encountered for each class during fitting (0: inactive, 1: active) 
- class_log_prior_: Does not apply [in our model](https://github.com/chembl/ModifiedNB/blob/master/ModifiedNB/ModifiedNB.py#L22) as PP/KNIME model didn't use it. The parameter exists as we are inheriting from scikit-learn [BaseDiscreteNB](https://github.com/scikit-learn/scikit-learn/blob/cf9a74059851c91b3b7c5b354b85e1e87ff628bf/sklearn/naive_bayes.py#L446)
- class_prior: Does neither apply. Always null
- classes_: List with different classes in the model (0: inactive, 1: active) 
- con_desc_list: List with continuous descriptor names used to train the model
- feature_count_: Number of samples encountered for each (class, feature) during fitting
- feature_log_prob_: Empirical log probability of features given a class, P(x_i|y). Calculated [here](https://github.com/chembl/ModifiedNB/blob/master/ModifiedNB/ModifiedNB.py#L34) in our model
- fit_prior: Does not apply. Always false
- fp_type: Type of fingerprint used to train the model
- fp_radius: Radius of the fingerprint used to train the model
- fps_vectoriser: From scikit-learn [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html), used to generate sparse matrices that made possible to train the model using big datasets
    - feature_names_: A list of length n_features containing the feature names
    - vocabulary_: A dictionary mapping feature names to feature indices
- informative_cvb: List of informative continuous variable bins


## coverage_values_model.json

json file with a list of coverage values for each molecule listed in the coverage_set file. Boxplots like the ones from the original publication will be generated with them for each model

## internal_validation_report.json

dumped using [json_tricks](https://pypi.org/project/json_tricks/) library. Contains different prediction performance metrics for each of the 5 splits of the dataset in the 5 K-Fold cross validation

## external_validation_report.json

contains eMolecules dataset prediction performance metrics

## predictions_model.csv

contains the predicted values(0/1) for each molecule in the eMolecules dataset