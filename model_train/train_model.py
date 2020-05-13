import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from standardiser import standardise
from ModifiedNB import ModifiedNB
from scipy import sparse
import numpy as np
import json_tricks
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
standardise.break_bonds.logger.setLevel(50)
standardise.neutralise.logger.setLevel(50)
standardise.unsalt.logger.setLevel(50)
standardise.rules.logger.setLevel(50)

def calc_descriptors(row, fp_type, fp_radius, con_desc_list, stdrise=True, hashed=False):
    rdmol = Chem.MolFromSmiles(row['smiles'])
    if rdmol:
        if stdrise:
            try:
                rdmol = standardise.run(rdmol, output_rules_applied=[])
            except:
                return None, None, None, None, None, None, None
        if fp_type == 'ecfp':
            if hashed:
                fps = AllChem.GetMorganFingerprintAsBitVect(rdmol, fp_radius, 2048, useFeatures=False)
                fps = {key: 1 for key in fps.GetOnBits()}
            else:
                # NB works better with binary features, removing FP feature freq (useCounts=False)
                fps = AllChem.GetMorganFingerprint(rdmol, fp_radius, useFeatures=False, useCounts=False).GetNonzeroElements()
        else:
            if hashed:
                fps = AllChem.GetMorganFingerprintAsBitVect(rdmol, fp_radius, 2048, useFeatures=True)
                fps = {key: 1 for key in fps.GetOnBits()}
            else:
                fps = AllChem.GetMorganFingerprint(rdmol, fp_radius, useFeatures=True, useCounts=False).GetNonzeroElements()
        alogp = Descriptors.MolLogP(rdmol) if 'alogp' in con_desc_list else None
        mw = Descriptors.MolWt(rdmol) if 'mw' in con_desc_list else None
        n_h_atoms = Descriptors.HeavyAtomCount(rdmol) if 'n_h_atoms' in con_desc_list else None
        rtb = Descriptors.NumRotatableBonds(rdmol) if 'rtb' in con_desc_list else None
        hbd = Descriptors.NumHDonors(rdmol) if 'hbd' in con_desc_list else None
        hba = Descriptors.NumHAcceptors(rdmol) if 'hba' in con_desc_list else None
        return fps, alogp, mw, n_h_atoms, rtb, hbd, hba
    else:
        return None, None, None, None, None, None, None

    
def load_data(fname, fp_type, fp_radius, con_desc_list, stdrise=True):
    df = pd.read_csv(fname)
    df['fps'], df['alogp'], df['mw'], df['n_h_atoms'], df['rtb'], df['hbd'], df['hba'] = zip(*df.apply(lambda row: calc_descriptors(row, fp_type, fp_radius, con_desc_list, stdrise), axis=1))
    del df['smiles']
    df = df[con_desc_list + ['fps', 'active']]
    df = df.dropna()
    X = df[con_desc_list + ['fps']]
    y = df['active']
    return X, y


class MMVModel(BaseEstimator):
    
    fp_type = None
    con_desc_list = []
    trained = False
    
    def __init__(self, fp_type, fp_radius, con_desc_list):
        self.fp_type = fp_type
        self.fp_radius = fp_radius
        self.con_desc_list = con_desc_list
    
    def _create_sparse_matrix(self, df, train=True):
        # transform the fp column to a sparse matrix
        if train:
            v_fps = DictVectorizer()
            fps = v_fps.fit_transform(df['fps'])
            self._total_fp_features = fps.shape[1]
            self.v_fps = v_fps
        else:
            fps = self.v_fps.transform(df['fps'])

        # transform continous descriptors to a sparse matrix
        if self.con_desc_list:
            if train:
                kbd = KBinsDiscretizer(n_bins=10, encode='onehot', strategy='quantile')
                descs = kbd.fit_transform(df[self.con_desc_list])
                self._total_continous_variable_bins = descs.shape[1]
                self.kbd = kbd
            else:
                descs = self.kbd.transform(df[self.con_desc_list])
                # get only the informative ones
                descs = descs[:, self.informative_cvb]
            sm = sparse.hstack([fps, descs])
        else:
            sm = fps
        return sm.tocsc()
    
    def _get_too_few(self, sm):
        # remove invariante features
        sel = VarianceThreshold()
        sel.fit(sm)
        return sel.get_support()

    def _get_informative_descs(self, sm):
        # Uninformative bins which contribute < 0.05 to the overall model score are removed
        active_idx = np.where(self.uninf_clf.classes_ == 1)[0][0]
        informative = np.where((np.abs(self.uninf_clf.feature_log_prob_[active_idx]) >= 0.05))[0]
        return informative
        
    def fit(self, X, y):
        sm = self._create_sparse_matrix(X)
        
        # remove "too few" descriptors
        tf_mask = self._get_too_few(sm)
        sm = sm[:, tf_mask]
        
        # train model to remove "non informative"
        self.uninf_clf = ModifiedNB()
        self.uninf_clf.fit(sm, y)
        
        # keep only informative descriptors
        informative = self._get_informative_descs(sm)
        sm = sm[:, informative]
        
        # update fps dict vectoriser to keep only informative fps
        # so data to predict will be correctly generated
        # also keep informative continous variable bins
        informative_fps = []
        informative_cvb = []
        max_fps_value = max(self.v_fps.vocabulary_.values())
        for i in informative:
            if i <= max_fps_value:
                informative_fps.append(i)
            else:
                informative_cvb.append(i - max_fps_value)
        self.v_fps = self.v_fps.restrict(informative_fps, indices=True)
        self._num_informative_fps = len(informative_fps)
        self.informative_cvb = informative_cvb
        self._num_informative_cvb = len(informative_cvb)

        # train the final model
        self.clf = ModifiedNB()
        self.clf.fit(sm, y)
        self.train_sm = sm
        self.trained = True
    
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.clf.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
    def predict_log_proba(self, X):
        if not self.trained:
            raise NotFittedError('You must first train the model.')
        sm = self._create_sparse_matrix(X, train=False)
        return self.clf.predict_log_proba(sm)
    
    def pp_predict(self, X):
        if not self.trained:
            raise NotFittedError('You must first train the model.')
        sm = self._create_sparse_matrix(X, train=False)
        return self.clf._joint_log_likelihood(sm)
    
    def to_json(self, out_fname):
        clf_dict = self.clf.__dict__
        clf_dict['fp_type'] = self.fp_type
        clf_dict['fp_radius'] = self.fp_radius
        clf_dict['con_desc_list'] = self.con_desc_list
        clf_dict['informative_cvb'] = self.informative_cvb

        clf_dict['fps_vectoriser'] = {'vocabulary_': self.v_fps.vocabulary_,
                                      'feature_names_': self.v_fps.feature_names_}

        if hasattr(self, 'kbd'):
            clf_dict['binariser'] = {'n_bins': self.kbd.n_bins,
                                     'n_bins_': self.kbd.n_bins_,
                                     'bin_edges_': [x.tolist() for x in self.kbd.bin_edges_],
                                     'categories': self.kbd._encoder.categories}
        json_tricks.dump(clf_dict, open(out_fname, 'w'), indent=2, sort_keys=True)

    def load_from_json(self, fname):
        # load the model
        import_data = json_tricks.load(open(fname))
        import_clf = ModifiedNB()
        import_clf.class_count_ = import_data['class_count_']
        import_clf.class_log_prior_ = import_data['class_log_prior_']
        import_clf.classes_ = import_data['classes_']
        import_clf.feature_count_ = import_data['feature_count_']
        import_clf.feature_log_prob_ = import_data['feature_log_prob_']
        self.clf =  import_clf

        # load the fps dict vectoriser
        v_fps = DictVectorizer()
        dv = import_data['fps_vectoriser']
        v_fps.vocabulary_ = {int(k):v for k, v in dv['vocabulary_'].items()}
        v_fps.feature_names_ = dv['feature_names_']
        self.v_fps = v_fps
        
        # load the continous variables binariser
        try:
            binariser = import_data['binariser']
            kbd = KBinsDiscretizer(n_bins=10, encode='onehot', strategy='quantile')
            kbd.n_bins = binariser['n_bins']
            kbd.n_bins_ = binariser['n_bins_']
            kbd.bin_edges_ = np.asarray([np.asarray(x) for x in binariser['bin_edges_']])
            encoder = OneHotEncoder()
            encoder.categories = binariser['categories']
            encoder._legacy_mode = False
            kbd._encoder = encoder
            self.kbd = kbd
        except Exception as e:
            pass
        
        # extra parameters
        self.trained = True
        self.con_desc_list = import_data['con_desc_list']
        self.fp_type = import_data['fp_type']
        self.fp_radius = import_data['fp_radius']
        self.informative_cvb = import_data['informative_cvb']


def calc_coverage(fname, fp_type, fp_radius, model):
    df = pd.read_csv(fname)
    fps_model = set(model.v_fps.vocabulary_.keys())
    coverage = []
    for s in df['smiles']:
        try:
            rdmol = Chem.MolFromSmiles(s)
        except:
            continue
        if fp_type == 'ecfp':
            fps = AllChem.GetMorganFingerprint(rdmol, fp_radius, useFeatures=False).GetNonzeroElements()
        else:
            fps = AllChem.GetMorganFingerprint(rdmol, fp_radius, useFeatures=True).GetNonzeroElements()
        fps_molecule = set(fps.keys())
        m = len(fps_molecule - fps_model)
        coverage.append((len(fps_molecule) - m) / len(fps_molecule))
    return coverage


def get_classification_report(model, X, y):
    report = {}
    preds = model.predict(X)

    # save the predictions
    pdf = pd.DataFrame(preds)
    pdf.columns = ['pred']
    pdf.to_csv('outputs/eMolecules_predictions_{}.csv'.format(model_name), index=False)
    del pdf

    roc_auc = roc_auc_score(y, model.predict_proba(X)[:,1])
    mt = matthews_corrcoef(y, preds)
    f1 = f1_score(y, preds)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)

    report.update({'precision': precision})
    report.update({'sensitivity': sensitivity})
    report.update({'specificity': specificity})
    report.update({'roc_auc_score': roc_auc})
    report.update({'matthews_corrcoef': mt})
    report.update({'f1_score': f1})
    report.update({'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}})
    return report 


def specificity(estimator, X_test, y_test):
    preds = estimator.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec = tn / (tn + fp)
    return spec


if __name__ == '__main__':

    model_configs = json_tricks.load(open('model_configs.json'))

    for model_name, conf in model_configs.items():

        fp_type = conf['fp_type']
        fp_radius = conf['fp_radius']
        con_desc_list = conf['con_desc_list']
        stdrise = conf['standardise']
        n_jobs = conf['cross_val_cores']

        print()
        print(model_name)
        print('---------------------')
        print('FP type: {}, continous descs: {}, standardise molecules: {}'.format(fp_type, con_desc_list, stdrise))
        print()

        # load data and calc descriptors
        X, y = load_data('training_set.csv', fp_type, fp_radius, con_desc_list, stdrise)

        # ----------------------------------------------------------------------------------
        # internal validation report
        scoring = {'roc_auc': 'roc_auc',
                   'precision': 'precision',
                   'sensitivity': 'recall',
                   'specificity': specificity,
                   'f1': 'f1',
                   'matthews_corrcoef': make_scorer(matthews_corrcoef)}

        clf = MMVModel(fp_type=fp_type, fp_radius=fp_radius, con_desc_list=con_desc_list)
        internal_report = cross_validate(clf, X, y, scoring=scoring, cv=KFold(n_splits=5, shuffle=True), n_jobs=n_jobs)
        json_tricks.dump(internal_report, open('outputs/internal_validation_report_{}.json'.format(model_name), 'w'), indent=2, sort_keys=True)


        # ----------------------------------------------------------------------------------
        # train the model with whole data

        clf = MMVModel(fp_type=fp_type, fp_radius=fp_radius, con_desc_list=con_desc_list)
        clf.fit(X, y)

        print('Total unique {} features in the dataset: {}'.format(fp_type, clf._total_fp_features))
        if clf.con_desc_list:
            print('Total continous variable bins for the dataset: {}'.format(clf._total_continous_variable_bins))
        print('Informative {} features in the dataset: {}'.format(fp_type, clf._num_informative_fps))
        if clf.con_desc_list:
            print('Informative continous variable bins in the dataset: {}'.format(clf._num_informative_cvb))

        # export the model to json
        clf.to_json('outputs/{}.json'.format(model_name))

        # calc coverage values for all molecules
        coverage = calc_coverage('eMolecules.csv', fp_type, fp_radius, clf)

        # load eMolecules set and create the external classification report
        X1, y1 = load_data('eMolecules.csv', fp_type, fp_radius, con_desc_list, stdrise)
        external_report = get_classification_report(clf, X1, y1)
        json_tricks.dump(external_report, open('outputs/external_validation_report_{}.json'.format(model_name), 'w'), indent=2, sort_keys=True)

        # print coverage and save values for all molecules
        print("FP coverage value:", sum(coverage) / len(coverage))
        json_tricks.dump(coverage, open('outputs/coverage_values_{}.json'.format(model_name), 'w'), indent=2, sort_keys=True)
