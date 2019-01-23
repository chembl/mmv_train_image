import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.exceptions import NotFittedError
from standardiser import standardise
from ModifiedNB import ModifiedNB
from scipy import sparse
import numpy as np
from tqdm import tqdm
import json_tricks
from rdkit import RDLogger


def calc_descriptors(row, fp_type, con_desc_list, stdrise=True):
    rdmol = Chem.MolFromSmiles(row['smiles'])
    if rdmol:
        if stdrise:
            try:
                rdmol = standardise.run(rdmol, output_rules_applied=[])
            except:
                return None, None, None, None, None, None, None
        if fp_type == 'ecfp':
            fps = AllChem.GetMorganFingerprint(rdmol, 3, useFeatures=False).GetNonzeroElements()
        else:
            fps = AllChem.GetMorganFingerprint(rdmol, 3, useFeatures=True).GetNonzeroElements()
        alogp = Descriptors.MolLogP(rdmol) if 'alogp' in con_desc_list else None
        mw = Descriptors.MolWt(rdmol) if 'mw' in con_desc_list else None
        n_h_atoms = Descriptors.HeavyAtomCount(rdmol) if 'n_h_atoms' in con_desc_list else None
        rtb = Descriptors.NumRotatableBonds(rdmol) if 'rtb' in con_desc_list else None
        hbd = Descriptors.NumHDonors(rdmol) if 'hbd' in con_desc_list else None
        hba = Descriptors.NumHAcceptors(rdmol) if 'hba' in con_desc_list else None
        return fps, alogp, mw, n_h_atoms, rtb, hbd, hba
    else:
        return None, None, None, None, None, None, None

    
def load_data(fname, fp_type, con_desc_list, stdrise=True):
    df = pd.read_csv(fname)
    df['fps'], df['alogp'], df['mw'], df['n_h_atoms'], df['rtb'], df['hbd'], df['hba'] = zip(*df.progress_apply(lambda row: calc_descriptors(row, fp_type, con_desc_list, stdrise), axis=1))
    del df['smiles']
    df = df[con_desc_list + ['fps', 'active']]
    df = df.dropna()
    X = df[con_desc_list + ['fps']]
    y = df['active']
    return X, y


class MMVModel:
    
    fp_type = None
    con_desc_list = []
    trained = False
    
    def __init__(self, fp_type, con_desc_list):
        self.fp_type = fp_type
        self.con_desc_list = con_desc_list
    
    def _create_sparse_matrix(self, df, train=True):
        # transform the fp column to a sparse matrix
        if train:
            v_fps = DictVectorizer()
            fps = v_fps.fit_transform(df['fps'])
            print('Total FP features for the dataset: {}'.format(fps.shape[1]))
            self.v_fps = v_fps
        else:
            fps = self.v_fps.transform(df['fps'])

        # transform continous descriptors to a sparse matrix
        if self.con_desc_list:
            if train:
                kbd = KBinsDiscretizer(n_bins=10, encode='onehot', strategy='quantile')
                descs = kbd.fit_transform(df[self.con_desc_list])
                print('Total continous variable bins: {}'.format(descs.shape[1]))
                self.kbd = kbd
            else:
                descs = self.kbd.transform(df[self.con_desc_list])
                # get only the informative ones
                descs = descs[:, self.informative_cvb]
            sm = sparse.hstack([fps, descs])
        else:
            sm = fps
        # NB works better with binary features, removing FP feature freq (set all to 1)
        sm.data = np.ones(sm.data.shape[0], dtype=np.int8)
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
        print('Informative FP features for the dataset: {}'.format(len(informative_fps)))
        self.informative_cvb = informative_cvb
        print('Informative continous variable bins: {}'.format(len(informative_cvb)))

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
        clf_dict = clf.clf.__dict__
        clf_dict['fp_type'] = self.fp_type
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
            kbd._encoder = encoder
            self.kbd = kbd
        except Exception as e:
            pass
        
        # extra parameters
        self.trained = True
        self.con_desc_list = import_data['con_desc_list']
        self.fp_type = import_data['fp_type']
        self.informative_cvb = import_data['informative_cvb']


def calc_coverage(fname, fp_type, model):
    df = pd.read_csv(fname)
    fps_model = set(model.v_fps.vocabulary_.keys())
    coverage = []
    for s in df['smiles']:
        try:
            rdmol = Chem.MolFromSmiles(s)
        except:
            continue
        if fp_type == 'ecfp':
            fps = AllChem.GetMorganFingerprint(rdmol, 3, useFeatures=False).GetNonzeroElements()
        else:
            fps = AllChem.GetMorganFingerprint(rdmol, 3, useFeatures=True).GetNonzeroElements()
        fps_molecule = set(fps.keys())
        m = len(fps_molecule - fps_model)
        coverage.append((len(fps_molecule) - m) / len(fps_molecule))
    return coverage


tqdm.pandas()

# load config file
model_configs = json_tricks.load(open('model_configs.json'))

for model_name, conf in model_configs.items():
    print()
    print(model_name)
    print('---------------------')

    fp_type = conf['fp_type'] 
    con_desc_list = conf['con_desc_list']
    stdrise = conf['standardise']

    # load data and calc descriptors
    X, y = load_data('training_set.csv', fp_type, con_desc_list, stdrise)

    # train the model
    clf = MMVModel(fp_type=fp_type, con_desc_list=con_desc_list)
    clf.fit(X, y)

    # export the model to json
    clf.to_json('{}.json'.format(model_name))

    # calc coverage values for all molecules
    coverage = calc_coverage('coverage_set.csv', fp_type, clf)
    
    # load eMolecules set and create classification report
    X1, y1 = load_data('eMolecules.csv', fp_type, con_desc_list, stdrise)
    preds = clf.predict(X1)

    pdf = pd.DataFrame(preds)
    pdf.columns = ['pred']
    pdf.to_csv('predictions_{}.csv'.format(model_name), index=False)

    report = {}
    roc_auc = roc_auc_score(y1, clf.predict_proba(X1)[:,1])
    mt = matthews_corrcoef(y1, preds)
    f1 = f1_score(y1, preds)
    tn, fp, fn, tp = confusion_matrix(y1, preds).ravel()
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

    json_tricks.dump(report, open('report_{}.json'.format(model_name), 'w'), indent=2, sort_keys=True)

    print("FP coverage value:", sum(coverage) / len(coverage))
    json_tricks.dump(coverage, open('coverage_values_{}.json'.format(model_name), 'w'), indent=2, sort_keys=True)
