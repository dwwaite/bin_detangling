'''
    Class to handle the instantiation, training, validation, archiving and classification of machine learning tools in bin assignment

    TODO: Add documentation for the internal variables of MachineController
'''
# General modules
import sys, os, glob, warnings#, joblib
from collections import Counter
from operator import itemgetter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn modules for overhead
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.externals import joblib

# sklearn classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class MachineController():

    def __init__(self, classiferChoices, output_path, reload, nn_nodes, rf_trees, num_threads, rand_seed = None):

        self._validOptions = ['RF', 'NN', 'SVML', 'SVMR', 'SVMP']
        self._threadCount = num_threads
        self._outputPath = output_path
        self._output_path_map = { 'training_table': '{}.training_table.txt'.format(self._outputPath),
                                'training_plot_raster': '{}.training_table.png'.format(self._outputPath),
                                'training_plot_vector': '{}.training_table.svg'.format(self._outputPath),
                                'model_pattern': self._outputPath + '.model_{}.perm{}.pkl'
                              }

        '''
            TODO: Explain the logic of this implementation
        '''
        self._textMask = {'RF': 'Random forest', 'NN': 'Neural network', 'SVML': 'SVM (linear)', 'SVMR': 'SVM (radial basis function)', 'SVMP': 'SVM (polynomial)'}
        self._model_base = {'RF': None, 'NN': None, 'SVML': None, 'SVMR': None, 'SVMP': None}
        self._models = {'RF': [], 'NN': [], 'SVML': [], 'SVMR': [], 'SVMP': []}
        self._accuracy = None

        '''
            If this is a reload event, load in the pickled models.
            Otherwise, instantiate fresh models.
        '''
        if reload:
            
            print('User specified reloading of previously generated models...')
            self._importModels(classiferChoices)
        
        else:

            classiferChoices = set(classiferChoices)
            if 'RF' in classiferChoices:  self.instantiate_random_forest(rf_trees, rand_seed)
            if 'NN' in classiferChoices: self.instantiate_neural_network(nn_nodes, rand_seed)
            if 'SVML' in classiferChoices: self.instantiate_svm_linear(rand_seed)
            if 'SVMR' in classiferChoices: self.instantiate_svm_rbf(rand_seed)
            if 'SVMP' in classiferChoices: self.instantiate_svm_polynomial(rand_seed)

    def to_string(self):

        print( '\n\nMachineController details:\n' )

        print('Internal information')
        print( '  Supported models: {}'.format( ', '.join( self._textMask.values() ) ) )
        print( '  Output path: {}'.format(self._outputPath) )
        print( '  Classifier threads: {}'.format(self._threadCount) )
        print('')

        print('Model information')

        for opt in self._validOptions:
    
            print( '  {} model is instantiated: {}'.format( self._textMask[opt], self._model_base[opt] is not None ) )

        print( '\n  Models are {}.'.format( 'trained' if self._accuracy else 'untrained' ) ) 
        print('')

    # region Model creation, export and import

    def instantiate_random_forest(self, nTrees, seed = None):

        if seed:
            self._model_base['RF'] = RandomForestClassifier(n_estimators=nTrees, n_jobs=self._threadCount, random_state=seed)

        else:
            self._model_base['RF'] = RandomForestClassifier(n_estimators=nTrees, n_jobs=self._threadCount)

    def instantiate_neural_network(self, layerSizes, seed = None):

        if seed:
            self._model_base['NN'] = MLPClassifier(hidden_layer_sizes=layerSizes, max_iter=1000, activation='relu', solver='adam', random_state=seed)

        else:
            self._model_base['NN'] = MLPClassifier(hidden_layer_sizes=layerSizes, max_iter=1000, activation='relu', solver='adam')

    def _instantiate_svm(self, kernelType, seed = None):

        if seed:
            return svm.SVC(kernel=kernelType, gamma=0.001, C=100.0, probability=True, random_state=seed)

        else:
            return svm.SVC(kernel=kernelType, gamma=0.001, C=100.0, probability=True)
    
    def instantiate_svm_linear(self, seed = None):
        self._model_base['SVML'] = self._instantiate_svm('linear', seed)

    def instantiate_svm_rbf(self, seed = None):
        self._model_base['SVMR'] = self._instantiate_svm('rbf', seed)

    def instantiate_svm_polynomial(self, seed = None):
        self._model_base['SVMP'] = self._instantiate_svm('poly', seed)

    def save_models(self):

        try:

            for opt in self._validOptions:

                for i, model in enumerate(self._models[opt]):

                    joblib.dump(model, self._output_path_map['model_pattern'].format(opt, i + 1) )
        
        except:

            print('Error saving output files. Please check path carefully.')
            sys.exit()

    def _importModels(self, classiferChoices):

        importLengths = set()
        for opt in classiferChoices:

            candidateFiles = glob.glob( self._output_path_map['model_pattern'].format(opt, '*') )
            importLengths.add( len(candidateFiles) )

            assert(len(candidateFiles) > 0), 'Error: Unable to find files for model {}. Aborting...'.format(opt)
            assert(len(importLengths) == 1), 'Error: Inconsistent number of models loaded for model {}. Aborting...'.format(opt)

            self._models[opt] = [ joblib.load(candFile) for candFile in sorted(candidateFiles) ]
            self._model_base[opt] = clone( self._models[opt][0] )

            ''' Set to True to toggle the to_string report '''
            self._accuracy = True

    # endregion

    # region Model training

    def train_models(self, num_splits, esom_core, seed = None):

        model_accuracy_list = []

        validation_confidence_list = []

        split_iteration = 1
        for data_train, label_train, data_validate, label_validate in MachineController.YieldPermutations(num_splits, esom_core, seed):

                for model_type, model_base in self._model_base.items():

                    ''' Weird syntax, but not able to do a "if model_base:" call '''
                    if not model_base is None:

                        currModel = self._train_model(model_base, data_train, label_train)
                        self._models[ model_type ].append(currModel)

                        model_calls, model_conf = self._classify_data(model_type, currModel, data_validate)

                        ''' For each contig in the validation set, record the expected bin, called bin, and model + confidence '''
                        data_validate_conf = self._create_confidence_list(model_type, label_validate, model_calls, model_conf, split_iteration)
                        validation_confidence_list.extend( data_validate_conf )

                        ''' If all matches are correct, values are 1.0 by definition.
                            This is a workaround to the issues that crop up occassionally for MCC when denominator values are 0, creating a divide by zero error '''
                        model_eval_dict = self._compute_model_scores(model_type, split_iteration, label_validate, model_calls, model_conf)

                        model_accuracy_list.append( model_eval_dict )

                split_iteration += 1

        self._accuracy = pd.DataFrame(model_accuracy_list)
        self._accuracy = self._accuracy[ ['Model', 'Iteration', 'F1', 'MCC', 'ROC_AUC'] ]

        return pd.DataFrame(validation_confidence_list)

    def _compute_model_scores(self, model_type, split_iteration, label_validate, model_calls, model_conf):

        f1 = self._masked_f_score(label_validate, model_calls)
        roc_auc = self._masked_roc_auc_score(label_validate, model_calls, model_conf, model_type, split_iteration)
        mcc = self._masked_mcc_score(label_validate, model_calls, model_type, split_iteration)

        return { 'Model': model_type, 'Iteration': split_iteration, 'F1': f1, 'MCC': mcc, 'ROC_AUC':  roc_auc }

    def _train_model(self, _model_base, _data_train, _label_train):

        ''' Create a deep copy with the original parameters, then train it '''
        currModel = clone(_model_base, safe=True)
        currModel.fit(_data_train, _label_train)
        return currModel

    def _masked_roc_auc_score(self, exp, pred, conf, model, i):

        ''' The ROC-AUC is a bit different to F1 and MCC, which are just worked out from the confusion matrix.

            ROC-AUC requires a binary mask of correct/incorrect predictions, and the model confidences associated with them. In the event of all predictions being
                correct of incorrect, it will crash, so this is avoided with this function. Crashes are avoided, but NaN is returned. '''

        model_mask = [ x == y for x, y in zip(exp, pred) ]

        if len( set(model_mask) ) == 1:

            print( 'Warning: Invalid values encountered for ROC AUC on model {} (iteration {}).'.format(model, i) )
            return np.NaN

        else:
            return roc_auc_score(model_mask, conf)

    def _masked_mcc_score(self, exp, pred, model, i):

        ''' There's an edge case for the MCC calculation where if all predictions are to a single class, the calculation fails due to zeroes in the donominator.
        
            In this case, the return value is 0. This is fine, because 0 is the worst outcome for MCC and the summary will report it as rubbish.
            This function is just to suppress the warnings, since I'm happy with how they handle the case. '''

        with warnings.catch_warnings(record=True) as w:

            mcc = matthews_corrcoef(exp, pred)

            if len(w) > 0:

                warning_msg = str(w[-1].message)
                if 'invalid value encountered in double_scalars' in warning_msg:

                    print( 'Warning: Invalid values encountered for MCC on model {} (iteration {}).'.format(model, i) )
                    return np.NaN

                else:
                    print( warning_msg )

        return mcc

    def _masked_f_score(self, exp, pred):

        ''' sklearn implementation of the F1 throws warnings with weighted averaging method, claiming it is setting value to 0.
            After extensive testing, I'm happy that this is not the case and that it is returning a real F1.

            Use a context manager to filter out these specific warnings. '''

        with warnings.catch_warnings(record=True) as w:

            f1 = f1_score(exp, pred, average='weighted')

            if len(w) > 0:

                warning_msg = str(w[-1].message)
                if not 'F-score is ill-defined and being set to 0.0' in warning_msg:

                    print( warning_msg )

            return f1

    def report_training(self):

        ''' Write as table... '''
        self._accuracy.to_csv(self._output_path_map['training_table'], index=False, sep='\t')
        
        ''' Begin plotting '''

        model_seq = sorted( self._accuracy.Model.unique() )
        title_seq = ['F1 values', 'MCC values', 'ROC-AUC values']
        col_seq = [ 'F1', 'MCC', 'ROC_AUC' ]

        plt.clf()
        _, axes = plt.subplots(nrows=1, ncols=len(col_seq), sharey=True)

        for i, plot_array in self._summarise_model_scores(col_seq, model_seq):

            axes[i].set_title( title_seq[i] )
            p = axes[i].violinplot(plot_array, showmeans=False, showmedians=True)

            self._customise_violin_axes(axes[i], p, model_seq, plot_array)
            self._customise_violin_colours(p)

        plt.savefig(self._output_path_map['training_plot_raster'], bbox_inches='tight')
        plt.savefig(self._output_path_map['training_plot_vector'], bbox_inches='tight')

    def _summarise_model_scores(self, col_seq, model_seq):

        for i, c in enumerate(col_seq):

            plot_array = []

            for model in model_seq:

                df = self._accuracy[ self._accuracy.Model == model ]

                score_vector = [ v for v in df[c] if not np.isnan(v) ]
                if len(score_vector) > 0:
                    plot_array.append( score_vector )

                else:
                    plot_array.append( [0] )

            yield i, plot_array

    def _customise_violin_axes(self, a, p, model_seq, plot_array):

        ''' Determine the text for plot axes. Since there is a single value of 0 for models with no valid scores,
                need flow control to determine the correct label for each model '''

        x_tick_text = []
        for i, m in enumerate(model_seq):

            if len(plot_array[i]) == 1 and plot_array[i][0] == 0:
                x_tick_text.append('{} (n=0)'.format( self._textMask[m] ) )

            else:
                x_tick_text.append('{} (n={})'.format( self._textMask[m], len(plot_array[i]) ) )

        n_models = len( model_seq )
        plt.setp(a, xticks=[ y+1 for y in range(n_models) ], xticklabels=x_tick_text)

        plt.setp(a.get_xticklabels(), rotation=90)

    def _customise_violin_colours(self, _v):

        for b in _v['bodies']:
            b.set_facecolor('g')
            b.set_edgecolor('black')

        for p in ['cbars','cmins','cmaxes', 'cmedians']:
            _v[p].set_edgecolor('black')
            _v[p].set_linewidth(0.5)

    # endregion

    # region Classification of data

    def classify_by_ensemble(self, esom_obj):

        model_report = []

        for opt in self._validOptions:

                    for i, model in enumerate(self._models[opt]):

                        model_calls, model_conf = self._classify_data(opt, model, esom_obj.scaled_features)

                        iteration_report = self._create_confidence_list(opt, esom_obj.original_bin, model_calls, model_conf, i)
                        model_report.extend( iteration_report )

        return pd.DataFrame(model_report)

    def _classify_data(self, _model_type, _model, _data):

        n_entries = _data.shape[0]
        call_results = [None] * n_entries
        conf_results = [None] * n_entries

        hit_map = self._map_classes_to_names(_model)

        for i in range(n_entries):
            call_results[i], conf_results[i] = self._calc_confidence(_model, hit_map, _data[i,])

        return call_results, conf_results

    def _map_classes_to_names(self, model):
        return { i: c for i, c in enumerate(model.classes_) }

    def _calc_confidence(self, _model, _hit_map, _dataRow):

        pred = _model.predict_proba( _dataRow.reshape(1, -1) )[0]
        
        ''' Reshape the data as a dict of probabilities, then return the top value and confidence '''
        callsDict = { c: p for c, p in zip(_model.classes_, pred) }
        return self._extract_top_hit(callsDict)

    def _extract_top_hit(self, _classificationDict):
        topHit = max(_classificationDict.items(), key=itemgetter(1))[0]
        return topHit, _classificationDict[topHit]

    def _create_confidence_list(self, model_type, original_bins, model_calls, model_conf, iteration):

        model_report = [ { 'OriginalBin': original_bin,
                           'Model': model_type,
                           'Iter': iteration + 1,
                           'PredictedBin': call,
                           'Confidence': conf } for original_bin, call, conf in zip(original_bins, model_calls, model_conf) ]

        return model_report

    # endregion

    # region Static functions

    @staticmethod
    def YieldPermutations(num_splits, esom_obj, seed = None):

        if seed:
            spliterObj = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.5, random_state=seed)

        else:
            spliterObj = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.5)

        ''' Textbox implementation of the SSS function, so not really commented here. Basically, split the data into:
                1. dTrain = features for training data
                2. lTrain = labels matching dTrain rows to group (bin)
                3. dValidate = features not used in model training, for testing model accuracy
                4. lValidate = labels of the expected classification for dValidate '''        
        for tr, te in spliterObj.split(esom_obj.scaled_features, esom_obj.original_bin):

            dTrain = esom_obj.scaled_features[ tr, ]
            lTrain = [ list(esom_obj.original_bin)[x] for x in tr ]

            dValidate = esom_obj.scaled_features[ te, ]            
            lValidate = [ list(esom_obj.original_bin)[x] for x in te ]

            yield dTrain, lTrain, dValidate, lValidate

    # endregion