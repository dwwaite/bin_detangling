'''
    Class to handle the instantiation, training, validation, archiving and classification of machine learning tools in bin assignment

    TODO: Add documentation for the internal variables of MachineController
    TODO: Implement ThreadManager class to speed up assignment using RandomForest
'''
# General modules
import sys, os, glob
from collections import Counter
from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt

# My modules
from scripts.ThreadManager import ThreadManager

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
        self._outputPathMap = { 'training_table': '{}.training_table.txt'.format(self._outputPath),
                                'training_plot_raster': '{}.training_table.png'.format(self._outputPath),
                                'training_plot_vector': '{}.training_table.svg'.format(self._outputPath),
                                'model_pattern': self._outputPath + '.model_{}.perm{}.pkl'
                              }

        '''
            TODO: Explain the logic of this implementation
        '''
        self._textMask = {'RF': 'Random forest', 'NN': 'Neural network', 'SVML': 'SVM (linear)', 'SVMR': 'SVM (radial basis function)', 'SVMP': 'SVM (polynomial)'}
        self._modelBase = {'RF': None, 'NN': None, 'SVML': None, 'SVMR': None, 'SVMP': None}
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
    
            print( '  {} model is instantiated: {}'.format( self._textMask[opt], self._modelBase[opt] is not None ) )

        print( '\n  Models are {}.'.format( 'trained' if self._accuracy else 'untrained' ) ) 
        print('')

    # region Model creation, export and import

    def instantiate_random_forest(self, nTrees, seed = None):

        if seed:
            self._modelBase['RF'] = RandomForestClassifier(n_estimators=nTrees, random_state=seed)

        else:
            self._modelBase['RF'] = RandomForestClassifier(n_estimators=nTrees)

    def instantiate_neural_network(self, layerSizes, seed = None):

        if seed:
            self._modelBase['NN'] = MLPClassifier(hidden_layer_sizes=layerSizes, max_iter=1000, activation='relu', solver='adam', random_state=seed)

        else:
            self._modelBase['NN'] = MLPClassifier(hidden_layer_sizes=layerSizes, max_iter=1000, activation='relu', solver='adam')

    def _instantiate_svm(self, kernelType, seed = None):

        if seed:
            return svm.SVC(kernel=kernelType, gamma=0.001, C=100.0, probability=True, random_state=seed)

        else:
            return svm.SVC(kernel=kernelType, gamma=0.001, C=100.0, probability=True)
    
    def instantiate_svm_linear(self, seed = None):
        self._modelBase['SVML'] = self._instantiate_svm('linear', seed)

    def instantiate_svm_rbf(self, seed = None):
        self._modelBase['SVMR'] = self._instantiate_svm('rbf', seed)

    def instantiate_svm_polynomial(self, seed = None):
        self._modelBase['SVMP'] = self._instantiate_svm('poly', seed)

    def SaveModels(self):

        try:

            for opt in self._validOptions:

                for i, model in enumerate(self._models[opt]):

                    joblib.dump(model, self._outputPathMap['model_pattern'].format(opt, i + 1) )
        
        except:

            print('Error saving output files. Please check path carefully.')
            sys.exit()

    def _importModels(self, classiferChoices):

        importLengths = set()
        for opt in classiferChoices:

            candidateFiles = glob.glob( self._outputPathMap['model_pattern'].format(opt, '*') )
            importLengths.add( len(candidateFiles) )

            assert(len(candidateFiles) > 0), 'Error: Unable to find files for model {}. Aborting...'.format(opt)
            assert(len(importLengths) == 1), 'Error: Inconsistent number of models loaded for model {}. Aborting...'.format(opt)

            self._models[opt] = [ joblib.load(candFile) for candFile in sorted(candidateFiles) ]
            self._modelBase[opt] = clone( self._models[opt][0] )

            ''' Set to True to toggle the to_string report '''
            self._accuracy = True

    # endregion

    # region Model training

    def TrainModels(self, numSplits, trainingData, trainingLabels, seed = None):

        classificationAccuracyList = []

        splitIteration = 1
        for dataTrain, labelTrain, dataValidate, labelValidate in MachineController.YieldPermutations(numSplits, trainingData, trainingLabels, seed):

                for modelType, modelBase in self._modelBase.items():

                    ''' Weird syntax, but not able to do a "if modelBase:" call '''
                    if not modelBase is None:

                        currModel = self._trainModel(modelBase, dataTrain, labelTrain)
                        self._models[ modelType ].append(currModel)

                        modelCalls, modelConf = self._classifyData(modelType, currModel, dataValidate)
                        modelMask = [ x == y for x, y in zip(labelValidate, modelCalls) ]

                        ''' If all matches are correct, values are 1.0 by definition '''
                        if len( set(modelMask) ) == 1 and modelMask[0]:

                            classificationAccuracyList.append( { 'Model': modelType, 'Iteration': splitIteration, 'F1': 1.0, 'MCC': 1.0, 'ROC_AUC':  1.0 } )

                        else:

                            classificationAccuracyList.append( { 'Model': modelType,
                                                                'Iteration': splitIteration,
                                                                'F1': f1_score(labelValidate, modelCalls, average='weighted'),
                                                                'MCC': matthews_corrcoef(labelValidate, modelCalls),
                                                                'ROC_AUC':  roc_auc_score(modelMask, modelConf) } )

                splitIteration += 1

        self._accuracy = pd.DataFrame(classificationAccuracyList)
        self._accuracy = self._accuracy[ ['Model', 'Iteration', 'F1', 'MCC', 'ROC_AUC'] ]

    def _trainModel(self, _modelBase, _dataTrain, _labelTrain):

        ''' Create a deep copy with the original parameters, then train it '''
        currModel = clone(_modelBase, safe=True)
        currModel.fit(_dataTrain, _labelTrain)
        return currModel

    def ReportTraining(self):

        ''' Write as table... '''
        self._accuracy.to_csv(self._outputPathMap['training_table'], index=False, sep='\t')
        
        ''' Organise data ready for violin plots '''

        modelSequence = sorted( self._accuracy.Model.unique() )
        f1 = [None] * len(modelSequence)
        mcc = [None] * len(modelSequence)
        auc = [None] * len(modelSequence)

        for i, model in enumerate(modelSequence):

            df = self._accuracy[  self._accuracy.Model == model  ]
            f1[i] = df.F1
            mcc[i] = df.MCC
            auc[i] = df.ROC_AUC

        ''' Begin plotting '''
        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)

        seq_ax = (ax1, ax2, ax3)
        seq_title = ('F1 values', 'MCC values', 'ROC-AUC values')
        seq_value = (f1, mcc, auc)

        for a, t, v in zip(seq_ax, seq_title, seq_value):

            a.set_title(t)
            p = a.violinplot(v, showmeans=False, showmedians=True)
            self._customiseViolinPlot(p)

            plt.setp(a, xticks=[ y+1 for y in range(len(modelSequence)) ], xticklabels=[ self._textMask[m] for m in modelSequence ])
            plt.setp(a.get_xticklabels(), rotation=90)

        plt.savefig(self._outputPathMap['training_plot_raster'], bbox_inches='tight')
        plt.savefig(self._outputPathMap['training_plot_vector'], bbox_inches='tight')

    def _customiseViolinPlot(self, _v):

        for b in _v['bodies']:
            b.set_facecolor('g')
            b.set_edgecolor('black')

        for p in ['cbars','cmins','cmaxes', 'cmedians']:
            _v[p].set_edgecolor('black')
            _v[p].set_linewidth(0.5)


    # endregion

    # region Classification of data

    def ClassifyByEnsemble(self, classificationData, classificationContigs):

        modelReport = []

        for opt in self._validOptions:

                    for i, model in enumerate(self._models[opt]):

                        modelCalls, modelConf = self._classifyData(opt, model, classificationData)

                        for contig, call, conf in zip(classificationContigs, modelCalls, modelConf):

                            modelReport.append( {'Contig': contig, 'Model': opt, 'Iter': i + 1, 'Bin': call, 'Confidence': conf} )

        return pd.DataFrame(modelReport)

    def _classifyData(self, _modelType, _model, _data):

        nEntries = _data.shape[0]
        callResults = [None] * nEntries
        confResults = [None] * nEntries

        hitMap = MachineController.MapClassesToNames(_model)
        
        '''
            Flow control for RF vs other models
            RF does not natively determine a confidence value, whereas the NN and SVM variants all do under the same parameter name.
        '''
        if _modelType == 'RF':

            ''' Store an index of the row being classified, so the results are stored in the correct order '''
            tManager = ThreadManager(self._threadCount, self._calcConfidenceRf)
            funcArgList = [ (_model, hitMap, i, _data.iloc[i,:], tManager.queue) for i in range(nEntries) ]

            tManager.ActivateMonitorPool(sleepTime=1, funcArgs=funcArgList)

            for i, hit, conf in tManager.results:
                callResults[i], confResults[i] = hit, conf

        else:
            for i in range(nEntries):
                callResults[i], confResults[i] = self._calcConfidence(_model, hitMap, _data.iloc[i,:])

        return callResults, confResults

    def _calcConfidenceRf(self, args):

        _rfModel, _hitMap, _index, _dataRow, _q = args

        ''' Create a list of the individual calls for each decision tree '''
        nTrees = len(_rfModel.estimators_)
        calls = [ _rfModel.estimators_[i].predict( _dataRow.values.reshape(1, -1) )[0] for i in range(nTrees) ]

        ''' Reshape the data as a dict of occurences, then return the top value and confidence '''
        callsDict = dict( Counter( [ _hitMap[c] for c in calls ] ) )
        topHit, topConf = self._extractTopHit(callsDict)

        _q.put( (_index, topHit, topConf / nTrees) )

    def _calcConfidence(self, _model, _hitMap, _dataRow):

        pred = _model.predict_proba( _dataRow.values.reshape(1, -1) )[0]
        
        ''' Reshape the data as a dict of probabilities, then return the top value and confidence '''
        callsDict = { c: p for c, p in zip(_model.classes_, pred) }
        return self._extractTopHit(callsDict)

    def _extractTopHit(self, _classificationDict):
        topHit = max(_classificationDict.items(), key=itemgetter(1))[0]
        return topHit, _classificationDict[topHit]

    # endregion

    # region Static functions

    @staticmethod
    def YieldPermutations(num_splits, trainingData, trainingLabels, seed = None):

        if seed:
            spliterObj = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.5, random_state=seed)

        else:
            spliterObj = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.5)
        
        for tr, te in spliterObj.split(trainingData, trainingLabels):
            dTrain, dValidate = trainingData.iloc[ tr, : ], trainingData.iloc[ te, : ]
            lTrain = [ list(trainingLabels)[x] for x in tr ]
            lValidate = [ list(trainingLabels)[x] for x in te ]
            yield dTrain, lTrain, dValidate, lValidate

    @staticmethod
    def MapClassesToNames(model):
        return { i: c for i, c in enumerate(model.classes_) }

    # endregion