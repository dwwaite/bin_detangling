'''
    Rapid/debug run - drop RF for increased speed
    python recruit_by_ml.py -e tests/mock.table.txt --evaluate-only tests/mock.table.core_table.txt
    python recruit_by_ml.py -e tests/mock.table.txt --reload -m RF,SVML,SVMP,SVMR tests/mock.table.core_table.txt
'''

import sys, os, re
import pandas as pd
import numpy as np
from optparse import OptionParser
from collections import namedtuple
import matplotlib.pyplot as plt

# sklearn libraries for model training, classification, and saving
from sklearn import preprocessing

# Confidence intervals for final assignment of contig
import scipy.stats as st

# My functions and classes
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateStringParameter, ValidateDataFrameColumns
from scripts.MachineModelController import MachineController

def main():

    # Parse options
    usageString = "usage: %prog [options] [core contig table]"
    parser = OptionParser(usage=usageString)

    # Basic options
    parser.add_option('-e', '--esom-table', help='A table produced by the vizbin_files_to_table.py script', dest='esomTable')
    parser.add_option('-c', '--coverage-table', help='A table of per-contig coverage values to use as features in classification (Default: None)', dest='coverageTable', default=None)
    parser.add_option('-o', '--output', help='An output prefix for all generated files (Default: Inferred from ESOM table)', dest='output', default=None)

    # Machine-learning parameters - general
    parser.add_option('-m', '--models', help='Comma-separated list of model types to build. Supported options are RF (random forest), NN (neural network), SVML (SVM with linear kernel), SVMR (SVM with radial basis function kernel), SVMP (SVM with polynomial kernel). Classification with be an ensemble of methods (Default: all)', dest='models', default='RF,NN,SVML,SVMR,SVMP')
    parser.add_option('-s', '--seed', help='Random seed for data permutation and model building', dest='seed', default=None)
    parser.add_option('-t', '--threads', help='Number of threads to use for model training and classification (Default: 1)', dest='threads', default=1)
    parser.add_option('-v', '--cross-validate', help='Perform X-fold cross validation in model building (Default: 10)', dest='cross_validate', default=10)
    parser.add_option('--evaluate-only', help='Only run as far as validating the training models, then terminate. Useful for deciding which models to use (Default: False)', dest='evaluate_only', action='store_true', default=False)
    parser.add_option('--reload', help='Load trained models from a previous run instead of training new ones (Default: False)', dest='reload', action='store_true', default=False)
    parser.add_option('--use-bin-membership', help='Encode the original binning information as a feature in prediction (Default: False)', dest='use_bin_membership', action='store_true', default=False)
    
    # Machine-learning parameters - classifier-specific
    parser.add_option('--rf-trees', help='Number of decision trees for random forest classification (Default: 1000)', dest='rf_trees', default=1000)
    parser.add_option('--nn-nodes', help='Comma-separated list of the number of neurons in the input, hidden, and output layers (Default: Input = number of features + 1, Output = number of classification outcomes, Hidden = mean of Input and Output)', dest='nn_nodes', default=None)

    options, args = parser.parse_args()
    coreContigFile = args[0]

    '''
        Pre-workflow overhead: Validation of user choices.

        This occurs in two stages;
            1. Import and validation of ESOM table and coverage table
            2. Validation of remaining parameters

        This order was chosen because in the event of a default neural network being requested, the layer sizes is calculated from the ESOM table structure

    '''
    options.esomTable = ValidateFile(inFile=options.esomTable, fileTypeWarning='ESOM table', behaviour='abort')
    if options.coverageTable:
        options.coverageTable = ValidateFile(inFile=options.coverageTable, fileTypeWarning='coverage table', behaviour='skip')

    outputFileStub = options.output if options.output else os.path.splitext(options.esomTable)[0]

    '''
        Original ESOM table is split into the following variables:
            1. ESOM table - A DataFrame for modelling. ESOM coordinates as V*, Coverage values as Coverage*, Bin membership as bin names
            2. Feature list - The column names in the ESOM table that are used for modelling.
            3. Contig list - A list of the contig fragment names, in the order they occur in the ESOM table
            4. Bin membership list - A list of the bins that each contig is found in. Order matches that of outputs 1 and 2.
    '''
    userTable = pd.read_csv(options.esomTable, sep='\t')
    ValidateDataFrameColumns(df=userTable, columnsRequired=['BinID', 'ContigName', 'ContigBase'])
    coreContigTable = pd.read_csv(coreContigFile, sep='\t')
    ValidateDataFrameColumns(df=coreContigTable, columnsRequired=['Bin', 'ContigBase'])
   
    esomCore, esomCloud = ParseEsomForTraining(userTable, options.coverageTable, options.use_bin_membership, coreContigTable)
    #_peakIntoObj(esomCore)
    #_peakIntoObj(esomCloud)
    #sys.exit()

    options.models = ExtractAndVerifyModelChoices(options.models)
    options.seed = ValidateInteger(userChoice=options.seed, parameterNameWarning='random seed', behaviour='skip')
    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.cross_validate = ValidateInteger(userChoice=options.cross_validate, parameterNameWarning='number of training splits', behaviour='default', defaultValue=10)
    options.rf_trees = ValidateInteger(userChoice=options.rf_trees, parameterNameWarning='decision trees', behaviour='default', defaultValue=1000)
    options.nn_nodes = ExtractAndVerifyLayerChoices(options.nn_nodes, esomCore.ordValues.columns, coreContigTable)

    '''

        Workflow:

            1. For each model specified, activate a classifier
               a) If this is a novel run, instantiate them
               b) If this is a restart, load them and skip to Step 4.
            2. Evaluate training data, record quality in terms of F1, MCC, and ROC-AUC
            3. Archive the data so far
               a) Archive models
               a) Terminate if in training-only mode
            4. Classify unknown data
               a) Report data as a table of the confidence of each assignment
               b) Report data as a simple tsv of contig/bin pairs
                  i) This will require some error modelling with the confidence of part (a)

    '''

    ''' Step 1. '''
    machineModelController = MachineController(options.models, outputFileStub, options.reload, options.nn_nodes, options.rf_trees, options.threads, options.seed)
    #machineModelController.to_string()

    ''' Step 2. '''

    if not options.reload:
        machineModelController.TrainModels(options.cross_validate, esomCore.ordValues, esomCore.coreBinList, options.seed)

    ''' Step 3. '''
    if not options.reload:

        machineModelController.SaveModels()
        machineModelController.ReportTraining()

        if options.evaluate_only: sys.exit()

    ''' Step 4. '''
    # Internal validation of the confidence ranges seen for true and false positive results

    esomConfidence = ParseEsomForErrorProfiling(userTable, options.coverageTable, options.use_bin_membership, coreContigTable)

    #_peakIntoObj(esomConfidence)

    confidenceClassify = machineModelController.ClassifyByEnsemble(esomConfidence.ordValues, esomConfidence.contigList)
    confidenceCritical = ProduceConfidenceIntervals(esomConfidence, confidenceClassify, outputFileStub)

    ensembleResult = machineModelController.ClassifyByEnsemble(esomCloud.ordValues, esomCloud.contigList)
    ReportFinalAssignments(ensembleResult, confidenceCritical, outputFileStub)

###############################################################################

# region User input validation

def ExtractAndVerifyModelChoices(modelString):

    if not modelString:

        print('No model provided. Aborting...')
        sys.exit()

    supportedOpts = ['RF', 'NN', 'SVML', 'SVMR', 'SVMP']

    if ',' in modelString:

        modelChoices = modelString.split(',')
        for i, mC in enumerate(modelChoices):

            modelChoices[i] = ValidateStringParameter(userChoice=mC, choiceTypeWarning='model choice', allowedOptions=supportedOpts, behaviour='skip')

    else:

        modelChoices = [ ValidateStringParameter(userChoice=modelString, choiceTypeWarning='model choice', allowedOptions=supportedOpts, behaviour='skip') ]

    modelChoices = [ mC for mC in modelChoices if mC ]

    if len(modelChoices) > 0:
        return modelChoices

    else:

        print('No valid models selected. Aborting...')
        sys.exit()

def ExtractAndVerifyLayerChoices(neuronString, featureList, coreContigTable):

    ''' If the user has specified an input '''
    if neuronString:

        try:

            assert(',' in neuronString), 'Unable to split neuron string'

            neuronVals = neuronString.split(',')
            assert(len(neuronVals) == 3), 'Incorrect number of layers described'

            neuronVals = tuple( map(int, neuronVals) )
            return neuronVals

        except AssertionError as ae:
            print( '{}. Aborting...'.format(ae) )
            sys.exit()

        except:
            print( 'Unable to convert values to integer. Aborting...' )
            sys.exit()

    ''' Otherwise, infer from the data '''
    iLayer = len(featureList) + 1
    oLayer = len( coreContigTable.Bin.unique() )
    hLayer = (iLayer + oLayer) / 2
    return iLayer, int(hLayer), oLayer

def _peakIntoObj(o):
    print('')
    print(o.ordValues.head(5))
    print(o.contigList[0:5] )
    print( set(o.coreBinList) )
    print( set(o.originalBinList) )

# endregion

# region User data import

def ParseEsomForTraining(esomTable, coverageTablePath, binMembershipFlag, coreContigTable):

    '''
        Begin by reading in the esomTable that was used in the expand_by_mcc.py script. For each contig, take the average (median) V1 and V2 positions,
        and use these as the coordinates the the full contig.

        Table consists of the following columns:
        1. V* - Coordinates from ESOM
        2. BinID - The text name of the bin from which the contig originally derives
        3. ContigName - The name of the contig fragment. Takes the form ContigBase|<i>
        4. ContigBase - The name of the original contig.

        Return objects are namedtuples with the following values:
        1. ordValues - The numeric value matrix to be used in training - V* and Coverage* columns
        2. contigList - The names of the contigs in ordValues
        3. coreBinList - If contig is a core contig, which bin it belonged to
        4. originalBinList - The original assignment of the contig

        Also returns a list of the features in the ordValues frames. May not be needed?
    '''

    featureList = _identifyFeatureColumns( esomTable.columns )

    preTrainingList = []
    for c, df in esomTable.groupby('ContigBase'):

        dataRecord = { v: np.median(df[v]) for v in featureList }
        dataRecord['Contig'] = c
        dataRecord['OriginalBin'] = list(df.BinID)[0]

        preTrainingList.append( dataRecord )

    preTrainingFrame = pd.DataFrame(preTrainingList)

    ''' If required, append coverage information and normalise it.
        Normalisation is not performed if model training is done of ordination values alone '''
    if coverageTablePath:
        preTrainingFrame, coverageFeatures = _appendCoverageTable(preTrainingFrame, coverageTablePath, 'Contig')
        preTrainingFrame = _scaleColumns(preTrainingFrame, featureList)
        preTrainingFrame = _scaleColumns(preTrainingFrame, coverageFeatures)

    ''' If required, encode bin identity as new factors '''
    if binMembershipFlag:
        preTrainingFrame, _ = _appendBinMembership(preTrainingFrame, 'OriginalBin')

    '''
        Update the bin informating.
             For core contigs, assign bin identity
             For other contigs, assign placeholder value
        Finally, pop off the text columns
    '''
    preTrainingFrame['CoreBin'] = [ b for b in _binMembershipGenerator(preTrainingFrame.Contig, coreContigTable) ]

    validBins = list( coreContigTable.Bin.unique() )
    esomCore = _bindToTableObj( preTrainingFrame[ preTrainingFrame.CoreBin != '-' ] )
    esomCloud = _bindToTableObj( preTrainingFrame[ (preTrainingFrame.CoreBin == '-') &
                                                   (preTrainingFrame.OriginalBin.isin(validBins)) ] )

    return esomCore, esomCloud

def _bindToTableObj(dfSlice):

    eObj = namedtuple('eObj', 'ordValues contigList coreBinList originalBinList')
    ctL = dfSlice.pop('Contig')
    crL = dfSlice.pop('CoreBin')
    obL = dfSlice.pop('OriginalBin')

    return eObj(ordValues=dfSlice, contigList=ctL, coreBinList=crL, originalBinList=obL)

def ParseEsomForErrorProfiling(esomTable, coverageTablePath, binMembershipFlag, coreContigTable):

    '''
        Re-read the original ESOM table and format it into a per-fragment view of the data.
        Normalise the coverage values from this view to get slightly offset values to what was used in training (as the V* elements will also be different)

        Input and output data take the same form as ParseEsomForTraining().
    '''
    esomTableErr = esomTable.rename(index=str, columns={'ContigName': 'Contig', 'BinID': 'OriginalBin'} )

    ''' Append and normalise data, as for training workflow '''
    if coverageTablePath:
        esomTableErr, coverageFeatures = _appendCoverageTable(esomTableErr, coverageTablePath, 'Contig')
        esomTableErr = _scaleColumns(esomTableErr, coverageFeatures)

    if binMembershipFlag: esomTableErr, _ = _appendBinMembership(esomTableErr, 'BinID')

    ''' Slice the esomTableErr down to just the expected columns '''
    fragmentNames = esomTableErr.pop('ContigBase')
    esomTableErr['CoreBin'] = [ b for b in _binMembershipGenerator(fragmentNames, coreContigTable) ]

    return _bindToTableObj( esomTableErr[ esomTableErr.CoreBin != '-' ] )

def _identifyFeatureColumns(columnValues):
    ''' Simple regex for a V with any number of digits trailing. It is highly unlikely we will ever use more than half a dozen '''
    return [ x for x in columnValues if re.match( r'V\d+$', x) ]

def _appendCoverageTable(baseFrame, coverageTablePath, contigColumnName):

    covFrame = pd.read_csv(coverageTablePath, sep='\t')

    ''' Overwrite the column names with predictable values '''
    colNames = [ 'Coverage{}'.format(i) for i in range(1, covFrame.shape[1]) ]
    colNames.insert(0, contigColumnName)
    covFrame.columns = colNames

    ''' Append the data to the ESOM table '''
    baseFrame = baseFrame.merge(covFrame, how='left', on=contigColumnName)
    return baseFrame, colNames[1:]

def _appendBinMembership(baseFrame, columnName):

    onehotFrame = pd.get_dummies( baseFrame[columnName] )
    newFrame = pd.concat( [baseFrame, onehotFrame], axis=1 )
    return newFrame.drop(columnName, axis=1), onehotFrame.columns

def _scaleColumns(_df, _columns, _method=None):

    if _method:
        ''' Can add the option for more complicated scaling later on '''
        pass

    else:
        for c in _columns:
            _df[c] = preprocessing.scale( _df[c] )

    return _df

def _binMembershipGenerator(contigstoAssign, coreContigTable):

    contigMap = { c: b for c, b in zip(coreContigTable.ContigBase, coreContigTable.Bin) }

    for contig in contigstoAssign:

        yield contigMap[contig] if contig in contigMap else '-'

# endregion

# region Model building and testing

def ProduceConfidenceIntervals(esomConfidence, esomConfidenceClassify, outputFileStub):

    ''' Classify each contig in the confidence profile set, then append the original/correct bin membership and log the results '''
    esomConfidenceClassify['OriginalBin'] = _attachBinMembership(esomConfidenceClassify, esomConfidence.contigList, esomConfidence.originalBinList)

    esomConfidenceClassify.to_csv('{}.conf_profile.txt'.format(outputFileStub), index=False, sep='\t')


    ''' Iterate through the assignments, and find the per bin upper bound of the 99% confidence interval for incorrectly assigned contigs '''
    CI_BOUND = 0.99
    confidenceCritical = _computeConfidenceProfiles(esomConfidenceClassify, outputFileStub, CI_BOUND)

    return confidenceCritical

def _attachBinMembership(confidenceResult, confidenceContigs, confidenceBinMembership):

    binMapper = { c: b for c, b in zip(confidenceContigs, confidenceBinMembership) }
    return [ binMapper[c] for c in confidenceResult.Contig ]

def _computeConfidenceProfiles(confidenceResult, outputFileStub, CI_BOUND):

    binErrorProfile = {}

    for binName, df in confidenceResult.groupby('OriginalBin'):

        correctAssignmentConf = df[ df.Bin == df.OriginalBin ].Confidence
        incorrectAssignmentConf = df[ df.Bin != df.OriginalBin ].Confidence

        ciLow, ciHigh = st.t.interval(CI_BOUND, len(incorrectAssignmentConf)-1, loc=np.mean(incorrectAssignmentConf), scale=st.sem(incorrectAssignmentConf))
        binErrorProfile[binName] = ciHigh
        _plotConfidence(binName, correctAssignmentConf, incorrectAssignmentConf, outputFileStub)

    return binErrorProfile

def _plotConfidence(binName, correctValues, incorrectValues, outputFileStub):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    for a, t, c, v in zip( (ax1, ax2), ('correctly', 'incorrectly'), ('g', 'r'), (correctValues, incorrectValues) ):
        a.set_title( 'Confidence for {} assignment fragments'.format(t) )
        a.hist(v, bins=100, facecolor=c, alpha=0.75)
        a.set_xlim([0, 1])
        a.set_ylabel('Frequency')

    plt.xlabel('Confidence value')
    plt.savefig('{}.conf_profile_{}.png'.format(outputFileStub, binName), bbox_inches='tight')

def ReportFinalAssignments(ensembleResult, confidenceCritical, outputFileStub):

    ensembleResult['Bin_specific_crit'] = [ confidenceCritical[b] for b in ensembleResult.Bin  ]
    ensembleResult.to_csv('{}.confidience_assign.txt'.format(outputFileStub), sep='\t', index=False)

    ensembleResult.query('Confidence > Bin_specific_crit', inplace=True)

    for b, df in ensembleResult.groupby('Bin'):
        open( '{}.{}.assigned_contigs.txt'.format(outputFileStub, b), 'w' ).write( '\n'.join(df.Contig) )

# endregion

###############################################################################
if __name__ == '__main__':
    main()