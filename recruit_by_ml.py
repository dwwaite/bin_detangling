'''
    De novo run:
    python recruit_by_ml.py -e tests/recruit_bin.vizbin_table.csv -s 15 -t 2 -m RF,NN,SVML,SVMR,SVMP --use-bin-membership --evaluate-only -c tests/recruit_bin.coverage.txt tests/recruit_bin137.txt tests/recruit_bin341.txt tests/recruit_bin403.txt tests/recruit_bin417.txt

    Reboot run, skipping the SVMP model
    python recruit_by_ml.py -e tests/recruit_bin.vizbin_table.csv -s 15 -t 2 --reload -m RF,NN,SVML,SVMR --use-bin-membership -c tests/recruit_bin.coverage.txt tests/recruit_bin137.txt tests/recruit_bin341.txt tests/recruit_bin403.txt tests/recruit_bin417.txt

    Rapid/debug run
    python recruit_by_ml.py -e tests/recruit_bin.vizbin_table.csv -s 15 -m NN,SVMR --use-bin-membership --evaluate-only -c tests/recruit_bin.coverage.txt tests/recruit_bin137.txt tests/recruit_bin341.txt tests/recruit_bin403.txt tests/recruit_bin417.txt
    python recruit_by_ml.py -e tests/recruit_bin.vizbin_table.csv -s 15 -m NN,SVMR --use-bin-membership --reload -c tests/recruit_bin.coverage.txt tests/recruit_bin137.txt tests/recruit_bin341.txt tests/recruit_bin403.txt tests/recruit_bin417.txt

    Rapid/debug run with randomised contig sets
    Mock data is returning extremely high accuracy which is good, but concerning.
    Running with totally randomised values gives expected bad results, so this does not appear to be code error.

    python recruit_by_ml.py -e tests/recruit_bin.vizbin_table.csv -o shuffle.with_bins -s 15 -t 2 -m RF,NN,SVML,SVMR,SVMP --use-bin-membership --evaluate-only -c tests/recruit_bin.coverage.txt tests/shuffle0.txt tests/shuffle1.txt tests/shuffle2.txt tests/shuffle3.txt
    python recruit_by_ml.py -e tests/recruit_bin.vizbin_table.csv -o shuffle.no_bins -s 15 -t 2 -m RF,NN,SVML,SVMR,SVMP --evaluate-only -c tests/recruit_bin.coverage.txt tests/shuffle0.txt tests/shuffle1.txt tests/shuffle2.txt tests/shuffle3.txt

'''

import sys, os
import pandas as pd
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

# sklearn libraries for model training, classification, and saving
from sklearn import preprocessing

# My functions and classes
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateStringParameter, ValidateDataFrameColumns
from scripts.MachineModelController import MachineController

def main():

    # Parse options
    parser = OptionParser()
    usage = "usage: %prog [options] [contig list files]"

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

    options, coreContigLists = parser.parse_args()

    '''
        Validate the first set of user choices
    '''
    options.esomTable = ValidateFile(inFile=options.esomTable, fileTypeWarning='ESOM table', behaviour='abort')
    if options.coverageTable:
        options.coverageTable = ValidateFile(inFile=options.coverageTable, fileTypeWarning='coverage table', behaviour='skip')

    outputFileStub = options.output if options.output else os.path.splitext(options.esomTable)[0]

    '''
        User data is imported now, because it is required for setting default neural-network parameters.
        Table is split into the following variables:

        1. ESOM table - consists of columns used in modelling. ESOM coordinates as V*, Coverage values as Coverage*, Bin membership as bin names
        1. Contig list - 
        1. Feature list - 
    '''
    esomTable, contigList, binMembershipList = ParseEsomForTraining(options.esomTable, options.coverageTable, options.use_bin_membership, coreContigLists)
    featureList = esomTable.columns

    options.models = ExtractAndVerifyModelChoices(options.models)
    options.seed = ValidateInteger(userChoice=options.seed, parameterNameWarning='random seed', behaviour='skip')
    options.threads = ValidateInteger(userChoice=options.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    options.cross_validate = ValidateInteger(userChoice=options.cross_validate, parameterNameWarning='number of training splits', behaviour='default', defaultValue=10)
    options.rf_trees = ValidateInteger(userChoice=options.rf_trees, parameterNameWarning='decision trees', behaviour='default', defaultValue=1000)
    options.nn_nodes = ExtractAndVerifyLayerChoices(options.nn_nodes, featureList, coreContigLists)

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
    trainingData, trainingLabels, classificationData, classificationContigs = SplitData(esomTable, contigList, binMembershipList)

    if not options.reload:
        machineModelController.TrainModels(options.cross_validate, trainingData, trainingLabels, options.seed)

    ''' Step 3. '''
    if not options.reload:

        machineModelController.SaveModels()
        machineModelController.ReportTraining()

        if options.evaluate_only: sys.exit()

    ''' Step 4. '''

    #if not options.reload:

    # Internal validation of the confidence ranges seen for true and false positive results
    confidenceFrame, confidenceContigs, confidenceBinMembership = ParseEsomForErrorProfiling(options.esomTable, options.coverageTable, options.use_bin_membership, coreContigLists, featureList)
    confidenceResult = machineModelController.ClassifyByEnsemble(confidenceFrame, confidenceContigs)

    confidenceResult = AttachBinMembership(confidenceResult, confidenceContigs, confidenceBinMembership)
    confidenceDistributions = ComputeConfidenceProfiles(confidenceResult, outputFileStub)
    
    #errorModel = machineModelController.ClassifyByEnsemble()
    result = machineModelController.ClassifyByEnsemble(classificationData, classificationContigs)
    result.to_csv('debug.txt', index=False, sep='\t')

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
        print(modelString)
        modelChoices = [ ValidateStringParameter(userChoice=modelString, choiceTypeWarning='model choice', allowedOptions=supportedOpts, behaviour='skip') ]

    modelChoices = [ mC for mC in modelChoices if mC ]

    if len(modelChoices) > 0:
        return modelChoices

    else:

        print('No valid models selected. Aborting...')
        sys.exit()

def ExtractAndVerifyLayerChoices(neuronString, featureList, coreContigList):

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
    oLayer = len(coreContigList)
    hLayer = (iLayer + oLayer) / 2
    return iLayer, int(hLayer), oLayer

# endregion

# region User data import

def ParseEsomForTraining(esomTablePath, coverageTablePath, binMembershipFlag, coreContigLists):

    '''
        Begin by reading in the esomTable that was used in the expand_by_mcc.py script. For each contig, take the average (median) V1 and V2 positions,
        and use these as the coordinates the the full contig.

        Table consists of the following columns:
        1. V* - Coordinates from ESOM
        2. BinID - The text name of the bin from which the contig originally derives
        3. ContigName - The name of the contig fragment. Takes the form ContigBase|<i>
        4. ContigBase - The name of the original contig.

        Output table has the following columns
        1. V* - Normalised coordinates from ESOM
        2. Coverage* - Normalised coverage values from each sample
        3. Bin* - Normalised coordinates from ESOM
    '''

    ''' Read the minimum amount of data required to create the training frame '''
    esomTable = pd.read_csv(esomTablePath, sep='\t')

    '''
        TODO: Allow arbitrary number of V* columns
    '''
    preTrainingList = [ {'V1': np.median(df.V1), 'V2': np.median(df.V2), 'Contig': c, 'OriginalBin': list(df.BinID)[0]} for c, df in esomTable.groupby('ContigBase') ]
    preTrainingFrame = pd.DataFrame(preTrainingList)

    featureList = ['V1', 'V2']

    ''' If required, append coverage information and normalise it '''
    if coverageTablePath:
        preTrainingFrame, coverageFeatures = _appendCoverageTable(preTrainingFrame, coverageTablePath, 'Contig')
        preTrainingFrame = _scaleColumns(preTrainingFrame, coverageFeatures)

        featureList.extend( coverageFeatures )

    ''' If required, encode bin identity as new factors '''
    if binMembershipFlag:
        preTrainingFrame, additionalFeatures = _appendBinMembership(preTrainingFrame, 'OriginalBin')
        featureList.extend( additionalFeatures )

    '''
        Finally, update the bin informating.
             For core contigs, assign bin identity
             For other contigs, assign placeholder value
    '''
    trainingBinList = [ b for b in _binMembershipGenerator(preTrainingFrame.Contig, coreContigLists) ]

    ''' Pop off the Contigs column '''
    contigList = list( preTrainingFrame.pop('Contig') )

    return preTrainingFrame, contigList, trainingBinList

def ParseEsomForErrorProfiling(esomTablePath, coverageTablePath, binMembershipFlag, coreContigLists, featureList):

    '''
        Re-read the original ESOM table and format it into a per-fragment view of the data.
        Normalise the coverage values from this view to get slightly offset values to what was used in training (as the V* elements will also be different)

        Input and output data take the same form as ParseEsomForTraining().
    '''
    esomTable = pd.read_csv(esomTablePath, sep='\t')
    esomTable.rename(index=str, columns={'ContigBase': 'Contig'}, inplace=True )

    ''' Append and normalise data, as for training workflow '''
    if coverageTablePath:
        esomTable, coverageFeatures = _appendCoverageTable(esomTable, coverageTablePath, 'Contig')
        esomTable = _scaleColumns(esomTable, coverageFeatures)

    if binMembershipFlag: esomTable, _ = _appendBinMembership(esomTable, 'BinID')

    ''' Slice the esomTable down to just the expected columns '''

    esomTable['binMembership'] = [ b for b in _binMembershipGenerator(esomTable.Contig, coreContigLists) ]
    esomTable = esomTable[ esomTable.binMembership != '-' ]

    binMembership = list( esomTable['binMembership'] )
    contigNames = list( esomTable['Contig'] )

    return esomTable.loc[:,featureList], contigNames, binMembership

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

def _binMembershipGenerator(contigstoAssign, coreContigLists):

    contigMap = {}
    for coreContigList in coreContigLists:

        binFile = os.path.split(coreContigList)[1]
        binName = os.path.splitext(binFile)[0]

        nDict = { x.strip(): binName for x in open(coreContigList, 'r') }
        contigMap = { **contigMap, **nDict }

    for contig in contigstoAssign:

        yield contigMap[contig] if contig in contigMap else '-'

# endregion

# region Model building and testing

def SplitData(_esomTable, _contigList, _binMembershipList):

    inLocs = []
    outLocs = []
    trainLabels = []
    classifyContigs = []

    for i, b in enumerate(_binMembershipList):

        if b == '-':
            outLocs.append(i)
            classifyContigs.append( _contigList[i] )

        else:
            inLocs.append(i)
            trainLabels.append(b)

    trainDf = _esomTable.iloc[ inLocs , : ]
    classifyDf = _esomTable.iloc[ outLocs , : ]

    return trainDf, trainLabels, classifyDf, classifyContigs

def AttachBinMembership(confidenceResult, confidenceContigs, confidenceBinMembership):

    binMapper = { c: b for c, b in zip(confidenceContigs, confidenceBinMembership) }
    confidenceResult['OriginalBin'] = [ binMapper[c] for c in confidenceResult.Contig ]
    return confidenceResult

def ComputeConfidenceProfiles(confidenceResult, outputFileStub):

    for binName, df in confidenceResult.groupby('OriginalBin'):

        print( df.OriginalBin.unique() )
        correctAssignmentConf = df[ df.Bin == df.OriginalBin ].Confidence
        incorrectAssignmentConf = df[ df.Bin != df.OriginalBin ].Confidence

        _plotConfidence(binName, correctAssignmentConf, incorrectAssignmentConf, outputFileStub)


def _plotConfidence(binName, correctValues, incorrectValues, outputFileStub):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    for a, t, c, v in zip( (ax1, ax2), ('correctly', 'incorrectly'), ('g', 'r'), (correctValues, incorrectValues) ):
        a.set_title( 'Confidence for {} assignment fragments'.format(t) )
        a.hist(v, bins=100, facecolor=c, alpha=0.75)
        a.set_xlim([0, 1])

    plt.xlabel('Confidence value')
    plt.ylabel('Frequency')
    plt.savefig('{}.conf_profile_{}.png'.format(outputFileStub, binName), bbox_inches='tight')

# endregion

###############################################################################
if __name__ == '__main__':
    main()