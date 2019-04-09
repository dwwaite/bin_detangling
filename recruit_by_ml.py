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
    esomConfidence = ParseEsomForErrorProfiling(userTable, options.coverageTable, options.use_bin_membership, coreContigTable)
    #_peakIntoObj(esomConfidence)
    #sys.exit()

    confidenceClassify = machineModelController.ClassifyByEnsemble(esomConfidence.ordValues, esomConfidence.contigList)
    confidenceCritical = ProduceConfidenceIntervals(esomConfidence, confidenceClassify, outputFileStub)

    ensembleResult = machineModelController.ClassifyByEnsemble(esomCloud.ordValues, esomCloud.contigList)
    ReportFinalAssignments(ensembleResult, confidenceCritical, outputFileStub)

###############################################################################

# region User input validation

def ExtractAndVerifyModelChoices(modelString):

    '''
        Input:
            1. A user-specified string on ML models to use in classification

        Action:
            1. Verify that there is data provided
            2. Ensure that each value is a valid model choice

        Result:
            1. Returns a list of model choices to the main() function
            2. If no valid choices are provided or remain, forces a system exit
    '''

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

    '''
        Input:
            1. A user-specified list of neurons to use in the input, hidden, and output layer of neural network
            2. The list of all features in the esomTable DataFrame
            3. A DataFrame with the columns Bin, ContigBase

        Action:
            1. Wrapped in flow control, cases are:
                1.1 Verifies that the user values are valid integers, and returns
                1.2 Determines default number of input, hidden, and output layer neurons to use with the rule
                    1.2.1 Input layer = Number of features for classification + 1
                    1.2.2 Hidden layer = Mean of input and output neuron counts
                    1.2.3 Output later = Number of classification options (number of bins in the coreContigTable)

        Result:
            1. A tuple in integers, reflecting input, hidden, and output layer neurons is returned to main()
            2. If no choices are valid, forces a system exit
    '''

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

    '''
        DEBUG ONLY

        Input:
            1. An eObj tuple with the variables ordValues, contigList, coreBinList, and originalBinList

        Action:
            1. Print the first 5 values of each variable within the eObj

        Result:
            1. No values are returned to calling function

    '''
    print('\n\nordValues (first 5)'); print(o.ordValues.head(5))
    print('\ncontigList (first 5)'); print(o.contigList[0:5] )
    print('\ncoreBinList (set)'); print( set(o.coreBinList) )
    print('\noriginalBinList (set)'); print( set(o.originalBinList) )

# endregion

# region User data import

def ParseEsomForTraining(esomTable, coverageTablePath, binMembershipFlag, coreContigTable):

    '''
        Input:
            1. A DataFrame with the following columns:
               1.1 V* - Coordinates from ESOM
               1.2 BinID - The text name of the bin from which the contig originally derives
               1.3 ContigName - The name of the contig fragment. Takes the form ContigBase|<i>
               1.4 ContigBase - The name of the original contig
            2. File path to a coverage table (optional)
            3. Parameter determining whether or not the original bin information should be included in the training table
            4. A DataFrame with the columns Bin, ContigBase

        Action:
            1. For each contig fragment in esomTable, take the average (median) V1 and V2 positions and use these as the coordinates the the full contig
            2. Parse these results into the DataFrame preTrainingFrame, with the columns:
                1.1 V* - New ~coordinates from ESOM
                1.2 Contig - The value of ContigBase used to group the fragments
                1.3 OriginalBin - The value of BinID
            3. If coverage values are provided, these are appended through the _appendCoverageTable() function, then scaled with the _scaleColumns() function
            4. If bin membership is request for training, these are appended through the _appendBinMembership() function
            5. The column CoreBin is appended to preTrainingFrame, via the _binMembershipGenerator() function
            6. preTrainingFrame is split into the eObj variables esomCore and esomCloud
                These objects capture which contigs used for training/validation (esomCore) and classification (esomCloud)

        Result:
            1. Two eObj tuples are returned to main(). These each have the values
                1.1 ordValues - The numeric value matrix to be used in training - V* and Coverage* columns
                1.2 contigList - The names of the contigs in ordValues
                1.3 coreBinList - If contig is a core contig, which bin it belonged to
                1.4 originalBinList - The original assignment of the contig

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

    '''
        Input:
            1. A DataFrame with the columns V*, Coverage* (optional), Contig, OriginalBin, and CoreBin
    
        Action:
            1. Splits the DataFrame into a new DataFrame retaining only numeric values needed for modeling
            2. Bind the text columns to new variables in the object

        Result:
            1. An eObj is returned to the calling function ParseEsomForTraining() or ParseEsomForErrorProfiling()
    '''

    eObj = namedtuple('eObj', 'ordValues contigList coreBinList originalBinList')
    ctL = dfSlice.pop('Contig')
    crL = dfSlice.pop('CoreBin')
    obL = dfSlice.pop('OriginalBin')

    return eObj(ordValues=dfSlice, contigList=ctL, coreBinList=crL, originalBinList=obL)

def ParseEsomForErrorProfiling(esomTable, coverageTablePath, binMembershipFlag, coreContigTable):

    '''
        Input:
            1. A DataFrame with the following columns:
               1.1 V* - Coordinates from ESOM
               1.2 BinID - The text name of the bin from which the contig originally derives
               1.3 ContigName - The name of the contig fragment. Takes the form ContigBase|<i>
               1.4 ContigBase - The name of the original contig
            2. File path to a coverage table (optional)
            3. Parameter determining whether or not the original bin information should be included in the training table
            4. A DataFrame with the columns Bin, ContigBase
    
        Action:
            1. Read in the esomTable as esomTableErr, and remap column maps to that of the training data
                1.1 ContigName => Contig
                1.2 BinID => OriginalBin
            2. If coverage values are provided, these are appended through the _appendCoverageTable() function, then scaled with the _scaleColumns() function
            3. If bin membership is request for training, these are appended through the _appendBinMembership() function
            4. The column CoreBin is appended to esomTableErr, via the _binMembershipGenerator() function
            5. esomTableErr is split into an eObj variable for non-core contigs

        Result:
            1. An eObj tuple is returned to main(), with the values
                1.1 ordValues - The numeric value matrix to be used in training - V* and Coverage* columns
                1.2 contigList - The names of the contigs in ordValues
                1.3 coreBinList - A list of '-' values, as we are only considering non-core contigs
                1.4 originalBinList - The original assignment of the contig
    '''

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

    '''
        Input:
            1. a DataFrame of unknown columns, but where some meet the criteria of V[digit]

        Action:
            1. Identify which columns in the DataFrame match the criteria via regex
        
        Result:
            1. A list of column names matching the pattern are returned to the calling function ParseEsomForTraining()
    '''

    return [ x for x in columnValues if re.match( r'V\d+$', x) ]

def _appendCoverageTable(baseFrame, coverageTablePath, contigColumnName):

    '''
        Input:
            1. A DataFrame with the columns V*, Contig, and OriginalBin
            2. The path to the table of coverage values
            3. The name for the column that identifies contigs in the baseFrame

        Action:
            1. Read the coverage table, and rename columns to [ contigColumnName, Coverage1, Coverage2, ... ]
            2. Merge the baseFrame and coverage table by a left join, discarding coverage values for contigs not in the ESOM

        Result:
            1. A modified baseFrame with the new columns Coverage* is returned to the calling function
            2. A list of coverage column names is returned to the calling function
    '''

    covFrame = pd.read_csv(coverageTablePath, sep='\t')

    ''' Overwrite the column names with predictable values '''
    colNames = [ 'Coverage{}'.format(i) for i in range(1, covFrame.shape[1]) ]
    colNames.insert(0, contigColumnName)
    covFrame.columns = colNames

    ''' Append the data to the ESOM table '''
    baseFrame = baseFrame.merge(covFrame, how='left', on=contigColumnName)
    return baseFrame, colNames[1:]

def _appendBinMembership(baseFrame, columnName):

    '''
        Input:
            1. A DataFrame with the columns V*, Coverage* (optional) , Contig, and one that identifies bins (expected: OriginalBin)
            3. The name for the column that identifies bins in baseFrame

        Action:
            1. Apply one-hot encoding to the bin column, and merge this DataFrame with baseFrame
            2. Drop the bin identifier column

        Result:
            1. A modified baseFrame with the in identifies encoded as numeric values is returned to the calling function
            2. A list of bin column names is returned to the calling function (not needed?)
    '''

    onehotFrame = pd.get_dummies( baseFrame[columnName] )
    newFrame = pd.concat( [baseFrame, onehotFrame], axis=1 )
    return newFrame.drop(columnName, axis=1), onehotFrame.columns

def _scaleColumns(_df, _columns, _method=None):

    '''
        Input:
            1. A DataFrame of unknown columns
            2. A list of columns to be normalised
            3. Method of normalisation. Currently not active, Z-transform by default

        Action:
            1. For each column, normalise according to calling specification

        Result:
            1. A DataFrame with the same columns as input, with relevant columns normalised, is returned to the calling function

    '''

    if _method:
        ''' Can add the option for more complicated scaling later on '''
        pass

    else:
        for c in _columns:
            _df[c] = preprocessing.scale( _df[c] )

    return _df

def _binMembershipGenerator(contigstoAssign, coreContigTable):

    '''
        Input:
            1. A list of all contigs needing to be attributed to a core bin
            2. A DataFrame of core contigs and their bin, with the columns Bin, ContigBase
    
        Action:
            1. Build a generator that maps contigs to either their core bin, or '-' for non-core contigs

        Result:
            1. No variables are returned to calling function. Acts as a generator that exhausts itself then terminates

    '''

    contigMap = { c: b for c, b in zip(coreContigTable.ContigBase, coreContigTable.Bin) }

    for contig in contigstoAssign:

        yield contigMap[contig] if contig in contigMap else '-'

# endregion

# region Model building and testing

def ProduceConfidenceIntervals(esomConfidence, esomConfidenceClassify, outputFileStub):

    '''
        Input:
            1. An eObj tuple, with the following values:
                1.1 ordValues: A DataFrame with columns V1, V2
                1.2 contigList: A list of conti fragment names
                1.3 coreBinList: A list of the bin assignment for contigs which are core
                1.4 originalBinList: A list of the bin assignment for all contigs (including core)
            2. A DataFrame with the columns Contig, Model, Iter, Bin, Confidence
            3. The prefix for writing output files

        Action:
            1. Bind the original bin information in esomConfidence to the esomConfidenceClassify DataFrame under the column OriginalBin
            2. Write this DataFrame to the harddrive, as an archiving step
            3. Calculate the upper confidence interval for the incorrect assignments on each bin (via the _computeConfidenceProfiles() function)

        Result:
            1. A tab-delimited table is saved to the harddrive.
            2. A dict of bin and critical value for incorrect confidence value is returned to main()
    '''

    ''' Classify each contig in the confidence profile set, then append the original/correct bin membership and log the results '''
    esomConfidenceClassify['OriginalBin'] = _attachBinMembership(esomConfidenceClassify, esomConfidence.contigList, esomConfidence.originalBinList)

    esomConfidenceClassify.to_csv('{}.conf_profile.txt'.format(outputFileStub), index=False, sep='\t')

    ''' Iterate through the assignments, and find the per bin upper bound of the 99% confidence interval for incorrectly assigned contigs '''
    CI_BOUND = 0.99
    confidenceCritical = _computeConfidenceProfiles(esomConfidenceClassify, outputFileStub, CI_BOUND)

    return confidenceCritical

def _attachBinMembership(confidenceResult, confidenceContigs, confidenceBinMembership):

    '''
        Input:
            1. A DataFrame with the columns Contig, Model, Iter, Bin, Confidence
            2. A list of the contig names
            3. A list of the bin membership of each contig, synchronised with confidenceContigs
    
        Action:
            1. Create an unsorted dict of the contigs and the bin they are assigned to { Contig => Bin }
            2. Build a list of bin assignments ordered according to confidenceResult Contig column
        
        Result:
            1. A list of bin assignment matching the order of confidenceResult
    '''

    binMapper = { c: b for c, b in zip(confidenceContigs, confidenceBinMembership) }
    return [ binMapper[c] for c in confidenceResult.Contig ]

def _computeConfidenceProfiles(confidenceResult, outputFileStub, CI_BOUND):

    '''
        Input:
            1. A DataFrame with the columns Contig, Model, Iter, Bin, Confidence
            2. The prefix for writing output files
            3. The confidence threshold for calculating confidence intervals.
                Currently set as 99%, not visible to user.

        Action:
            1. For each bin, calculate the lower and upper confidence intervals for the incorrectly assigned contigs.
            2. Index the results in a dict of { Bin => Upper CI }, which is hereafter refered to as the 'critical value'

        Result:
            1. A dict of bin and critical value for incorrect confidence value is returned to calling function ProduceConfidenceIntervals()
    '''

    binErrorProfile = {}

    for binName, df in confidenceResult.groupby('OriginalBin'):

        correctAssignmentConf = df[ df.Bin == df.OriginalBin ].Confidence
        incorrectAssignmentConf = df[ df.Bin != df.OriginalBin ].Confidence

        ciLow, ciHigh = st.t.interval(CI_BOUND, len(incorrectAssignmentConf)-1, loc=np.mean(incorrectAssignmentConf), scale=st.sem(incorrectAssignmentConf))
        binErrorProfile[binName] = ciHigh
        _plotConfidence(binName, correctAssignmentConf, incorrectAssignmentConf, outputFileStub)

    return binErrorProfile

def _plotConfidence(binName, correctValues, incorrectValues, outputFileStub):

    '''
        Input:
            1. The name of the bin to plot
            2. A list of confidence values for correctly assigned contigs
            3. A list of confidence values for incorrectly assigned contigs
            4. The prefix for writing output files
    
        Action:
            1. Create a histogram for the values in inputs 2 and 3. Shared x-axis to make it easy to compare spreads of values
            2. Y-axis is NOT shared, so the scale between correct/incorrect is often vastly different
        
        Result:
            1. A histogram figure is saved to the harddrive.
            2. No values are returned to main()
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    for a, t, c, v in zip( (ax1, ax2), ('correctly', 'incorrectly'), ('g', 'r'), (correctValues, incorrectValues) ):
        a.set_title( 'Confidence for {} assignment fragments'.format(t) )
        a.hist(v, bins=100, facecolor=c, alpha=0.75)
        a.set_xlim([0, 1])
        a.set_ylabel('Frequency')

    plt.xlabel('Confidence value')
    plt.savefig('{}.conf_profile_{}.png'.format(outputFileStub, binName), bbox_inches='tight')

def ReportFinalAssignments(ensembleResult, confidenceCritical, outputFileStub):

    '''
        Input:
            1. A DataFrame with the columns: Bin, Confidence, Contig, Iter, Model
            2. A dict of the critical value for false positive classification for each bin
            3. The prefix for writing output files
    
        Action:
            1. Append the critical values to the DataFrame, so that filtering can occur
            2. Filter the DataFrame, removing classifications without sufficient confidence
            3. For remaining results, find the most likely (greatest sum of confidences) bin for each contig
            4. Create a single table of all assignments, with the columns Bin, Contig, and Median Confidence
        
        Result:
            1. A tab-delimited table of the bin/contig/model/confidence scores, prior to critical value filtering
            2. A tab-delimited table of the bin/contig/confidence summaries.
            3. No values are returned to main()
    '''

    ensembleResult['Bin_specific_crit'] = [ confidenceCritical[b] for b in ensembleResult.Bin  ]
    ensembleResult.to_csv('{}.confidence_report.txt'.format(outputFileStub), sep='\t', index=False)

    ensembleResult.query('Confidence > Bin_specific_crit', inplace=True)

    outputBufferList = []
    for c, cdf in ensembleResult.groupby(['Contig']):

        topBin = cdf.groupby('Bin')['Confidence'].agg('sum').nlargest(1).index[0]
        topSlice = cdf[ cdf.Bin == topBin ]

        outputBufferList.append( { 'Bin': topBin, 'Contig': c, 'Median.Confidence': np.median( topSlice.Confidence ) } )
    
    outputFrame = pd.DataFrame(outputBufferList)
    outputFrame.to_csv('{}.confident_assignments.txt'.format(outputFileStub), sep='\t', index=False)

# endregion

###############################################################################
if __name__ == '__main__':
    main()