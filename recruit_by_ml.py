'''
    Rapid/debug run - drop RF for increased speed
    python recruit_by_ml.py -e tests/mock.table.txt -m NN,SVML,SVMP,SVMR --evaluate-only tests/mock.table.core_table.txt
    python recruit_by_ml.py -e tests/mock.table.txt --reload -m RF,SVML,SVMP,SVMR tests/mock.table.core_table.txt
'''

import sys, os, re
import pandas as pd
from optparse import OptionParser
from collections import namedtuple
from sklearn import preprocessing

#import numpy as np
#import matplotlib.pyplot as plt

# My functions and classes
from scripts.OptionValidator import ValidateFile, ValidateInteger, ValidateStringParameter, ValidateDataFrameColumns
from scripts.MachineModelController import MachineController

def main():

    '''
        Pre-workflow overhead: Validation of user choices.

        This occurs in two stages;
            1. Import and validation of ESOM table and coverage table
            2. Validation of remaining parameters

        This order was chosen because in the event of a default neural network being requested, the layer sizes is calculated from the ESOM table structure

    '''
    options, core_contig_file = parse_user_input()

    '''
        Original ESOM table is split into the following variables:
            1. ESOM table - A DataFrame for modelling. ESOM coordinates as V*, Coverage values as Coverage*, Bin membership as bin names
            2. Feature list - The column names in the ESOM table that are used for modelling.
            3. Contig list - A list of the contig fragment names, in the order they occur in the ESOM table
            4. Bin membership list - A list of the bins that each contig is found in. Order matches that of outputs 1 and 2.
    '''
    user_table, core_contig_table = import_user_tables(options.esomTable, core_contig_file)
   
    esom_core, esom_cloud = split_esom_for_training(user_table, options.use_bin_membership, core_contig_table)

    options = validate_ml_options(options, esom_core)

    output_file_stub = options.output if options.output else os.path.splitext(options.esomTable)[0]

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
    machineModelController = MachineController(options.models, output_file_stub, options.reload, options.nn_nodes, options.rf_trees, options.threads, options.seed)

    ''' Step 2. '''

    if not options.reload:

        validation_confidence_list = machineModelController.train_models(options.cross_validate, esom_core, options.seed)
        report_classification(validation_confidence_list, 'core_validation', output_file_stub)

    ''' Step 3. '''
    if not options.reload:

        machineModelController.save_models()
        machineModelController.report_training()

        if options.evaluate_only: sys.exit()

    ''' Step 4. '''
    classification_result = machineModelController.classify_by_ensemble(esom_cloud)
    classification_result['ContigBase'] = esom_cloud.contig_base
    classification_result['ContigName'] = esom_cloud.contig_fragments

    report_classification(classification_result, 'contig_classification', output_file_stub)

###############################################################################

# region User input validation

def parse_user_input():

    ''' Parse options '''
    usage_string = "usage: %prog [options] [core contig table]"
    parser = OptionParser(usage=usage_string)

    ''' Basic options '''
    parser.add_option('-e', '--esom-table', help='A table produced by the vizbin_files_to_table.py script', dest='esomTable')
    parser.add_option('-o', '--output', help='An output prefix for all generated files (Default: Inferred from ESOM table)', dest='output', default=None)

    ''' Machine-learning parameters - general '''
    parser.add_option('-m', '--models', help='Comma-separated list of model types to build. Supported options are RF (random forest), NN (neural network), SVML (SVM with linear kernel), SVMR (SVM with radial basis function kernel), SVMP (SVM with polynomial kernel). Classification with be an ensemble of methods (Default: all)', dest='models', default='RF,NN,SVML,SVMR,SVMP')
    parser.add_option('-s', '--seed', help='Random seed for data permutation and model building', dest='seed', default=None)
    parser.add_option('-t', '--threads', help='Number of threads to use for model training and classification (Default: 1)', dest='threads', default=1)
    parser.add_option('-v', '--cross-validate', help='Perform X-fold cross validation in model building (Default: 10)', dest='cross_validate', default=10)
    parser.add_option('--evaluate-only', help='Only run as far as validating the training models, then terminate. Useful for deciding which models to use (Default: False)', dest='evaluate_only', action='store_true', default=False)
    parser.add_option('--reload', help='Load trained models from a previous run instead of training new ones (Default: False)', dest='reload', action='store_true', default=False)
    parser.add_option('--use-bin-membership', help='Encode the original binning information as a feature in prediction (Default: False)', dest='use_bin_membership', action='store_true', default=False)
    
    ''' Machine-learning parameters - classifier-specific '''
    parser.add_option('--rf-trees', help='Number of decision trees for random forest classification (Default: 1000)', dest='rf_trees', default=1000)
    parser.add_option('--nn-nodes', help='Comma-separated list of the number of neurons in the input, hidden, and output layers (Default: Input = number of features + 1, Output = number of classification outcomes, Hidden = mean of Input and Output)', dest='nn_nodes', default=None)

    options, args = parser.parse_args()

    if len(args) == 0:
        print('No core contig table provided. Aborting...')
        sys.exit()

    return options, args[0]

def import_user_tables(esom_table_file, core_contig_file):

    ''' Import and test the ESOM coordinates table '''
    ValidateFile(inFile=esom_table_file, fileTypeWarning='ESOM table', behaviour='abort')
    user_table = pd.read_csv(esom_table_file, sep='\t')

    ValidateDataFrameColumns(df=user_table, columnsRequired=['V1', 'V2', 'BinID', 'ContigName', 'ContigBase'])

    ''' Import and test the core contig table '''
    core_contig_table = pd.read_csv(core_contig_file, sep='\t')
    ValidateDataFrameColumns(df=core_contig_table, columnsRequired=['Bin', 'Contig'])

    return user_table, core_contig_table

def validate_ml_options(_opts, esom_core):

    _opts.models = _parse_model_choices(_opts.models)
    _opts.seed = ValidateInteger(userChoice=_opts.seed, parameterNameWarning='random seed', behaviour='skip')
    _opts.threads = ValidateInteger(userChoice=_opts.threads, parameterNameWarning='threads', behaviour='default', defaultValue=1)
    _opts.cross_validate = ValidateInteger(userChoice=_opts.cross_validate, parameterNameWarning='number of training splits', behaviour='default', defaultValue=10)
    _opts.rf_trees = ValidateInteger(userChoice=_opts.rf_trees, parameterNameWarning='decision trees', behaviour='default', defaultValue=1000)
    _opts.nn_nodes = _parse_layer_choices(_opts.nn_nodes, esom_core)

    return _opts

def _parse_model_choices(modelString):

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

def _parse_layer_choices(neuronString, esom_core):

    '''
        Input:
            1. A user-specified list of neurons to use in the input, hidden, and output layer of neural network
            2. The esom_core namedtuple, containing variables encoding the number of features and classification options

        Action:
            1. Wrapped in flow control, cases are:
                1.1 Verifies that the user values are valid integers, and returns
                1.2 Determines default number of input, hidden, and output layer neurons to use with the rule
                    1.2.1 Input layer = Number of features for classification + 1
                    1.2.2 Hidden layer = Mean of input and output neuron counts
                    1.2.3 Output later = Number of classification options (number of bins in the core_contig_table)

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
    input_layer = esom_core.scaled_features.shape[1]
    output_layer = len( esom_core.original_bin.unique() )
    hidden_layer = (input_layer + output_layer) / 2

    return input_layer, int(hidden_layer), output_layer

# endregion

# region User data processing

def split_esom_for_training(esomTable, binMembershipFlag, core_contig_table):

    '''
        Input:
            1. A DataFrame with the following columns:
               1.1 V* - Coordinates from ESOM
               1.2 BinID - The text name of the bin from which the contig originally derives
               1.3 ContigName - The name of the contig fragment. Takes the form ContigBase|<i>
               1.4 ContigBase - The name of the original contig
            2. Parameter determining whether or not the original bin information should be included in the training table
            3. A DataFrame with the columns Bin, Contig (equivalent to ContigBase in the esomTable)

        Action:
            1. Perform a left join appending the data from core_contig_table onto the esomTable. Non-core contigs are marked with '-'
            2. Parse these results into the DataFrame preTrainingFrame, with the columns:
                1.1 V* - New ~coordinates from ESOM
                1.2 Contig - The value of ContigBase used to group the fragments
                1.3 OriginalBin - The value of BinID
            3. If bin membership is request for training, these are appended through the _append_bin_membership() function
            4. esomTable is split into the eObjs esom_core and esom_cloud
                These capture which contigs used for training/validation (esom_core) and classification (esom_cloud)

        Result:
            1. Two DataFrames are returned to main(), split according to whether they are the core of unbinned contigs

    '''

    join_df = pd.merge(esomTable, core_contig_table, how='left', left_on='ContigBase', right_on='Contig', left_index=False, right_index=False).fillna('-')
    join_df.drop('Contig', axis=1, inplace=True)

    ''' If required, encode bin identity as new factors '''
    if binMembershipFlag:
        join_df = _append_bin_membership(join_df)

    ''' Pop off the text columns and create namedtuples carrying the information needed for ML processing '''
    esom_core = _bind_to_table_obj( join_df[ join_df.Bin != '-' ] )
    esom_cloud = _bind_to_table_obj( join_df[ join_df.Bin == '-' ] )

    return esom_core, esom_cloud

def _append_bin_membership(baseFrame):

    '''
        Input:
            1. A DataFrame with the columns V*, Coverage* (optional) , Contig, and one that identifies bins (expected: OriginalBin)
            3. The name for the column that identifies bins in baseFrame

        Action:
            1. Apply one-hot encoding to the bin column, and merge this DataFrame with baseFrame
            2. Drop the bin identifier column

        Result:
            1. A modified baseFrame with the in identifies encoded as numeric values is returned to the calling function
    '''

    onehotFrame = pd.get_dummies( baseFrame.BinID )
    newFrame = pd.concat( [baseFrame, onehotFrame], axis=1 )
    return newFrame

def _peak_into_obj(o):

    '''
        DEBUG ONLY

        Input:
            1. An eObj tuple with the variables ordValues, contigList, coreBinList, and originalBinList

        Action:
            1. Print the first 5 values of each vector within the eObj

        Result:
            1. No values are returned to calling function

    '''
    print('\n\nordValues'); print(o.scaled_features)
    print('\ncontig_base'); print(o.contig_base[0:5] )
    print('\ncontig_fragments'); print( o.contig_fragments[0:5] )
    print('\noriginal_bin (set)'); print( set(o.original_bin) )

def _bind_to_table_obj(dfSlice):

    '''
        Input:
            1. A DataFrame with the columns V*, ContigBase, ContigName, BinID, Bin, and optionally bin dummies        
        Action:
            1. Splits the DataFrame into a new DataFrame retaining only numeric values needed for modeling
            2. Scale the ESOM coords with unit scaling
            2. Bind the text columns to new variables in the object
        Result:
            1. An eObj is returned to the calling function split_esom_for_training() or ParseEsomForErrorProfiling()
    '''

    eObj = namedtuple('eObj', 'scaled_features contig_base contig_fragments original_bin')

    contig_base = dfSlice.pop('ContigBase')
    contig_fragments = dfSlice.pop('ContigName')
    original_bin = dfSlice.pop('BinID')

    ''' Create a copy of the original dfSlice, mainly just to avoid copy modification warnings '''
    feature_df = dfSlice.drop('Bin', axis=1)

    tsne_coord_columns = [ c for c in feature_df.columns if re.match( r'V\d+$', c) ]
    for tsne_coord_column in tsne_coord_columns:

        feature_df[ tsne_coord_column ] = preprocessing.scale( dfSlice[ tsne_coord_column ] )

    return eObj(scaled_features=feature_df.values, contig_base=contig_base, contig_fragments=contig_fragments, original_bin=original_bin)

# endregion

# region Saving classification confidence summaries

def report_classification(classification_df, data_type, output_file_stub):

    output_path = '{}.{}.txt'.format(output_file_stub, data_type)
    classification_df.to_csv(output_path, sep='\t', index=False)

# endregion

###############################################################################
if __name__ == '__main__':
    main()