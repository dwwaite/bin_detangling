'''
    A series of functions used to validate commonly used option types (files, numeric values, etc)
'''
import sys, os
from collections import namedtuple

# region File validation 

def ValidateFile(inFile, behaviour, fileTypeWarning=None, _callback=None, **kwargs):

    if os.path.isfile(inFile):

        return inFile

    else:

        if behaviour == 'callback':

            return _callback(kwargs)

        elif behaviour == 'skip':

            print( 'Warning: Unable to detect {} {}, skipping...'.format(fileTypeWarning, inFile) )
            return None

        elif behaviour == 'abort':

            print( 'Warning: Unable to detect {} {}, aborting...'.format(fileTypeWarning, inFile) )
            return None

def ValidateFolder(inFile, behaviour, fileTypeWarning=None, _callback=None, **kwargs):

    if os.path.isdir(inFile):

        return inFile

    else:

        if behaviour == 'callback':

            return _callback(kwargs)

        elif behaviour == 'skip':

            print( 'Warning: Unable to detect {} {}, skipping....'.format(fileTypeWarning, inFile) )
            return None

        elif behaviour == 'abort':

            print( 'Warning: Unable to detect {} {}, aborting....'.format(fileTypeWarning, inFile) )
            sys.exit()

def ValidateDataFrameColumns(df, columnsRequired):

    try:

        for reqColumn in columnsRequired:

            assert( reqColumn in df.columns ), reqColumn

        return True

    except AssertionError as ae:

        print( '\nUnable to find required column {}, aborting...'.format(ae) )
        return False

# endregion 

def ValidateStringParameter(userChoice, choiceTypeWarning, allowedOptions, behaviour, defBehaviour=None):

    if userChoice in set(allowedOptions):

        return userChoice

    else:

        if behaviour == 'skip':

            print( 'Warning: Unable to parse {} {}, skipping...'.format(choiceTypeWarning, userChoice) )
            return None

        if behaviour == 'abort':

            print( 'Warning: Unable to parse {} {}, aborting...'.format(choiceTypeWarning, userChoice, defBehaviour) )
            sys.exit()
            
        if behaviour == 'default':

            print( 'Warning: Unable to parse {} {}, using {} instead.'.format(choiceTypeWarning, userChoice, defBehaviour) )
            return defBehaviour

# region Numeric validation

def ValidateInteger(userChoice, parameterNameWarning, behaviour, defaultValue=None):

    try:

        i = int(userChoice)
        return i

    except:

        if behaviour == 'default':

            print( 'Unable to accept value {} for {}, using default ({}) instead.'.format(userChoice, parameterNameWarning, defaultValue) )
            return defaultValue

        if behaviour == 'abort':

            print( 'Unable to accept value {} for {}, aborting...'.format(userChoice, parameterNameWarning, defaultValue) )
            sys.exit()

        if behaviour == 'skip':

            return None

def ValidateFloat(userChoice, parameterNameWarning, behaviour, defaultValue=None):

    try:

        f = float(userChoice)
        return f

    except:

        if behaviour == 'default':

            print( 'Unable to accept value {} for {}, using default ({}) instead.'.format(userChoice, parameterNameWarning, defaultValue) )
            return defaultValue

        if behaviour == 'abort':

            print( 'Unable to accept value {} for {}, aborting...'.format(userChoice, parameterNameWarning, defaultValue) )
            return None

# endregion