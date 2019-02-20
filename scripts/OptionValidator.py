'''
    A series of functions used to validate commonly used option types (files, numeric values, etc)
'''
import sys, os
from collections import namedtuple

def ValidateFile(inFile, fileTypeWarning, behaviour):

    if os.path.isfile(inFile):

        return inFile

    else:

        if behaviour == 'skip':

            print( 'Warning: Unable to detect {} {}, skipping....'.format(fileTypeWarning, covFile) )
            return None

        if behaviour == 'abort':

            print( 'Warning: Unable to detect {} {}, aborting....'.format(fileTypeWarning, covFile) )
            sys.exit()

def ValidateStringParameter(userChoice, choiceTypeWarning, allowedOptions, behaviour, defBehaviour=None):

    if userChoice.lower() in set(allowedOptions):

        return userChoice.lower()

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