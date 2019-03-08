'''
    Just a quick QoL script to standardise the input format for downstream scripts.
'''
import sys, os
import pandas as pd
from optparse import OptionParser

from OptionValidator import ValidateFile, ValidateInteger

def main():

    parser = OptionParser()
    usage = "usage: %prog [options]"

    parser.add_option('-p', '--points', help='A VizBin exported points file.', dest='points')
    parser.add_option('-a', '--annotation', help='A VizBin exported annotation file. If left empty (e.g. unbinned data), will populate with dummy values', dest='annotation', default='-')
    parser.add_option('-f', '--fasta', help='The fasta file imported into VizBin.', dest='fasta')
    parser.add_option('--filter_length', help='The filtering length passed to VizBin during ESOM (default = 1000).', dest='filter_length', default=1000)
    parser.add_option('-o', '--output', help='Output table name. Default: [fasta file].csv', dest='output', default=None)

    # Validate user inputs
    options, args = parser.parse_args()

    options.points = ValidateFile(inFile=options.points, behaviour='abort', fileTypeWarning='points file')
    options.fasta = ValidateFile(inFile=options.fasta, behaviour='abort', fileTypeWarning='fasta file')
    options.annotation = ValidateFile(inFile=options.annotation, behaviour='skip', fileTypeWarning='annotation file')
    options.output = ValidateFile(inFile=options.output, behaviour='callback', _callback=_outputCallback, fastaFile=options.fasta, outputFile=options.output)

    # Parse the filter_length, if it exists.
    options.filter_length = ValidateInteger(userChoice=options.filter_length, parameterNameWarning='filter length', behaviour='default', defaultValue=1000)

    ''' Proceed... '''
    table = ImportVizBinData(options.points, options.fasta, options.annotation, options.filter_length)
    table.to_csv(options.output, index=False, sep='\t')
###############################################################################
# This a a silly over-complication of setting a default file, but I don't want defaults to be
# a standard part of the file validation library.
def _outputCallback(kwargs):

    if os.path.isfile( kwargs['outputFile'] ):
        return kwargs['outputFile']
    else:
        oName = os.path.splitext(kwargs['fastaFile'])[0] + '.csv'
        print( 'Warning: No output file name provided, using {}....'.format(oName) )
        return oName

###############################################################################

def ImportAndFilterFasta(fastaFile, filtLength):
    fastaNames = []
    fastaContent = open(fastaFile, 'r').read().split('>')[1:]
    for fC in fastaContent:
        seqName, *seqBases = fC.split('\n')
        seqBases = ''.join( seqBases )
        if len(seqBases) >= filtLength:
            fastaNames.append(seqName)
    return fastaNames

def ImportVizBinData(pointsFile, fastaFile, annotationFile, filtLength):
    points = pd.read_csv(pointsFile, header=None)
    points.columns = ['V1', 'V2']
    points['BinID'] = _importAnnotationData(annotationFile)

    seqNames = ImportAndFilterFasta(fastaFile, filtLength)
    points['ContigName'] = seqNames
    points['ContigBase'] = [ s.split('|')[0] if '|' in s else s for s in seqNames ]

    return points

def _importAnnotationData(annFile):
        if annFile:
            ann = pd.read_csv(annFile)
            return ann['label']
        else:
            return 'unbinned'
###############################################################################
if __name__ == '__main__':
    main()