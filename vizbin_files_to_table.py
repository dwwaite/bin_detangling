'''
    Just a quick QoL script to standardise the input format for downstream scripts.
'''
import sys, os
import pandas as pd
from optparse import OptionParser

from scripts.OptionValidator import ValidateFile, ValidateInteger

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

    if not options.output:
        oName = os.path.splitext(options.fasta)[0] + '.csv'
        print('Warning: No output file name provided, using {}....'.format(oName))
        options.output = oName

    # Parse the filter_length, if it exists.
    options.filter_length = ValidateInteger(userChoice=options.filter_length, parameterNameWarning='filter length', behaviour='default', defaultValue=1000)

    ''' Proceed... '''
    table = ImportVizBinData(options.points, options.fasta, options.annotation, options.filter_length)
    table.to_csv(options.output, index=False, sep='\t')
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