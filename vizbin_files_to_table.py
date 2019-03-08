'''
    Just a quick QoL script to standardise the input format for downstream scripts.
'''
import sys, os
import pandas as pd
from optparse import OptionParser

def main():

    parser = OptionParser()
    usage = "usage: %prog [options]"

    parser.add_option('-p', '--points', help='A VizBin exported points file.', dest='points')
    parser.add_option('-a', '--annotation', help='A VizBin exported annotation file. If left empty (e.g. unbinned data), will populate with dummy values', dest='annotation', default='-')
    parser.add_option('-f', '--fasta', help='The fasta file imported into VizBin.', dest='fasta')
    parser.add_option('--filter_length', help='The filtering length passed to VizBin during ESOM (default = 1000).', dest='filter_length', default=1000)
    parser.add_option('-o', '--output', help='Output table name. Default: [fasta file].csv', dest='output', default='-')

    options, args = parser.parse_args()
    VerifyOption(options.points, 'points')
    VerifyOption(options.fasta, 'fasta')

    # Import the annotation data, if it exists
    options.annotation = ImportAnnotationData(options.annotation)

    # Parse the filter_length, if it exists.
    options.filter_length = ParseFilterLength(options.filter_length)

    if options.output == '-':
        options.output = os.path.splitext(options.fasta)[0] + '.csv'

    ''' Proceed... '''
    table = ImportVizBinData(options.points, options.annotation, options.fasta, options.filter_length)
    table.to_csv(options.output, index=False, sep='\t')
###############################################################################
def VerifyOption(opt, fType):
    if not opt:
        print('Unable to proceed without specifying a %s file. Aborting...' % fType)
        sys.exit()
    if not os.path.isfile(opt):
        print('Unable to open %s. Aborting...' % opt)
        sys.exit()

def ImportAnnotationData(annFile):
        if annFile == '-':
            return 'unbinned'
        else:
            try:
                ann = pd.read_csv(annFile)
                ann = ann['label']
                return ann
            except:
                print( 'Unable to parse annotation file {}, using dummy values instead.'.format(options.annotation) )
                return 'unbinned'

def ParseFilterLength(fLength):
    try:
        fLength = int(fLength)
        return fLength
    except:
        print( 'Unable to cast filter length {} as an integer. Aborting...'.format(fLength) )
        sys.exit()
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

def ImportVizBinData(pointsFile, annotation, fastaFile, filtLength):
    points = pd.read_csv(pointsFile, header=None)
    points.columns = ['V1', 'V2']
    points['BinID'] = annotation

    seqNames = ImportAndFilterFasta(fastaFile, filtLength)
    points['ContigName'] = seqNames
    points['ContigBase'] = [ s.split('|')[0] if '|' in s else s for s in seqNames ]

    return points
###############################################################################
if __name__ == '__main__':
    main()