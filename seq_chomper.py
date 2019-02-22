'''
    A simple script to chomp up contigs into smaller pieces for use in ESOM clustering. Where multiple fasta files are placed together, they are pooled into a single file.
'''
import sys, os
from optparse import OptionParser

# My scripts
from scripts.OptionValidator import ValidateFile, ValidateStringParameter, ValidateInteger
from scripts.SequenceManipulation import IndexFastaFile

def main():


    ''' Set up the options '''
    parser = OptionParser()
    parser.add_option('-c', '--contig-size', help='Size to chop contigs', dest='contigSize', default=1500)

    options, inputFiles = parser.parse_args()

    ''' Validate inputs '''
    options.contigSize = ValidateInteger(userChoice=options.contigSize, parameterNameWarning='contig size', behaviour='default', defaultValue=1500)

    ''' Set output file names '''
    fastaOutputName, fastaWriter = OpenFastaStream(inputFiles[0], options.contigSize)
    binRecordWriter = OpenBinRecordWriter(inputFiles, fastaOutputName)

    for inputFile in inputFiles:

        if not ValidateFile(inFile=inputFile, fileTypeWarning='fasta file', behaviour='skip'):
            continue

        ProcessFastaFile(inputFile, fastaWriter, options.contigSize, binRecordWriter)

###############################################################################

# region File name handlers

def OpenFastaStream(infileName, contigSize):
    infBase, infExt = os.path.splitext(infileName)
    fastaName = '{}.chomp{}{}'.format(infBase, contigSize, infExt)
    return fastaName, open(fastaName, 'w')

def OpenBinRecordWriter(inputFiles, fastaName):

    if len(inputFiles) <= 1:
        return None

    else:

        fExt = os.path.splitext(fastaName)[1]
        return open(fastaName.replace(fExt, '.txt'), 'w')

# endregion

# region Sequence cutting

def MakeSequenceCuts(sequence, size):
    chopStart = 0
    while(chopStart + size < len(sequence) - chopStart):
        yield sequence[chopStart:(chopStart + size)]
        chopStart += size
    yield sequence[chopStart:]

def ChompSequences(chompSize, binName):

    fastaContent = IndexFastaFile(binName)

    for seqName, seq in fastaContent.items():

        ''' If the sequence is longer than the cutting size, slice it up.
            If not, just return it with a modified name. '''
        if len(seq) > chompSize:

            seqBuffer = ''
            chompNameBuffer = ''
            for i, seqSlice in enumerate( MakeSequenceCuts(seq, chompSize) ):

                chompNameBuffer = '{}|{}'.format(seqName, i)

                if len(seqBuffer) > chompSize:

                    ''' If the fragment is greater than or equal to the limit, release the previous sequence '''
                    yield (chompNameBuffer, seqBuffer)
                    seqBuffer = seqSlice

                else:

                    ''' If the sequence fragment is not big enough, append it to the last fragment and release it
                        This only happens on the last piece of a sequence, so no need to set the buffers again '''

                    yield (chompNameBuffer, seqBuffer + seqSlice)
                    
        else:
            yield '{}|{}'.format(seqName, 0), seq

# endregion

def ProcessFastaFile(binFile, fastaStream, contigSize, binStream = None):

    for seqName, seq in ChompSequences(contigSize, binFile):

        fastaStream.write( '>{}\n{}\n'.format(seqName, seq) )

        if binStream:
            binStream.write( '{}\t{}\n'.format(seqName, binFile) )

###############################################################################
if __name__ == '__main__':
     main()
