'''
    A simple script to chomp up contigs into smaller pieces for use in ESOM clustering.

    First argument is the contig fragment size to chop to, subsequent arguments are bin files.
'''
import sys, os

def main():

    cSize = sys.argv[1]
    try:
        cSize = int(cSize)
    except:
        print( 'Unable to convert contig size ({}) to an integer. Aborting...'.format(cSize) )
        sys.exit()

    # Can change if needed.
    fOutput = 'vizbin.fna'
    aOutput = 'vizbin.ann'

    binFiles = sys.argv[2:]
    ProcessVizBinInputs(binFiles, fOutput, aOutput, cSize)

###############################################################################
def FastaAsTuples(_binName):
    content = open(_binName, 'r').read().split('>')[1:]
    ret = []
    for c in content:
        head, *seq = c.split('\n')
        seq = ''.join( list(seq) )
        ret.append( (head, seq) )
    return ret

def ChompSeq(sequence, size):
    chopStart = 0
    while(chopStart + size < len(sequence) - chopStart):
        yield sequence[chopStart:(chopStart + size)]
        chopStart += size
    yield sequence[chopStart:]

def ChompifySequences(chompSize, binName):
    for seqName, seq in FastaAsTuples(binName):
        if len(seq) > chompSize:
            for i, seqSlice in enumerate( ChompSeq(seq, chompSize) ):
                chompName = '{}|{}'.format(seqName, i)
                yield (chompName, seqSlice)
        else:
            yield '{}|{}'.format(seqName, 0), seq

def ProcessVizBinInputs(_binFiles, _fastaStream, _annotStream, _chompFactor):

    _fastaStream = open(_fastaStream, 'w')
    _annotStream = open(_annotStream, 'w')
    _annotStream.write('label\n')

    for b in _binFiles:
        for chompName, seq in ChompifySequences(_chompFactor, b):
            _fastaStream.write( '>{}\n{}\n'.format(chompName, seq) )
            _annotStream.write( '{}\n'.format(b) )

    _fastaStream.close()
    _annotStream.close()
###############################################################################
if __name__ == '__main__':
     main()
