'''
    Functions that I'm constantly having to re-write between scripts
'''
import sys, os

def IndexFastaFile(fileName):

    index = {}

    content = open(fileName, 'r').read().split('>')[1:]
    for entry in content:

        seqName, *seqContent = entry.split('\n')
        index[seqName] = ''.join(seqContent)

    return index