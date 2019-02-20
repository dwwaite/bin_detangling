'''
    Functions that I'm constantly having to re-write between scripts
'''
import sys, os

def IndexFastaFile(fileName):

    content = open(fileName, 'r').read().split('>')[1:]

    index = {}

    for entry in content:

        entry = entry.split('\n')
        contigName = entry[0]
        sequence = ''.join(entry[1:])
        index[contigName] = sequence

    return index