'''
    A simple class to abstract multithread managing of various scripts within this repo.

'''
import sys, time
from multiprocessing import Pool, Queue, Manager

class ThreadManager:

    def __init__(self, nThreads, func):

        self._manager = Manager()
        self.queue = self._manager.Queue()
        self._pool =  Pool(nThreads)
        self._func = func
        
        ''' An unused pointer for now, will be used to catch the map_async call when it happens '''
        self._tracker = None

    def ActivateMonitorPool(self, sleepTime, funcArgs, trackingString = None, totalJobSize = None):

        '''
            For using trackingString:
                The string must include two {} placeholders, one for the currently completed jobs, and one for the total number of jobs.
                Will print to stderr, and use carriage returns without newline, so just updating itself in place.
        '''

        try:
            sleepTime = int(sleepTime)
        except:
            sleepTime = 60

        self._tracker = self._pool.map_async(self._func, funcArgs)

        while not self._tracker.ready():

            if trackingString and totalJobSize:
                pStr = trackingString.format(self.queue.qsize(), totalJobSize)
                print(pStr, end='\r', file=sys.stderr, flush=True)

            time.sleep(sleepTime)

        if trackingString and totalJobSize:
            pStr = trackingString.format(totalJobSize, totalJobSize)
            print(pStr, file=sys.stderr, flush=True)

    @property
    def results(self):
        return [ x for x in self._extractQueueResults() ]

    def _extractQueueResults(self):
        while not self.queue.empty():
            yield self.queue.get(True)