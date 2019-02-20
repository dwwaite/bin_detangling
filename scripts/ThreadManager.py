'''
    Just a few helper methods for ease of use
'''
import sys, time
from multiprocessing import Pool, Queue, Manager

def SpawnPoolMananger(nThreads):
    return Manager().Queue(), Pool(nThreads)

def DrainQueue(q):
    while not q.empty():
        yield q.get_nowait()

def MonitorActivePool(nJobs, poolManager, poolTracker):
    while not poolTracker.ready():
        pStr = 'Completed {} of {} jobs.'.format(poolManager.qsize(), nJobs)
        print(pStr, end='\r', file=sys.stderr, flush=True)
        time.sleep(60) # Print once per minute

    print('Completed {} of {} jobs.'.format(poolManager.qsize(), nJobs),  file=sys.stderr)