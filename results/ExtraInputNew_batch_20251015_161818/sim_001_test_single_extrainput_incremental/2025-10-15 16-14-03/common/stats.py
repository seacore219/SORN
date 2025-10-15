from __future__ import division
from pylab import *
from utils import DataLog
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    imported_mpi = True
except ImportError:
    imported_mpi = False

import utils
utils.backup(__file__)


class AbstractStat(object):
    '''A Stat is meant to encapsulate a single bit of data stored in a floating
    numpy array. All data that might need to be stored gets put into a
    utils.Bunch object and passed between each stat.  This allows the
    calculation of more complicated stats which rely on several bits of
    data.'''
    def __init__(self):
        self.name = None #<---Must be set to a valid variable name
        self.collection = "reduce"
        self.parent = None

    def connect(self,parent):
        self.parent = parent

    def start(self,c,obj):
        '''The intention of start is to be first call after object is created
        Also this gets called once during lifetime of stats object'''
        pass

    def clear(self, c, obj):
        '''Called whenever statistics should be reset'''
        pass

    def add(self, c, obj):
        '''Called after each simulation step'''
        pass

    def report(self, c, obj):
        '''To ensure things work on the cluster, please always return a
        Numpy *double* array!
        There is some flexibility in what you return, either return a
        single array (and the name and collection method will be inferred
        from self.name and self.collection respectively).  Or return a
        list of tuples and give name, collection and data explicitly.'''
        #Called after a block of training has occurred
        raise NotImplementedError("Bad user!")

class CounterStat(AbstractStat):
    '''A simple stat that counts the number of add() calls.'''
    def __init__(self,name='num_steps'):
        self.name = name
        self.collection = "reduce"

    def start(self,c,obj):
        c[self.name] = 0.0 #Everything needs to be a float :-/

    def add(self,c,obj):
        c[self.name] += 1.0

    def report(self,c,obj):
        return array(c[self.name]) #And an array :-/

def _getvar(obj,var):
    if '.' in var:
        (obj_name,_,var) = var.partition('.')
        obj = obj.__getattribute__(obj_name)
        return _getvar(obj,var)
    return obj.__getattribute__(var)

class HistoryStat(AbstractStat):
    '''A stat that monitors the value of a variable at every step of
    the simulation'''
    def __init__(self, var='x',collection="gather",record_every_nth=1):
        self.var = var
        self.name = var+"_history"
        self.counter = var+"_counter"
        self.collection = collection
        self.record_every_nth = record_every_nth

    def start(self,c,obj):
        if 'history' not in c:
            c.history = utils.Bunch()
        c.history[self.counter] = 0

    def clear(self, c, obj):
        c.history[self.name] = []

    def add(self,c,obj):
        if not (c.history[self.counter] % self.record_every_nth):
            tmp = _getvar(obj,self.var)
            if callable(tmp):
                tmp=tmp()
            c.history[self.name].append(np.copy(tmp))
        c.history[self.counter] += 1

    def report(self,c,obj):
        try:
            return array(c.history[self.name])
        except ValueError as v:
            print 'Error in stats.py', v, self.name
            #~ import pdb
            #~ pdb.set_trace()


class StatsCollection:
    def __init__(self,obj,dlog=None):
        '''The StatsCollection object holds many statistics objects and
        distributes the calls to them.  It also simplifies the collection
        of information when report() and cluster_report() are called.'''
        self.obj = obj
        self.c = utils.Bunch()
        self.disable = False
        self.methods = []
        if dlog is None:
            self.dlog = DataLog()
        else:
            self.dlog = dlog

    def start(self):
        '''The start() method is called once per simulation'''
        for m in self.methods:
            m.connect(self)
            m.start(self.c,self.obj)
            m.clear(self.c,self.obj)

    def clear(self):
        '''The clear() method is called at the start of an epoch that will
        be monitored'''
        for m in self.methods:
            m.clear(self.c,self.obj)

    def add(self):
        if self.disable:
            return
        for m in self.methods:
            m.add(self.c,self.obj)

    def _report(self):
        '''report() is called at the end of an epoch and returns a list
        of results in the form:
         [(name,collection,value)]
        where:
          name = name of statistic
          collection = how to communicate statistic when on a cluster
          value = the value observed.
        '''
        l = []
        for m in self.methods:
            val  = m.report(self.c,self.obj)
            if isinstance(val, list):
                import ipdb; ipdb.set_trace()
                for (name,collection,v) in val:
                    if v.size == 0:
                        continue
                    l.append((name,collection,v))
            else:
                if val.size==0:
                    continue
                l.append((m.name,m.collection,val))
        return l

    def single_report(self):
        l = self._report()
        for (name,coll, val) in l:
            self.dlog.append(name,val)

    def cluster_report(self,cluster):
        '''Same intent as single_report(), but communicate data across the
        cluster.  The cluster variable that is passed in needs to have
        the following attribute:
        cluster.NUMBER_OF_CORES
        '''
        rank = comm.rank
        #Same logic from report()
        l = self._report()
        #Now we do cluster communication
        for (name,coll, val) in l:
            if coll is "reduce":
                # Change because gather was changed -> consistency
                # This way, stuff is only collected on one node
                #~ temp = empty_like(val)
                #~ comm.Allreduce([val, MPI.DOUBLE],[temp, MPI.DOUBLE])
                temp = comm.reduce(val)
                temp = temp/cluster.NUMBER_OF_CORES

            if coll is "gather":
                # Why is there an allgatherv? Doesn't make sense...
                #~ temp = empty( (cluster.NUMBER_OF_CORES,) + val.shape )
                #~ gathers = [prod(val.shape)] * cluster.NUMBER_OF_CORES
                #~ comm.Allgatherv(sendbuf=[val, MPI.DOUBLE],
                                #~ recvbuf=[temp, (gathers, None), MPI.DOUBLE])
                # gather leads to problems with large arrays (Gather instead)
                #~ temp = comm.gather(val, root=0)
                
                if rank == 0:
                    temp = empty((comm.size,)+(prod(val.shape),))
                else:
                    temp = None
                # for debugging
                #~ print "in stats"
                #~ print name
                #~ print shape(val)
                #~ print shape(val.flatten())
                comm.Gather(val.flatten(), temp, root=0)
                if rank == 0:
                    temp = [temp[i].reshape(val.shape) for i in range(comm.size)]
            if coll is "gatherv": #Variable gather size
                # TODO: In principle, use gather here again...
                arrsizes = empty( cluster.NUMBER_OF_CORES, dtype=int )
                arrsize  = array( prod(val.shape) )
                comm.Allgather(sendbuf=[arrsize, MPI.LONG],
                               recvbuf=[arrsizes,MPI.LONG])
                if comm.rank==0:
                    temp = zeros(sum(arrsizes))
                else:
                    temp = zeros(0)
                comm.Gatherv([val.flatten(), (arrsize, None), MPI.DOUBLE],
                             [temp,          (arrsizes,None), MPI.DOUBLE],
                             root=0)

            if coll is "root":
                if rank == 0:
                    temp = val
                else:
                    temp = array([])
            
            self.dlog.append(name,temp)
            del temp #Delete temp to ensure that we fail if coll is unknown


class IncrementalStatsCollection(StatsCollection):
    """Modified StatsCollection that handles chunk rotation"""
    
    def __init__(self, obj, dlog=None, rotate_interval=10000):
        super(IncrementalStatsCollection, self).__init__(obj, dlog)
        self.rotate_interval = rotate_interval
        self.step_count = 0
        self.chunk_id = 0
        
    def add(self):
        """Modified to track steps and rotate files"""
        if self.disable:
            return
            
        # Call original add
        super(IncrementalStatsCollection, self).add()
        
        self.step_count += 1
        
        # Rotate files at interval
        if self.step_count % self.rotate_interval == 0:
            self.rotate_chunks()
    
    def rotate_chunks(self):
        """Rotate H5 files to prevent them from getting too large"""
        print "[Rotate] Rotating chunks at step %d (chunk %d)" % (self.step_count, self.chunk_id)
        
        # Flush and close all H5 files in stats
        for m in self.methods:
            # Force flush if method has buffer
            if hasattr(m, '_flush_buffer'):
                m._flush_buffer()
            
            # Close and rename H5 file
            if hasattr(m, 'h5_file') and m.h5_file:
                m.h5_file.close()
                
                # Rename with chunk ID
                old_path = m.h5_file.filename
                new_path = old_path.replace('_temp.h5', '_chunk%03d.h5' % self.chunk_id)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print "  Saved: %s" % os.path.basename(new_path)
                
                # Restart the file for next chunk
                if hasattr(m, 'start'):
                    m.start(self.c, self.obj)
        
        self.chunk_id += 1
        
        # Delete old chunks to save space (keep only last 2)
        if self.chunk_id > 2:
            self.cleanup_old_chunks()
    
    def cleanup_old_chunks(self):
        """Remove old chunk files to save disk space"""
        old_chunk_id = self.chunk_id - 3
        
        # Find and delete old chunk files
        if hasattr(self.obj, 'c') and hasattr(self.obj.c, 'logfilepath'):
            logpath = self.obj.c.logfilepath
            for filename in os.listdir(logpath):
                if '_chunk%03d.h5' % old_chunk_id in filename:
                    filepath = os.path.join(logpath, filename)
                    os.remove(filepath)
                    print "  Cleaned up: %s" % filename