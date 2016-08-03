"""
Module with methods to return timing information.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import logging
import resource
import time

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_LOG = logging.getLogger("timer")

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

class Timer(object):
    """
    Instance of this class can be used to track consumed time.

    This class is also a context manager and can be used in
    a `with` statement. By default it prints consumed CPU time
    and real time spent in a context.

    Example:

        with Timer('SelectTimer'):
            engine.execute('SELECT ...')

    """

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, name="", doPrint=True):
        """
        @param name:  Time name, will be printed together with statistics
        @param doPrint: if True then print statistics on exist from context
        """
        self._name = name
        self._print = doPrint

        self._startReal = None
        self._startUser = None
        self._startSys = None
        self._sumReal = 0.
        self._sumUser = 0.
        self._sumSys = 0.

    #-------------------
    #  Public methods --
    #-------------------

    def start(self):
        """
        Start timer.
        """
        self._startReal = time.time()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self._startUser = ru.ru_utime
        self._startSys = ru.ru_stime
        return self

    def stop(self):
        """
        Stop timer.
        """
        if self._startReal is not None:
            self._sumReal += time.time() - self._startReal
            ru = resource.getrusage(resource.RUSAGE_SELF)
            self._sumUser += ru.ru_utime - self._startUser
            self._sumSys += ru.ru_stime - self._startSys
            self._startReal = None
            self._startUser = None
            self._startSys = None
        return self

    def dump(self):
        """
        Dump timer statistics
        """
        _LOG.info("%s", self)
        return self

    def __str__(self):
        real = self._sumReal
        user = self._sumUser
        sys = self._sumSys
        if self._startReal is not None:
            real += time.time() - self._startReal
            ru = resource.getrusage(resource.RUSAGE_SELF)
            user += ru.ru_utime - self._startUser
            sys += ru.ru_stime - self._startSys
        info = "real=%.3f user=%.3f sys=%.3f" % (real, user, sys)
        if self._name:
            info = self._name + ": " + info
        return info

    def __enter__(self):
        """
        Enter context, start timer
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context, stop and dump timer
        """
        if exc_type is None:
            self.stop()
            if self._print: self.dump()
        return False
