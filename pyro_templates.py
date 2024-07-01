"""
@w-garcia
"""
import Pyro4

from Pyro4.errors import NamingError
from custom_logging import get_logger


class PyroPublisher:
    def __init__(self, *args, **kwargs):
        """
        Set up a Pyro daemon to expose the initiated authenticator object.
        The authenticator object gets called by the client directly.
        This class normally doesn't have any other methods other than __init__,
        since the client doesn't interact with it.
        :param args:
        :param kwargs:
        """
        self.logger = get_logger(type(self).__name__)
        self.logger.info("HMAC key: ")
        hk = input("> ")
        ns = Pyro4.locateNS(hmac_key=hk)
        daemon = Pyro4.Daemon(host=kwargs["hostname"])
        daemon._pyroHmacKey = hk

        authenticator = kwargs["authenticator"]
        uri = daemon.register(authenticator)
        ns.register(kwargs["objname"], uri)
        self.logger.info("Daemon ready.")
        daemon.requestLoop()


class PyroClient:
    """
    Client object to interact with Pyro published objects. Mainly here to standardize error checking.
    query expects numpy or list-type (not string)
    """
    def __init__(self, ORACLENAME, hk=None, ns=None):
        """
        Set up a Pyro client around a given ORACLENAME lookup tag
        :param ORACLENAME: A lookup tag to find the associated object in Pyro nameserver
        :param hk: Key for authentication
        :param ns: IP of nameserver, if known.
        """
        self.logger = get_logger(__name__)
        if ns is None:
            self.logger.info("Server IP:")
            ns = input("> ")
        if hk is None:
            self.logger.info("HMAC key:")
            hk = input("> ")

        self.ORACLENAME = ORACLENAME
        self.logger = get_logger(type(self).__name__)
        self.logger.debug("Attempting namespace search on ns={}".format(ns))
        # TODO: Some issue with sacred and hmac becoming unicode...
        self.ns = Pyro4.locateNS(ns, hmac_key=hk, broadcast=False)
        try:
            uri = self.ns.lookup(ORACLENAME)
        except NamingError as e:
            self.logger.fatal("Caught exception:")
            self.logger.exception(e)
            raise NamingError("We failed to find the published object in the nameserver. Is publisher running?")

        self.authenticator = Pyro4.Proxy(uri)
        self.authenticator._pyroHmacKey = hk

    def query(self, *args, **kwargs):
        """
        Error checking wrapper around our call to the published object.
        :param args:
        :param kwargs:
        :return:
        """
        try:
            if len(args) == 1:
                # Attempted batch mode. Squeeze it
                args = args[0]
            if type(args[1]) is not list and len(args[1].shape) >= 2:
                # Given many signatures with 1 credential
                res = []
                for s in args[1]:
                    args_c = [args[0], s]
                    res.append(self._query(*args_c, **kwargs))
                return res

            return self._query(*args, **kwargs)
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("The query system crashed. Restart <r> or quit <q>:")
            inp = input("> ")
            if inp == 'q' or inp == 'Q':
                return ""
            else:
                self.logger.info("Enter HMAC key: ")
                hk = input("> ")
                self.logger.info("Enter Server IP: ")
                ns = input("> ")
                self.ns = Pyro4.locateNS(ns, hmac_key=hk, broadcast=False)
                uri = self.ns.lookup(self.ORACLENAME)
                self.authenticator = Pyro4.Proxy(uri)
                self.authenticator._pyroHmacKey = hk
                return self.query(*args, **kwargs)

    def _query(self, *args, **kwargs):
        """
        Overload according to specific authentication domain.
        Returns call to self.oracle.
        :param args:
        :param kwargs:
        :return:
        """
        pass
