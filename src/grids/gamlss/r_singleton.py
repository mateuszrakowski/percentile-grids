# r_singleton.py
import warnings


class REnvironment:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(REnvironment, cls).__new__(cls)

            # Suppress warnings during initialization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Initialize R
                import rpy2.robjects as ro
                from rpy2.robjects import pandas2ri
                from rpy2.robjects.packages import importr

                # Activate automatic conversion
                pandas2ri.activate()

                # Store references to commonly used objects
                cls._instance.ro = ro
                cls._instance.base = importr("base")
                cls._instance.stats = importr("stats")
                cls._instance.gr_devices = importr("grDevices")
                cls._instance.gamlss_r = importr("gamlss")
                cls._instance.gamlss_dist = importr("gamlss.dist")

        return cls._instance
