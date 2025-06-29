import warnings
from unittest.mock import MagicMock

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class REnvironment:
    """
    A Singleton class to manage the R environment connection.

    It ensures that R packages are imported and the environment is
    initialized only once during the application's lifecycle.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(REnvironment, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Store references to commonly used R objects and packages
                self.base = rpackages.importr("base")
                self.stats = rpackages.importr("stats")
                self.grDevices = rpackages.importr("grDevices")
                self.gamlss_r = rpackages.importr("gamlss")
                self.gamlss_dist = rpackages.importr("gamlss.dist")

                # Store converter tools for safe, localized conversions
                self.pandas2ri = pandas2ri
                self.localconverter = localconverter
                self.robjects = robjects

                # Activate the pandas to R converter
                self.pandas2ri.activate()

                print("Successfully initialized R environment Singleton.")
                self._initialized = True
            except Exception as e:
                print(f"Warning: Failed to import R's gamlss package: {e}")
                self._setup_mock_objects()

    def _setup_mock_objects(self):
        """Sets up mock objects if R or gamlss is not available."""
        self.base = self.stats = self.grDevices = self.gamlss_r = self.gamlss_dist = (
            MagicMock()
        )
        self.pandas2ri = MagicMock()
        self.localconverter = MagicMock()
        self.robjects = MagicMock()
        self._initialized = True  # Mark as initialized to avoid retries
