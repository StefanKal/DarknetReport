"""Base class for under-sampling."""
from __future__ import print_function
from __future__ import division

from abc import ABCMeta, abstractmethod

from ..base_sampler import BaseSampler


class UnderSampler(BaseSampler):
    """Base class for under-sampling.

    Warning: This class should not be used directly. Use the derive classes
    instead.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, ratio='auto', return_indices=False, random_state=None,
                 verbose=True):
        """Initialize this object and its instance variables.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

        return_indices : bool, optional (default=False)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        Returns
        -------
        None

        """

        super(UnderSampler, self).__init__(ratio=ratio,
                                           random_state=random_state,
                                           verbose=verbose)

        self.return_indices = return_indices

    @abstractmethod
    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        super(UnderSampler, self).fit(X, y)

    @abstractmethod
    def transform(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """
        super(UnderSampler, self).transform(X, y)
