"""Class to perform under-sampling based on the instance hardness
threshold."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.utils import check_X_y
from sklearn.cross_validation import StratifiedKFold

from .under_sampler import UnderSampler


class InstanceHardnessThreshold(UnderSampler):
    """Class to perform under-sampling based on the instance hardness
    threshold.

    Parameters
    ----------
    estimator : sklearn classifier
        Classifier to be used in to estimate instance hardness of the samples.

    ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

    cv : int, optional (default=5)
        Number of folds to be used when estimating samples' instance hardness.

    return_indices : bool, optional (default=False)
        Either to return or not the indices which will be selected from
        the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing

    n_jobs : int, optional (default=-1)
        The number of thread to open when it is possible.

    Attributes
    ----------
    ratio_ : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balanced
        the dataset. Otherwise, the ratio will corresponds to the number
        of samples in the minority class over the the number of samples
        in the majority class.

    rs_ : int or None, optional (default=None)
        Seed for random number generation.

    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    estimator_ : sklearn classifier
        Classifier  used in to estimate instance hardness of the samples.

    cv : int, optional (default=5)
        Number of folds used when estimating samples' instance hardness.

    Notes
    -----
    The method is based on [1]_.

    This class does not support multi-class.

    References
    ----------
    .. [1] D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
       "An instance level analysis of data complexity." Machine learning
       95.2 (2014): 225-256.

    """

    def __init__(self, estimator, ratio='auto', return_indices=False, cv=5,
                 random_state=None, verbose=True, n_jobs=-1):
        """Initialisation of Instance Hardness Threshold object.

        Parameters
        ----------
        estimator : sklearn classifier
            Classifier to be used in to estimate instance hardness of the
            samples.

        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

        cv : int, optional (default=5)
            Number of folds to be used when estimating samples' instance
            hardness.

        return_indices : bool, optional (default=False)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        n_jobs : int, optional (default=-1)
            The number of thread to open when it is possible.

        Returns
        -------
        None

        """
        super(InstanceHardnessThreshold, self).__init__(
            ratio=ratio,
            return_indices=return_indices,
            random_state=random_state,
            verbose=verbose)

        if not hasattr(estimator, 'predict_proba'):
            raise ValueError('Estimator does not have predict_proba method.')
        else:
            self.estimator_ = estimator

        self.cv = cv
        self.n_jobs = n_jobs

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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(InstanceHardnessThreshold, self).fit(X, y)

        return self

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

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(InstanceHardnessThreshold, self).transform(X, y)

        skf = StratifiedKFold(y, n_folds=self.cv, shuffle=False,
                              random_state=self.rs_)

        probabilities = np.zeros(y.shape[0], dtype=float)

        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.estimator_.fit(X_train, y_train)

            probs = self.estimator_.predict_proba(X_test)
            classes = self.estimator_.classes_
            probabilities[test_index] = [
                probs[l, np.where(classes == c)[0][0]]
                for l, c in enumerate(y_test)]

        # Compute the number of cluster needed
        if self.ratio_ == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio_)

        # Find the percentile corresponding to the top num_samples
        threshold = np.percentile(
            probabilities[y != self.min_c_],
            (1. - (num_samples / self.stats_c_[self.maj_c_])) * 100.)

        mask = np.logical_or(probabilities >= threshold, y == self.min_c_)

        # Sample the data
        X_resampled = X[mask]
        y_resampled = y[mask]

        if self.verbose:
            print("Under-sampling performed: {}".format(Counter(y_resampled)))

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(mask)[0]
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
