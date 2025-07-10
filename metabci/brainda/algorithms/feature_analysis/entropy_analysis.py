import numpy as np
import mne

class EntropyAnalysis:
    def __init__(self, fs):
        """
        - author: X
        - Create on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            fs (float): Sampling rate of the EEG data
        """
        self.fs = fs

    def differential_entropy(self, data):
        """
        - author: X
        - Created on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            data (ndarray): EEG data array, shape (n_channels, n_times)

        Returns:
            de (ndarray): Differential entropy for each channel
        """
        # Assuming Gaussian distribution: DE = 0.5 * ln(2 * pi * e * var)
        var = np.var(data, axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        return de

    def sample_entropy(self, data, m=2, r=None):
        """
        - author: X
        - Created on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            data (ndarray): EEG data, shape (n_channels, n_times)
            m (int): Embedding dimension, default 2
            r (float): Tolerance (if None, set to 0.2 * std of data)

        Returns:
            sampen (ndarray): Sample entropy for each channel
        """
        n_channels, n_times = data.shape
        sampen = np.zeros(n_channels)
        for i in range(n_channels):
            x = data[i]
            if r is None:
                r = 0.2 * np.std(x)
            # Build templates
            def _phi(m):
                patterns = np.array([x[j:j+m] for j in range(n_times - m + 1)])
                C = []
                for pat in patterns:
                    dist = np.max(np.abs(patterns - pat), axis=1)
                    C.append(np.sum(dist <= r) - 1)
                return np.sum(C) / (n_times - m + 1)

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            sampen[i] = -np.log(phi_m1 / phi_m)
        return sampen

    def approximate_entropy(self, data, m=2, r=None):
        """
        - author: X
        - Created on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            data (ndarray): EEG data, shape (n_channels, n_times)
            m (int): Embedding dimension, default 2
            r (float): Tolerance (if None, set to 0.2 * std of data)

        Returns:
            apen (ndarray): Approximate entropy for each channel
        """
        n_channels, n_times = data.shape
        apen = np.zeros(n_channels)
        for i in range(n_channels):
            x = data[i]
            if r is None:
                r = 0.2 * np.std(x)
            def _phi(m):
                patterns = np.array([x[j:j+m] for j in range(n_times - m + 1)])
                C = []
                for pat in patterns:
                    dist = np.max(np.abs(patterns - pat), axis=1)
                    C.append(np.sum(dist <= r) / (n_times - m + 1))
                return np.sum(np.log(C)) / (n_times - m + 1)

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            apen[i] = phi_m - phi_m1
        return apen

    def fuzzy_entropy(self, data, m=2, r=None):
        """
        - author: X
        - Created on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            data (ndarray): EEG data, shape (n_channels, n_times)
            m (int): Embedding dimension, default 2
            r (float): Tolerance (if None, set to 0.2 * std of data)

        Returns:
            fuzzyen (ndarray): Fuzzy entropy for each channel
        """
        n_channels, n_times = data.shape
        fuzzyen = np.zeros(n_channels)
        for i in range(n_channels):
            x = data[i]
            if r is None:
                r = 0.2 * np.std(x)
            def _fuzzy(m):
                patterns = np.array([x[j:j+m] for j in range(n_times - m + 1)])
                D = np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2)
                mu = np.exp(- (D**2) / r)
                return np.sum(mu) / ((n_times - m + 1)**2)

            Fm = _fuzzy(m)
            Fm1 = _fuzzy(m + 1)
            fuzzyen[i] = -np.log(Fm1 / Fm)
        return fuzzyen

    def permutation_entropy(self, data, m=3, delay=1):
        """
        - author: X
        - Created on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            data (ndarray): EEG data, shape (n_channels, n_times)
            m (int): Embedding dimension (order), default 3
            delay (int): Time delay, default 1

        Returns:
            pe (ndarray): Permutation entropy for each channel
        """
        from math import factorial
        n_channels, n_times = data.shape
        pe = np.zeros(n_channels)
        for i in range(n_channels):
            x = data[i]
            perms = {}
            c = 0
            for j in range(n_times - delay*(m-1)):
                seq = x[j:j+delay*m:delay]
                rank = tuple(np.argsort(seq))
                perms[rank] = perms.get(rank, 0) + 1
                c += 1
            p = np.array(list(perms.values())) / c
            pe[i] = -np.sum(p * np.log(p)) / np.log(factorial(m))
        return pe

    def fun_topoplot(self, data, ch_names, sfreq=None, ch_types="eeg"):
        """
        - author: X
        - Created on: 2025-05-20
        - update log:
            2025-05-20 by X

        Args:
            data (ndarray): 1D array of feature values per channel
            ch_names (list): Names of channels in 10-20 standard
            sfreq (float): Sampling frequency, default self.fs
            ch_types (str or list): Channel types
        """
        if sfreq is None:
            sfreq = self.fs
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        evoked = mne.EvokedArray(data[:, None], info)
        evoked.set_montage("standard_1005")
        mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=True)
