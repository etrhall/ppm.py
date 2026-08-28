from itertools import chain
import pandas as pd

from ._ppm import ppm_simple


class PPMSimple(ppm_simple):
    def __init__(
        self,
        alphabet_size: int | None = None,
        order_bound: int = 10,
        shortest_deterministic: bool = True,
        exclusion: bool = True,
        update_exclusion: bool = True,
        escape: str = "c",
        debug_smooth: bool = False,
        alphabet_levels: list = [],
    ) -> None:
        """
        Creates simple PPM model

        Creates a simple PPM model, that is, a PPM model
        without any non-traditional features such as memory decay.

        :param int alphabet_size: The size of the alphabet upon which the
            model will be trained and tested.
            If not provided, this will be taken as ``len(alphabet_levels)``.

        :param int order_bound: The model's Markov order bound.
            For example, an order bound of two means that the model makes
            predictions based on the two preceding symbols.

        :param bool shortest_deterministic: If `True``, the model will 'select'
            the shortest available order that provides a deterministic
            prediction, if such an order exists, otherwise defaulting to the
            longest available order.
            For a given prediction, if this rule results in a lower model
            order than would have otherwise been selected, then full counts
            (not update-excluded counts) will be used for the highest model
            order (but not for lower model orders).
            This behaviour matches the implementations of PPM* in
            [Pearce2005]_ and [Bunton1996]_.

        :param bool exclusion: If ``True``, implements exclusion as defined in
            [Pearce2005]_ and [Bunton1996]_.

        :param bool update_exclusion: If ``True``, implements update exclusion
            as defined in [Pearce2005]_ and [Bunton1996]_.

        :param str escape: Takes values ``'a'``, ``'b'``, ``'c'``, ``'d'``, or ``'ax'``,
            corresponding to the eponymous escape methods [Pearce2005]_.
            Note that there is a mistake in the definition of escape method 'AX'
            in [Pearce2005]_; the denominator of lambda needs to have 1 added.
            This is what we implement here. Note that Pearce's LISP
            implementation correctly adds 1 here, like us.

        :param bool debug_smooth: Whether to print debug output for smoothing
            (currently rather messy and ad hoc).

        :param list alphabet_levels: Optional list of levels for the alphabet.

        .. note::
            The implementation does not scale well to very large order bounds (> 50).

        .. seealso::
            - ``new_ppm_decay``
            - ``model_seq``
        """
        if not alphabet_size and len(alphabet_levels) == 0:
            raise Exception("either alphabet size or levels must be specified")
        if not alphabet_size:
            alphabet_size = len(alphabet_levels)
        if len(alphabet_levels) > alphabet_size:
            raise Exception(
                "number of alphabet levels cannot be greater than alphabet_size"
            )
        self.alphabet_levels = alphabet_levels

        assert isinstance(alphabet_size, int), "alphabet_size must be int (or None)"
        assert isinstance(order_bound, int), "order_bound must be int"
        assert isinstance(shortest_deterministic, bool), (
            "shortest_deterministic must be bool"
        )
        assert isinstance(exclusion, bool), "exclusion must be bool"
        assert isinstance(update_exclusion, bool), "update_exclusion must be bool"
        escape_methods = ["a", "b", "c", "d", "ax"]
        if escape not in escape_methods:
            raise Exception(f"escape parameter must be one of {escape_methods}")

        super().__init__(
            alphabet_size,
            order_bound,
            shortest_deterministic,
            exclusion,
            update_exclusion,
            escape,
        )
        self.debug_smooth = debug_smooth

    def model_seq(
        self,
        seq: list,
        time: list[float] = [],
        train: bool = True,
        predict: bool = True,
        return_distribution: bool = True,
        return_entropy: bool = True,
        generate: bool = False,
    ) -> pd.DataFrame:
        """
        Model sequence

        Analyses or generates a sequence using a PPM model.

        :param list seq:  A list defining the input sequence.

        :param list time: Timepoints corresponding to each element of the sequence.
        Only used by certain model types (e.g. decay-based models).

        :param bool train: Whether or not the model should learn from the incoming sequence.

        :param bool predict: Whether or not to generate predictions for each
            element of the incoming sequence.

        :param bool return_distribution: Whether or not to return the
            conditional distribution over each potential continuation as part
            of the model output (ignored if ``predict = False``).

        :param bool return_entropy: Whether or not to return the entropy of
            each event prediction (ignored if ``predict = False``).

        :param bool generate: If ``True``, the output will correspond to a
            newly generated sequence with length as specified by the ``seq`` argument,
            produced by sampling from the model's predictive distribution.
            The default is ``False``.

        :return: A ``pandas.DataFrame`` which will be empty if ``predict = False``
            and otherwise will contain one row for each element in the sequence,
            with the following columns:
                - ``symbol`` - the symbol being predicted. This should be
                  identical to the input argument ``seq``.
                - ``i`` - the index of symbol in alphabet levels.
                - ``model_order`` - the model order used for generating predictions.
                - ``information_content`` - the information content
                  (i.e., negative log probability, base 2) of the observed symbol.
                - ``entropy`` - the expected information content when predicting the symbol.
                - ``distribution`` - the predictive probability distribution
                  for the symbol, conditioned on the preceding symbols.
        """
        alphabet_diff = set(seq) - set(self.alphabet_levels)
        if len(alphabet_diff) > 0:
            if (len(self.alphabet_levels) + len(alphabet_diff)) > self.alphabet_size:
                raise Exception(
                    "number of alphabet levels greater than model alphabet size"
                )
            self.alphabet_levels = self.alphabet_levels + list(alphabet_diff)
        iseq = [self.alphabet_levels.index(x) for x in seq]

        assert isinstance(train, bool), "train must be bool"
        assert isinstance(predict, bool), "predict must be bool"
        assert isinstance(return_distribution, bool), "return_distribution must be bool"
        assert isinstance(return_entropy, bool), "return_entropy must be bool"
        assert isinstance(generate, bool), "generate must be bool"

        result = super().model_seq(
            iseq, time, train, predict, return_distribution, return_entropy, generate
        )
        df = pd.DataFrame(
            {
                "symbol": [self.alphabet_levels[i] for i in result.symbol],
                "i": result.symbol,
                "model_order": result.model_order,
                "information_content": result.information_content,
                "entropy": result.entropy,
                "distribution": result.distribution,
            }
        )
        return df

    def model_poly(
        self,
        seqs: list[list],
        times: list[list[float]] = [],
        train: bool | list[bool] = True,
        predict: bool | list[bool] = True,
        return_distribution: bool = True,
        return_entropy: bool = True,
    ) -> pd.DataFrame:
        """
        Model a list of sequences

        Analyses concurrent sequences using a PPM model.

        :param list seqs: A list of lists defining the input sequences.

        :param list times: List of lists of timepoints corresponding to each element of the
            sequences. If empty symbol index is used.

        :param bool train: Whether or not the model should learn from each incoming sequence.

        :param bool predict: Whether or not to generate predictions for each
            element of each incoming sequence.

        :param bool return_distribution: Whether or not to return the
            conditional distribution over each potential continuation as part
            of the model output (ignored if ``predict = False``).

        :param bool return_entropy: Whether or not to return the entropy of
            each event prediction (ignored if ``predict = False``).

        :return: A ``pandas.DataFrame`` which will be empty if ``predict = False``
            and otherwise will contain one row for each element in each sequence,
            with the following columns:
                - ``time`` - the time of each event.
                - ``stream`` - index of input sequence in ```seqs`.
                - ``symbol`` - the symbol being predicted. This should be
                  identical to the input argument ``seqs``.
                - ``i`` - the index of symbol in alphabet levels.
                - ``model_order`` - the model order used for generating predictions.
                - ``information_content`` - the information content
                  (i.e., negative log probability, base 2) of the observed symbol.
                - ``entropy`` - the expected information content when predicting the symbol.
                - ``distribution`` - the predictive probability distribution
                  for the symbol, conditioned on the preceding symbols.
        """
        alphabet_diff = set(chain.from_iterable(seqs)) - set(self.alphabet_levels)
        if len(alphabet_diff) > 0:
            if (len(self.alphabet_levels) + len(alphabet_diff)) > self.alphabet_size:
                raise Exception(
                    "number of alphabet levels greater than model alphabet size"
                )
            self.alphabet_levels = self.alphabet_levels + list(alphabet_diff)
        iseqs = [[self.alphabet_levels.index(x) for x in seq] for seq in seqs]

        N = len(seqs)
        if not times:
            times = [
                [max([0] + self.all_time) + i + 1 for i in range(len(x))] for x in seqs
            ]
        if not isinstance(train, list):
            train = [train] * N
        if not isinstance(predict, list):
            predict = [predict] * N

        assert len(times) == N, "number of input sequences and times must match"
        assert len(train) == N, "number of input sequences and train states must match"
        assert len(predict) == N, "number of input sequences and predicts must match"
        assert all(isinstance(x, bool) for x in train), "all train must be bools"
        assert all(isinstance(x, bool) for x in predict), "all predict must be bools"
        assert isinstance(return_distribution, bool), "return_distribution must be bool"
        assert isinstance(return_entropy, bool), "return_entropy must be bool"
        for i, (s, t) in enumerate(zip(seqs, times)):
            assert len(s) == len(t), f"seqs[{i}] and times[{i}] must be of same length"

        results = super().model_poly(
            iseqs, times, train, predict, return_distribution, return_entropy
        )
        df = pd.DataFrame(
            {
                "time": chain.from_iterable(times),
                "stream": [i for i in range(N) for _ in results[i].symbol],
                "symbol": [self.alphabet_levels[i] for x in results for i in x.symbol],
                "i": chain.from_iterable(x.symbol for x in results),
                "model_order": chain.from_iterable(x.model_order for x in results),
                "information_content": chain.from_iterable(
                    x.information_content for x in results
                ),
                "entropy": chain.from_iterable(x.entropy for x in results),
                "distribution": chain.from_iterable(x.distribution for x in results),
            }
        )
        return df.sort_values(by=["time", "stream"], ignore_index=True)
