import pandas as pd
import _ppm


class PPMSimple(_ppm.ppm_simple):
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
        time: list = [],
        train: bool = True,
        predict: bool = True,
        return_distribution: bool = True,
        return_entropy: bool = True,
        generate: bool = False,
    ) -> pd.DataFrame:
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
