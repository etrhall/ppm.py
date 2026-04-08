import pandas as pd
import ppm


class PPMSimple(ppm.ppm_simple):
    def __init__(
        self,
        alphabet_size: int,
        order_bound: int = 10,
        shortest_deterministic: bool = True,
        exclusion: bool = True,
        update_exclusion: bool = True,
        escape: str = "c",
        debug_smooth: bool = False,
        alphabet_levels: list[str] = [],
    ) -> None:
        super().__init__(
            alphabet_size,
            order_bound,
            shortest_deterministic,
            exclusion,
            update_exclusion,
            escape,
            alphabet_levels,
        )

    def model_seq(
        self,
        seq: list[int],
        time: list = [],
        train: bool = True,
        predict: bool = True,
        return_distribution: bool = True,
        return_entropy: bool = True,
        generate: bool = False,
    ) -> pd.DataFrame:
        result = super().model_seq(
            seq, time, train, predict, return_distribution, return_entropy, generate
        )
        df = pd.DataFrame(
            {
                "symbol": result.symbol,
                "model_order": result.model_order,
                "information_content": result.information_content,
                "entropy": result.entropy,
                "distribution": result.distribution,
            }
        )
        return df
