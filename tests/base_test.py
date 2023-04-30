from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class Test:
    processingPath: Path = Path(__file__).parents[1] / "processing"
    outputPath: Path = Path(__file__).parents[1] / "output"
    resourcePath : Path = Path(__file__).parents[1] / "resources"

    @classmethod
    def saveOutput(
        im1: np.ndarray, im2: np.ndarray, cmap="gray", figname: str = "TestName"
    ) -> None:
        plt.figure()
        plt.imshow(im1)
