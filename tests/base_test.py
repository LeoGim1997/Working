from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class Test:
    processingPath: Path = Path(__file__).parents[1] / "processing"
    outputPath: Path = Path(__file__).parents[1] / "output"
    

    @classmethod
    def saveOutput(
        im1: np.ndarray, im2: np.ndarray, cmp="gray", figname: str = "TestName"
    ) -> None:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im1, cmap=cmp)
        plt.subplot(1, 2, 2)
        plt.imshow(im2, cmap=cmp)
        plt.savefig((Test.outputPath / "figname.svg").as_posix())
