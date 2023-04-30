from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class Test:
    processingPath: Path = Path(__file__).parents[1] / "processing"
    outputPath: Path = Path(__file__).parents[1] / "output"
    resourcePath: Path = Path(__file__).parents[1] / "resources"

    @classmethod
    def saveOutput(
        cls,
        img1: np.ndarray,
        img2: np.ndarray,
        icmap: str = "gray",
        figName: str = "Test",
        ext: str = "png",
    ) -> None:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap=icmap)
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap=icmap)
        plt.savefig((cls.outputPath / f"{figName}.{ext}").as_posix())
