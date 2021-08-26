import os
import math
import tempfile
from pathlib import Path
import torch
from torchvision import utils
import cog

from generate import sample, get_mean_style
from model import StyledGenerator

SIZE = 1024


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = StyledGenerator(512).to(self.device)
        print("Loading checkpoint")
        self.generator.load_state_dict(
            torch.load(
                "stylegan-1024px-new.model",
                map_location=self.device,
            )["g_running"],
        )
        self.generator.eval()

    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, seed):
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)
        print(f"seed: {seed}")

        mean_style = get_mean_style(self.generator, self.device)
        step = int(math.log(SIZE, 2)) - 2
        img = sample(self.generator, step, mean_style, 1, self.device)
        output_path = Path(tempfile.mkdtemp()) / "output.png"
        utils.save_image(img, output_path, normalize=True)
        return output_path
