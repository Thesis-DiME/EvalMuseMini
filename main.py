import hydra
from omegaconf import DictConfig
from pipeline import EvalMusePipeline
import time
from hydra.utils import get_original_cwd
from pathlib import Path


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = EvalMusePipeline(cfg)
    pipeline.evaluate()


if __name__ == "__main__":
    base_dir = Path('/app')
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    with open(base_dir / "results/individual" / "eval_muse.txt", "w") as f:
        f.write(f"TIME: {end_time}")
