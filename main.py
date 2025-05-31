import hydra
from omegaconf import DictConfig
from pipeline import EvalMusePipeline
import time


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = EvalMusePipeline(cfg)
    pipeline.evaluate()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"EVAL MUSE TIME: {time.time()-start_time}")