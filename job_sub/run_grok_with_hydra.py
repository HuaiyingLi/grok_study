import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_grok_with_hydra(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    run_grok_with_hydra()