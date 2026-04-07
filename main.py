"""Project entry point for training."""
import jax

from train import config, train

def main():
    jax.config.update("jax_traceback_filtering", "off")
    cfg = config.default()
    train.train(cfg)

if __name__ == "__main__":
    main()
