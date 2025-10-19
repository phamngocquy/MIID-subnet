import asyncio, argparse, os
import bittensor as bt
from MIID.protocol import IdentitySynapse
from MIID.utils import config as miid_cfg
from neurons.miner import Miner

# Build a proper bt.Config using the project’s arg builders
parser = argparse.ArgumentParser()
bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
bt.logging.add_args(parser)
bt.axon.add_args(parser)
miid_cfg.add_args(None, parser)
miid_cfg.add_miner_args(None, parser)

# Create config with defaults, then override a few fields
cfg = bt.config(parser)
cfg.logging.logging_dir = "/tmp/miid_miner_test"
cfg.neuron.model_name = "tinyllama:latest"     # small model for quick tests
cfg.neuron.ollama_url = "http://127.0.0.1:11434"
os.makedirs(cfg.logging.logging_dir, exist_ok=True)

async def main():
    miner = Miner(config=cfg)

    identities = [
        ["Nguyen Van A", "1990-01-15", "Vietnam"],
        ["John Smith",   "1985-07-20", "USA"],
    ]
    query_template = "Give me 10 comma separated alternative spellings of the name {name}. Provide only names."

    syn = IdentitySynapse(identity=identities, query_template=query_template, variations={}, timeout=60.0)
    out = await miner.forward(syn)
    print("Variations (preview):")
    for seed_name, triplets in out.variations.items():
        print(seed_name, "->", triplets[:5])

asyncio.run(main())