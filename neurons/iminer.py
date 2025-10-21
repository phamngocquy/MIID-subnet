# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Name Variation Miner Module

This module implements a Bittensor miner that generates alternative spellings for names
using a local LLM (via Ollama).
######### Ollama should be installed and running on the machine. ########
The miner receives requests from validators containing
a list of names and a query template, processes each name through the LLM, extracts
the variations from the LLM's response, and returns them to the validator.

The miner follows these steps:
1. Receive a request with names and a query template
2. For each name, query the LLM to generate variations
3. Process the LLM responses to extract clean variations
4. Return the variations to the validator

The processing logic handles different response formats from LLMs, including:
- Comma-separated lists
- Line-separated lists
- Space-separated lists with numbering

For debugging and analysis, the miner also saves:
- Raw LLM responses
- Processed variations in JSON format
- A pandas DataFrame with the variations

Each mining run is saved with a unique timestamp identifier to distinguish between
different runs and facilitate analysis of results over time.
"""

import time
import typing
import bittensor as bt
import ollama
import pandas as pd
import os
import numpy as np
from typing import List, Dict
from tqdm import tqdm

# Bittensor Miner Template:
from MIID.protocol import IdentitySynapse

# import base miner class which takes care of most of the boilerplate
from MIID.base.miner import BaseMinerNeuron

from bittensor.core.errors import NotVerifiedException
from MIID.validator.reward import transliterate_name_with_llm
from MIID.validator.rule_evaluator import is_letter_duplicated
from extensions import (
    address_variations,
    dob_variations,
    input_parse,
    llm_variations,
    name_variations,
    rule_based_variations,
)
from extensions.rule_based_variations import detect_script


class Miner(BaseMinerNeuron):
    """
    Name Variation Miner Neuron

    This miner receives requests from validators to generate alternative spellings for names,
    and responds with variations generated using a local LLM (via Ollama).

    The miner handles the following tasks:
    - Processing incoming requests for name variations
    - Querying a local LLM to generate variations
    - Extracting and cleaning variations from LLM responses
    - Returning the processed variations to the validator
    - Saving intermediate results for debugging and analysis

    Each mining run is saved with a unique timestamp identifier to distinguish between
    different runs and facilitate analysis of results over time.

    Configuration:
    - model_name: The Ollama model to use (default: 'tinyllama:latest')
    - output_path: Directory for saving mining results (default: logging_dir/mining_results)
    """

    WHITELISTED_VALIDATORS = {
        "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54": "RoundTable21",
        "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr": "Yanez",
        "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs": "Yuma",
        "5HbUFHW4XVhbQvMbSy7WDjvhHb62nuYgP1XBsmmz9E2E2K6p": "OpenTensor",
        "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54": "Rizzo",
        "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN": "Tensora",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u": "Tao.bot",
        "5GuPvuyKBJAWQbEGAkMbfRpG5qDqqhML8uDVSWoFjqcKKvDU": "Testnet_omar",
        "5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt": "Testnet_asem",
        "5G1kdTF1w1vF7Jg2NgYUDZdDySs8tpSP7VzS92nJyzUvKJkx": "mylocal",
    }

    def __init__(self, config=None):
        """
        Initialize the Name Variation Miner.

        Sets up the LLM client and creates directories for storing mining results.
        Each run will be saved in a separate directory with a unique timestamp.

        Args:
            config: Configuration object for the miner
        """
        super(Miner, self).__init__(config=config)

        # Initialize the LLM client
        # You can override this in your config by setting model_name
        # Ensure we have a valid model name, defaulting to llama3.2:1b if not specified
        self.model_name = (
            getattr(self.config.neuron, "model_name", None)
            if hasattr(self.config, "neuron")
            else None
        )
        if self.model_name is None:
            # self.model_name = 'llama3.2:1b'
            self.model_name = "tinyllama:latest"
            bt.logging.info(
                f"No model specified in config, using default model: {self.model_name}"
            )
        self.model_name = "llama3.1:latest"

        bt.logging.info(f"Using LLM model: {self.model_name}")

        # Check if Ollama is available
        try:
            # Check if model exists locally first
            models = ollama.list().get("models", [])
            model_exists = any(model.get("name") == self.model_name for model in models)

            if model_exists:
                bt.logging.info(f"Model {self.model_name} already pulled")
            else:
                # Model not found locally, pull it
                bt.logging.info(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
        except Exception as e:
            bt.logging.error(f"Error with Ollama: {str(e)}")
            bt.logging.error(
                "Make sure Ollama is installed and running on this machine"
            )
            bt.logging.error(
                "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
            )
            bt.logging.error("Start Ollama: ollama serve")
            raise RuntimeError(
                "Ollama is required for this miner. Please install and start Ollama."
            )

        # Create a directory for storing mining results
        # This helps with debugging and analysis
        self.output_path = os.path.join(
            self.config.logging.logging_dir, "mining_results"
        )
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")
        self.axon.verify_fns[IdentitySynapse.__name__] = self._verify_validator_request

    async def _verify_validator_request(self, synapse: IdentitySynapse) -> None:
        """
        Rejects any RPC that is not cryptographically proven to come from
        one of the whitelisted validator hotkeys.

        Signature *must* be present and valid.  If anything is missing or
        incorrect we raise `NotVerifiedException`, which the Axon middleware
        converts into a 401 reply.
        """
        # ----------  basic sanity checks  ----------
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey = synapse.dendrite.hotkey
        # signature = synapse.dendrite.signature
        nonce = synapse.dendrite.nonce
        uuid = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        # 1 — is the sender even on our allow‑list?
        if hotkey not in self.WHITELISTED_VALIDATORS:
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        # 3 — run all the standard Bittensor checks (nonce window, replay,
        #     timeout, signature, …).  This *does not* insist on a signature,
        #     so we still do step 4 afterwards.
        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        bt.logging.info(f"Verifying message: {message}")

        await self.axon.default_verify(synapse)

        # 5 — all good ➜ let the middleware continue
        bt.logging.info(
            f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})"
        )

    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        """
        Process a name variation request by generating variations for each name.

        This is the main entry point for the miner's functionality. It:
        1. Receives a request with names and a query template
        2. Processes each name through the LLM
        3. Extracts variations from the LLM responses
        4. Returns the variations to the validator

        Each run is assigned a unique timestamp ID and results are saved in a
        dedicated directory for that run.

        Args:
            synapse: The IdentitySynapse containing names and query template

        Returns:
            The synapse with variations field populated with name variations
        """
        run_id = int(time.time())
        bt.logging.info(f"Starting run {run_id} for {len(synapse.identity)} names")

        timeout = getattr(synapse, "timeout", 120.0)
        bt.logging.info(
            f"Request timeout: {timeout:.1f}s for {len(synapse.identity)} names"
        )

        run_dir = os.path.join(self.output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        variations = {}

        # Process each identity in the request, respecting the timeout
        for _, identity in enumerate(
            tqdm(synapse.identity, desc="Processing identities")
        ):
            # Check if we're approaching the timeout (reserve 15% for processing)
            # Extract name, dob, and address from identity array
            name = identity[0] if len(identity) > 0 else "Unknown"
            dob = identity[1] if len(identity) > 1 else "Unknown"
            address = identity[2] if len(identity) > 2 else "Unknown"

            is_latin = detect_script(name) == "latin"
            if not is_latin:
                transliterate_name = (
                    transliterate_name_with_llm(
                        name, name_variations.detect_script(name).name
                    )
                    .strip()
                    .capitalize()
                )
            else:
                transliterate_name = name

            formatted_query = synapse.query_template.replace(
                "{name}", transliterate_name
            )
            formatted_query = formatted_query.replace("{address}", address)
            formatted_query = formatted_query.replace("{dob}", dob)
            bt.logging.info(f"Formatted query: {formatted_query}")

            variation_count = input_parse.find_variation_count(formatted_query)

            bt.logging.info(
                f"""Processing {name} - is latin: {is_latin}, name: {transliterate_name},
                DOB: {dob}, Address: {address}, expecting {variation_count} variations"""
            )

            llm_name_variation = llm_variations.generate_variations(
                formatted_query, transliterate_name, is_latin
            )
            bt.logging.critical(f"llm_name_variation: {llm_name_variation}")

            name_variation_rule_based = []
            if is_latin:
                name_variation_rule_based = rule_based_variations.generate_variations(
                    transliterate_name,
                    formatted_query,
                    miner_salt=1,
                    batch_salt=1,
                )

            name_variation = [
                item.lower()
                for item in (name_variation_rule_based + llm_name_variation)
            ]

            dob_variation = dob_variations.generate_dob_variations(
                seed_dob=dob, expected_count=variation_count
            )

            address_variation = address_variations.generate_address_variations(
                address, expected_count=variation_count
            )

            print(dob_variation)
            print(name_variation)
            print(address_variation)

            bt.logging.critical(
                f"name_variation: {name_variation} | variation_count: {variation_count}"
            )
            assert len(name_variation) >= variation_count

            for idx in range(len(dob_variation)):
                var_name = name_variation[idx] if idx < len(name_variation) else ""
                var_dob = dob_variation[idx] if idx < len(dob_variation) else ""
                var_address = (
                    address_variation[idx] if idx < len(address_variation) else ""
                )

                if name not in variations:
                    variations[name] = [[var_name, var_dob, var_address]]
                else:
                    variations[name].append([var_name, var_dob, var_address])

        print(variations)
        synapse.variations = variations
        return synapse

    def save_variations_to_json(
        self, name_variations: Dict[str, List[str]], run_id: int, run_dir: str
    ) -> None:
        """
        Save processed variations to JSON and DataFrame for debugging and analysis.

        This function saves the processed variations in multiple formats:
        1. A pandas DataFrame saved as a pickle file in the run-specific directory
        2. A JSON file with the name variations in the run-specific directory
        3. A JSON file with the model name and run ID in the main output directory

        Each file is named with the run ID to distinguish between different runs.

        Args:
            name_variations: Dictionary mapping names to variations
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
        """
        bt.logging.info(f"=================== Name variations: {name_variations}")
        bt.logging.info(f"=================== Run ID: {run_id}")
        bt.logging.info(f"=================== Run directory: {run_dir}")
        bt.logging.info("Saving variations to JSON and DataFrame")

        # Find the maximum number of variations for any name
        max_variations = (
            max([len(vars) for vars in name_variations.values()])
            if name_variations
            else 0
        )
        bt.logging.info(f"Maximum number of variations found: {max_variations}")

        # Create a DataFrame with columns for the name and each variation
        columns = ["Name"] + [f"Var_{i+1}" for i in range(max_variations)]
        result_df = pd.DataFrame(columns=columns)

        # Fill the DataFrame with names and their variations, padding with empty strings if needed
        for i, (name, variations) in enumerate(name_variations.items()):
            row_data = [name] + variations + [""] * (max_variations - len(variations))
            result_df.loc[i] = row_data

        # Note: We no longer need to clean the data here since it's already cleaned
        # in the process_variations function

        # Save DataFrame to pickle for backup and analysis
        # Include run_id in the filename
        # df_path = os.path.join(run_dir, f"variations_df_{run_id}.pkl")
        # result_df.to_pickle(df_path)

        # Convert DataFrame to JSON format
        json_data = {}
        for i, row in result_df.iterrows():
            name = row["Name"]
            # Extract non-empty variations
            variations = [var for var in row[1:] if var != ""]
            json_data[name] = variations

        # Save to JSON file
        # Include run_id in the filename
        # json_path = os.path.join(run_dir, f"variations_{run_id}.json")
        # import json
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=4)
        # bt.logging.info(f"Saved variations to: {json_path}")
        # bt.logging.info(f"DataFrame shape: {result_df.shape} with {max_variations} variation columns")

    def Clean_extra(
        self,
        payload: str,
        comma: bool,
        line: bool,
        space: bool,
        preserve_name_spaces: bool = False,
    ) -> str:
        """
        Clean the LLM output by removing unwanted characters.

        Args:
            payload: The text to clean
            comma: Whether to remove commas
            line: Whether to remove newlines
            space: Whether to remove spaces
            preserve_name_spaces: Whether to preserve spaces between names (for multi-part names)
        """
        # Remove punctuation and quotes
        payload = payload.replace(".", "")
        payload = payload.replace('"', "")
        payload = payload.replace("'", "")
        payload = payload.replace("-", "")
        payload = payload.replace("and ", "")

        # Handle spaces based on preservation flag
        if space:
            if preserve_name_spaces:
                # Replace multiple spaces with single space
                while "  " in payload:
                    payload = payload.replace("  ", " ")
            else:
                # Original behavior - remove all spaces
                payload = payload.replace(" ", "")

        if comma:
            payload = payload.replace(",", "")
        if line:
            payload = payload.replace("\\n", "")

        return payload.strip()

    def validate_variation(self, name: str, seed: str, is_multipart_name: bool) -> str:
        """
        Helper function to validate if a variation matches the seed name structure.

        Args:
            name: The variation to validate
            seed: The original seed name
            is_multipart_name: Whether the seed is a multi-part name

        Returns:
            str: The validated and cleaned variation, or np.nan if invalid
        """
        name = name.strip()
        if not name or name.isspace():
            return np.nan

        # Handle cases with colons (e.g., "Here are variations: Name")
        if ":" in name:
            name = name.split(":")[-1].strip()

        # Check length reasonability (variation shouldn't be more than 2x the seed length)
        if len(name) > 2 * len(seed):
            return np.nan

        # Check structure consistency with seed name
        name_parts = name.split()
        if is_multipart_name:
            # For multi-part seed names (e.g., "John Smith"), variations must also have multiple parts
            if len(name_parts) < 2:
                bt.logging.warning(
                    f"Skipping single-part variation '{name}' for multi-part seed '{seed}'"
                )
                return np.nan
        else:
            # For single-part seed names (e.g., "John"), variations must be single part
            if len(name_parts) > 1:
                bt.logging.warning(
                    f"Skipping multi-part variation '{name}' for single-part seed '{seed}'"
                )
                return np.nan

        return name

    async def blacklist(self, synapse: IdentitySynapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.

        This function implements security checks to ensure that only authorized
        validators can query this miner. It verifies:
        1. Whether the request has a valid dendrite and hotkey
        2. Whether the hotkey is one of the ones on the white list

        Args:
            synapse: A IdentitySynapse object constructed from the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether the request should be blacklisted
                - str: The reason for the decision
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        if synapse.dendrite.hotkey not in self.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # If all checks pass, allow the request
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: IdentitySynapse) -> float:
        """
        The priority function determines the order in which requests are handled.

        This function assigns a priority to each request based on the stake of the
        calling entity. Requests with higher priority are processed first, which
        ensures that validators with more stake get faster responses.

        Args:
            synapse: The IdentitySynapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.
                  Higher values indicate higher priority.
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # Get the UID of the caller
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        # Use the stake as the priority
        # Higher stake = higher priority
        priority = float(self.metagraph.S[caller_uid])

        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(
                f"----------------------------------Name Variation Miner running... {time.time()}"
            )
            time.sleep(30)
