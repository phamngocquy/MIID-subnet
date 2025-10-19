```python
# 2. Set up phonetic similarity distribution with weighted selection
phonetic_configs_with_weights = [
    # Balanced distribution - high weight for balanced testing
    ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
    # Focus on Medium similarity - most common real-world scenario
    ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
    # Focus on Far similarity - important for edge cases
    ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
    # Light-Medium mix - moderate weight
    ({"Light": 0.5, "Medium": 0.5}, 0.12),
    # Medium-Far mix - moderate weight
    ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
    # Only Medium similarity - common case
    ({"Medium": 1.0}, 0.08),
    # High Light but not 100% - reduced frequency
    ({"Light": 0.7, "Medium": 0.3}, 0.05),
    # Only Far similarity - edge case
    ({"Far": 1.0}, 0.03),
    # Only Light similarity - reduced frequency
    ({"Light": 1.0}, 0.02),
]

# 3. Set up orthographic similarity distribution with weighted selection
orthographic_configs_with_weights = [
    # Balanced distribution - high weight for balanced testing
    ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
    # Focus on Medium similarity - most common real-world scenario
    ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
    # Focus on Far similarity - important for edge cases
    ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
    # Light-Medium mix - moderate weight
    ({"Light": 0.5, "Medium": 0.5}, 0.12),
    # Medium-Far mix - moderate weight
    ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
    # Only Medium similarity - common case
    ({"Medium": 1.0}, 0.08),
    # High Light but not 100% - reduced frequency
    ({"Light": 0.7, "Medium": 0.3}, 0.05),
    # Only Far similarity - edge case
    ({"Far": 1.0}, 0.03),
    # Only Light similarity - reduced frequency
    ({"Light": 1.0}, 0.02),
]

# Helper function for weighted random selection
# complex query
def weighted_random_choice(configs_with_weights):
    configs, weights = zip(*configs_with_weights)
    return random.choices(configs, weights=weights, k=1)[0]

# Select configurations using weighted random selection
phonetic_config = weighted_random_choice(phonetic_configs_with_weights)
orthographic_config = weighted_random_choice(orthographic_configs_with_weights)

# 4. Randomly choose rule_percentage for this query (e.g. 10-60%)
rule_percentage = random.randint(10, 60)

if self.use_default_query:
    bt.logging.info("Using default query template")
    variation_count = 10
    phonetic_config = {"Medium": 0.5}
    orthographic_config = {"Medium": 0.5}
    rule_percentage = 30  # fallback for default
```