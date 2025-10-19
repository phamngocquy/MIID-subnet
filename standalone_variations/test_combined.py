#!/usr/bin/env python3
"""
Combined Test: 30% Rule-based + Balanced Configuration + 15 Total Variations
"""

from name_variations import generate_name_variations
from typing import List, Optional

def generate_name_variations_balanced(
    name: str,
    expected_count: int = 10,
    miner_salt: int = 0,
    batch_salt: int = 0,
    country: Optional[str] = None,
    rule_percentage: Optional[int] = None,
    selected_rules: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate name variations using balanced similarity configurations.
    
    Uses the specific balanced configurations:
    - Phonetic: {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    - Orthographic: {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    
    Args:
        name: The original name to generate variations for
        expected_count: Total number of variations to generate
        miner_salt: Salt for miner-specific randomization
        batch_salt: Salt for batch-specific randomization
        country: Country name for script detection
        rule_percentage: Percentage of rule-based variations (0-100)
        selected_rules: List of selected rules for variation generation
    
    Returns:
        List of name variations
    """
    # Use balanced configurations
    phonetic_config = {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    orthographic_config = {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    
    return generate_name_variations(
        name=name,
        expected_count=expected_count,
        miner_salt=miner_salt,
        batch_salt=batch_salt,
        country=country,
        phonetic_config=phonetic_config,
        orthographic_config=orthographic_config,
        rule_percentage=rule_percentage,
        selected_rules=selected_rules
    )

def test_combined():
    """Single combined test with all requirements"""
    rule_percentage = 3
    
    total_variations = 15
    print("="*80)
    print(f"COMBINED TEST: {rule_percentage}% RULE-BASED + BALANCED CONFIG + 15 TOTAL VARIATIONS")
    print("="*80)
    print(f"  - Total variations: {total_variations}")
    print("="*80)
    
    # Test cases
    test_cases = [
        ("John Smith", "United States"),
        # ("José García", "Spain"),
        # ("François Dubois", "France"),
        # ("محمد أحمد", "Syria"),
        # ("王小明", "China"),
    ]
    
    # Available rules
    available_rules = [
        "replace_spaces_with_random_special_characters",
        "replace_double_letters_with_single_letter", 
        "replace_random_vowel_with_random_vowel",
        "replace_random_consonant_with_random_consonant",
        "shorten_name_to_abbreviations",
        "replace_random_letter_with_random_letter",
    ]
    
    for i, (name, country) in enumerate(test_cases, 1):
        print(f"\n{'-'*60}")
        print(f"TEST {i}: {name} from {country}")
        print(f"{'-'*60}")
        
        phonetic_config = {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
        orthographic_config = {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    
        combined_variations = generate_name_variations(
            name=name,
            expected_count=total_variations,
            miner_salt=1,
            batch_salt=1,
            country=country,
            phonetic_config=phonetic_config,
            orthographic_config=orthographic_config,
            rule_percentage=3,
            selected_rules=[]
        )
        
        print(f"Generated {len(combined_variations)} variations:")
        for j, variation in enumerate(combined_variations, 1):
            print(f"  {j:2d}. {variation}")
        
        # Analysis
        print(f"\nAnalysis:")
        # Check if we got the expected count
        if len(combined_variations) == total_variations:
            print(f"  ✅ SUCCESS: Got exactly 15 variations as expected!")
        else:
            print(f"  ❌ ISSUE: Expected 15 variations, got {len(combined_variations)}")
            print(f"  This indicates a problem with the variation generation logic.")
            
        # Show first few variations to verify quality
        print(f"\nvariations:")
        for j, var in enumerate(combined_variations[0:total_variations], 1):
            print(f"  {j}. {var}")
    
    print(f"\n{'='*80}")
    print("COMBINED TEST COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_combined()
