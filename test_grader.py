#!/usr/bin/env python3
"""
Test the grader with sample outputs to verify regex patterns work.
"""

import sys
sys.path.insert(0, str(__file__).replace('test_grader.py', ''))

from runner import BenchmarkConfig
from grader import RegexGrader, GradingResult, create_grader_from_test_file


def test_grader():
    """Test grader with sample good and bad outputs."""
    
    # Load test cases
    test_config = BenchmarkConfig.from_json("test_cases.json")
    test_case = test_config.test_cases[0]  # Test 1: "Sue told Sara qu xi shou ba"
    
    print(f"Test: {test_case.title}")
    print(f"Prompt: {test_case.prompt}")
    print(f"Expected: {test_case.expected_literal_chars}")
    print()
    
    grader = create_grader_from_test_file("test_cases.json")
    
    # Good response
    good_response = """[Phonetic Sandbox]
Literal Translation: 去洗手吧

| Chinese | Pinyin | English |
| :--- | :--- | :--- |
| 去洗手吧 | qù xǐ shǒu ba | go wash hand particle |

**Tutor's Note:**
- **Grammar/Vocab:** qù (go), xǐ (wash), shǒu (hand), ba (sentence-final particle for suggestion/command)
- **Natural Alternative:** 快去洗手吧 (kuài qù xǐ shǒu ba) - "Hurry up and go wash your hands" """
    
    print("=== GOOD RESPONSE ===")
    result = grader.grade(test_case.id, good_response, model="test-model", provider="test")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score:.2f}")
    print(f"Details: {result.details}")
    print(f"Errors: {result.errors}")
    print()
    
    # Bad response - translates English names
    bad_response = """[Phonetic Sandbox]
Literal Translation: 苏告诉莎拉去洗手吧

| Chinese | Pinyin | English |
| :--- | :--- | :--- |
| 苏告诉莎拉去洗手吧 | sū gào sù shā lā qù xǐ shǒu ba | sue told sara go wash hand particle |

**Tutor's Note:**
- **Grammar/Vocab:** sū (Sue), gào sù (told), shā lā (Sara), qù (go), xǐ (wash), shǒu (hand), ba (particle)
- **Natural Alternative:** 苏告诉莎拉去洗手吧 """
    
    print("=== BAD RESPONSE (translates English names) ===")
    result = grader.grade(test_case.id, bad_response, model="test-model", provider="test")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score:.2f}")
    print(f"Details: {result.details}")
    print(f"Errors: {result.errors}")
    print()
    
    # Bad response - wrong characters
    wrong_chars = """[Phonetic Sandbox]
Literal Translation: 去洗手

| Chinese | Pinyin | English |
| :--- | :--- | :--- |
| 去洗手 | qù xǐ shǒu | go wash hand |

**Tutor's Note:**
- **Grammar/Vocab:** qù (go), xǐ (wash), shǒu (hand)
- **Natural Alternative:** 去洗手吧 (qù xǐ shǒu ba)"""
    
    print("=== BAD RESPONSE (missing 'ba') ===")
    result = grader.grade(test_case.id, wrong_chars, model="test-model", provider="test")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score:.2f}")
    print(f"Details: {result.details}")
    print(f"Errors: {result.errors}")
    print()
    
    # Test all test cases
    print("=== TESTING ALL CASES WITH GOOD RESPONSES ===")
    for tc in test_config.test_cases:
        # Create a "perfect" response based on expected values
        perfect_response = f"""[Phonetic Sandbox]
Literal Translation: {tc.expected_literal_chars}

| Chinese | Pinyin | English |
| :--- | :--- | :--- |
| {tc.expected_pinyin_table} | {tc.expected_pinyin_tone_marked} | {tc.expected_english_literal} |

**Tutor's Note:**
- **Grammar/Vocab:** {tc.expected_grammar_vocab}
- **Natural Alternative:** {tc.expected_natural_alternative}"""
        
        result = grader.grade(tc.id, perfect_response, model="test", provider="test")
        status = "✓" if result.passed else "✗"
        print(f"  {status} Test {tc.id}: {tc.title} - Score: {result.score:.2f}")
        if not result.passed:
            print(f"      Errors: {result.errors}")


if __name__ == "__main__":
    test_grader()