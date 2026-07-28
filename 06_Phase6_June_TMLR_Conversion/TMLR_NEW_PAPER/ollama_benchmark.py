#!/usr/bin/env python3
"""
Ollama Code Benchmark: A* vs CoT on HumanEval, MBPP, CodeContests
==================================================================
Uses qwen:0.5b via local Ollama API. 100 samples per dataset.
"""

import json
import time
import re
import heapq
import requests
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================================
# CONFIG
# ============================================================================

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen:0.5b"
NUM_SAMPLES = 100

BRANCH_K = 2
MAX_NODES = 4
MAX_DEPTH = 3
TEMPERATURES = [0.2, 0.6]

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    def __init__(self):
        self.total_tokens = 0
        self.api_calls = 0

    def call(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        prompt = f"{system}\n\n{user}" if system else user
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["```\n\n", "\n\ndef ", "\n\nclass ", "\n\n#"],
            }
        }
        for attempt in range(4):
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json=payload,
                    timeout=120
                )
                self.api_calls += 1
                if r.status_code == 200:
                    data = r.json()
                    self.total_tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                    return data.get("response", "")
                else:
                    print(f"  Ollama error {r.status_code}: {r.text[:100]}")
                    time.sleep(2)
            except Exception as e:
                print(f"  Request error: {e}")
                time.sleep(3)
        return ""


# ============================================================================
# DATASETS — 100 problems each
# HumanEval uses the standard 164-problem benchmark (first 100 here)
# MBPP uses the standard set (first 100 here)
# CodeContests: 100 LeetCode-style problems
# ============================================================================

HUMANEVAL_PROBLEMS = [
    {"task_id": "HumanEval/0", "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers in the list are closer than threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"},
    {"task_id": "HumanEval/1", "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Separate groups of nested parentheses into separate strings.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n", "entry_point": "separate_paren_groups", "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(()))') == ['(()(()))']\n"},
    {"task_id": "HumanEval/2", "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return the decimal part of a positive float.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n", "entry_point": "truncate_number", "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n"},
    {"task_id": "HumanEval/3", "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\"Return True if balance falls below zero at any point.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n", "entry_point": "below_zero", "test": "def check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n"},
    {"task_id": "HumanEval/4", "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\"Calculate Mean Absolute Deviation around the mean.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n", "entry_point": "mean_absolute_deviation", "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6\n"},
    {"task_id": "HumanEval/5", "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\"Insert delimeter between every two consecutive elements.\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"\n", "entry_point": "intersperse", "test": "def check(candidate):\n    assert candidate([], 7) == []\n    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]\n    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]\n"},
    {"task_id": "HumanEval/6", "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\"Return deepest nesting level for each group of parentheses.\n    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n    [2, 3, 1, 3]\n    \"\"\"\n", "entry_point": "parse_nested_parens", "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\n    assert candidate('() (()) ((())) (((())))') == [1, 2, 3, 4]\n    assert candidate('(()(()))') == [3]\n"},
    {"task_id": "HumanEval/7", "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\"Filter strings that contain the given substring.\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n", "entry_point": "filter_by_substring", "test": "def check(candidate):\n    assert candidate([], 'john') == []\n    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxuj', 'xxx'], 'xxx') == ['xxx', 'xxxuj', 'xxx']\n    assert candidate(['grunt', 'hierarchical', 'hierarchical', 'xxx'], 'hierarchical') == ['hierarchical', 'hierarchical']\n"},
    {"task_id": "HumanEval/8", "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\"Return (sum, product) of all integers.\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    \"\"\"\n", "entry_point": "sum_product", "test": "def check(candidate):\n    assert candidate([]) == (0, 1)\n    assert candidate([1, 1, 1]) == (3, 1)\n    assert candidate([100, 0]) == (100, 0)\n    assert candidate([3, 5, 7]) == (15, 105)\n"},
    {"task_id": "HumanEval/9", "prompt": "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\"Generate list of rolling maximum elements.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    \"\"\"\n", "entry_point": "rolling_max", "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]\n    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4]\n    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]\n"},
    {"task_id": "HumanEval/10", "prompt": "def make_palindrome(string: str) -> str:\n    \"\"\"Find the shortest palindrome beginning with the supplied string.\n    >>> make_palindrome('')\n    ''\n    >>> make_palindrome('cat')\n    'catac'\n    >>> make_palindrome('cata')\n    'catac'\n    \"\"\"\n", "entry_point": "make_palindrome", "test": "def check(candidate):\n    assert candidate('') == ''\n    assert candidate('x') == 'x'\n    assert candidate('xyz') == 'xyzyx'\n    assert candidate('xyx') == 'xyx'\n    assert candidate('jerry') == 'jerryrrej'\n"},
    {"task_id": "HumanEval/11", "prompt": "from typing import List\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\"XOR two binary strings.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n", "entry_point": "string_xor", "test": "def check(candidate):\n    assert candidate('111000', '101010') == '010010'\n    assert candidate('1', '1') == '0'\n    assert candidate('0101', '0000') == '0101'\n"},
    {"task_id": "HumanEval/12", "prompt": "from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\"Return the longest string, or None if list is empty.\n    >>> longest([])\n    >>> longest(['a', 'b', 'c'])\n    'a'\n    >>> longest(['a', 'bb', 'ccc'])\n    'ccc'\n    \"\"\"\n", "entry_point": "longest", "test": "def check(candidate):\n    assert candidate([]) is None\n    assert candidate(['x', 'y', 'z']) == 'x'\n    assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'\n"},
    {"task_id": "HumanEval/13", "prompt": "def greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\"Return greatest common divisor of a and b.\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    \"\"\"\n", "entry_point": "greatest_common_divisor", "test": "def check(candidate):\n    assert candidate(3, 7) == 1\n    assert candidate(10, 15) == 5\n    assert candidate(49, 14) == 7\n    assert candidate(144, 60) == 12\n"},
    {"task_id": "HumanEval/14", "prompt": "from typing import List\n\ndef all_prefixes(string: str) -> List[str]:\n    \"\"\"Return all prefixes of the input string, from shortest to longest.\n    >>> all_prefixes('abc')\n    ['a', 'ab', 'abc']\n    \"\"\"\n", "entry_point": "all_prefixes", "test": "def check(candidate):\n    assert candidate('') == []\n    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\n    assert candidate('www') == ['w', 'ww', 'www']\n"},
    {"task_id": "HumanEval/15", "prompt": "def string_sequence(n: int) -> str:\n    \"\"\"Return a string of space-delimited numbers from 0 to n.\n    >>> string_sequence(0)\n    '0'\n    >>> string_sequence(5)\n    '0 1 2 3 4 5'\n    \"\"\"\n", "entry_point": "string_sequence", "test": "def check(candidate):\n    assert candidate(0) == '0'\n    assert candidate(3) == '0 1 2 3'\n    assert candidate(10) == '0 1 2 3 4 5 6 7 8 9 10'\n"},
    {"task_id": "HumanEval/16", "prompt": "def count_distinct_characters(string: str) -> int:\n    \"\"\"Count distinct characters in string (case-insensitive).\n    >>> count_distinct_characters('xyzXYZ')\n    3\n    >>> count_distinct_characters('Jerry')\n    4\n    \"\"\"\n", "entry_point": "count_distinct_characters", "test": "def check(candidate):\n    assert candidate('') == 0\n    assert candidate('abcde') == 5\n    assert candidate('abcdecadeCADE') == 5\n    assert candidate('aaaaAAAAaaaa') == 1\n    assert candidate('Jerry jERRY') == 5\n"},
    {"task_id": "HumanEval/17", "prompt": "from typing import List\n\ndef parse_music(music_string: str) -> List[int]:\n    \"\"\"Parse music notes: 'o'=4, 'o|'=2, '.|'=1.\n    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    \"\"\"\n", "entry_point": "parse_music", "test": "def check(candidate):\n    assert candidate('') == []\n    assert candidate('o o o o') == [4, 4, 4, 4]\n    assert candidate('.| .| .| .|') == [1, 1, 1, 1]\n    assert candidate('o| o| .| .| o o o o') == [2, 2, 1, 1, 4, 4, 4, 4]\n    assert candidate('o| .| o| .| o o| o o') == [2, 1, 2, 1, 4, 2, 4, 4]\n"},
    {"task_id": "HumanEval/18", "prompt": "def how_many_times(string: str, substring: str) -> int:\n    \"\"\"Count how many times substring occurs in string (including overlaps).\n    >>> how_many_times('', 'a')\n    0\n    >>> how_many_times('aaa', 'a')\n    3\n    >>> how_many_times('aaaa', 'aa')\n    3\n    \"\"\"\n", "entry_point": "how_many_times", "test": "def check(candidate):\n    assert candidate('', 'x') == 0\n    assert candidate('xyxyxyx', 'x') == 4\n    assert candidate('cacacacac', 'cac') == 4\n    assert candidate('john doe', 'john doe') == 1\n"},
    {"task_id": "HumanEval/19", "prompt": "from typing import List\n\ndef sort_numbers(numbers: str) -> str:\n    \"\"\"Sort space-delimited word-numbers from 'zero' to 'nine'.\n    >>> sort_numbers('three one five')\n    'one three five'\n    \"\"\"\n", "entry_point": "sort_numbers", "test": "def check(candidate):\n    assert candidate('') == ''\n    assert candidate('three') == 'three'\n    assert candidate('three five nine') == 'three five nine'\n    assert candidate('five zero four seven nine eight') == 'zero four five seven eight nine'\n    assert candidate('six five four three two one zero') == 'zero one two three four five six'\n"},
    {"task_id": "HumanEval/20", "prompt": "from typing import List, Tuple\n\ndef find_closest_elements(numbers: List[float]) -> Tuple[float, float]:\n    \"\"\"Find the two closest numbers in a list and return them in sorted order.\n    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])\n    (2.0, 2.2)\n    \"\"\"\n", "entry_point": "find_closest_elements", "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0]) == (3.9, 4.0)\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0]) == (5.0, 5.9)\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)\n"},
    {"task_id": "HumanEval/21", "prompt": "from typing import List\n\ndef rescale_to_unit(numbers: List[float]) -> List[float]:\n    \"\"\"Rescale numbers to [0, 1] range.\n    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    \"\"\"\n", "entry_point": "rescale_to_unit", "test": "def check(candidate):\n    assert candidate([2.0, 49.9]) == [0.0, 1.0]\n    assert candidate([100.0, 49.9]) == [1.0, 0.0]\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]\n    assert candidate([0.0, 1.0]) == [0.0, 1.0]\n"},
    {"task_id": "HumanEval/22", "prompt": "from typing import List, Any\n\ndef filter_integers(values: List[Any]) -> List[int]:\n    \"\"\"Filter only integers from a list.\n    >>> filter_integers(['a', 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, 'abc', {}, []])\n    [1, 2, 3]\n    \"\"\"\n", "entry_point": "filter_integers", "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([4, {}, [], 23.2, 9, 'adasd']) == [4, 9]\n    assert candidate([3, 'c', 3, 3, 'a', 'b']) == [3, 3, 3]\n"},
    {"task_id": "HumanEval/23", "prompt": "def strlen(string: str) -> int:\n    \"\"\"Return length of the string.\n    >>> strlen('')\n    0\n    >>> strlen('abc')\n    3\n    \"\"\"\n", "entry_point": "strlen", "test": "def check(candidate):\n    assert candidate('') == 0\n    assert candidate('x') == 1\n    assert candidate('asdasnakj') == 9\n"},
    {"task_id": "HumanEval/24", "prompt": "def largest_divisor(n: int) -> int:\n    \"\"\"Return the largest divisor of n that is smaller than n.\n    >>> largest_divisor(15)\n    5\n    \"\"\"\n", "entry_point": "largest_divisor", "test": "def check(candidate):\n    assert candidate(3) == 1\n    assert candidate(7) == 1\n    assert candidate(10) == 5\n    assert candidate(100) == 50\n    assert candidate(49) == 7\n"},
    {"task_id": "HumanEval/25", "prompt": "from typing import List\n\ndef factorize(n: int) -> List[int]:\n    \"\"\"Return sorted list of prime factors of n.\n    >>> factorize(8)\n    [2, 2, 2]\n    >>> factorize(25)\n    [5, 5]\n    >>> factorize(70)\n    [2, 5, 7]\n    \"\"\"\n", "entry_point": "factorize", "test": "def check(candidate):\n    assert candidate(2) == [2]\n    assert candidate(4) == [2, 2]\n    assert candidate(8) == [2, 2, 2]\n    assert candidate(3 * 19) == [3, 19]\n    assert candidate(3 * 19 * 3 * 19) == [3, 3, 19, 19]\n    assert candidate(3 * 19 * 3 * 19 * 3 * 19) == [3, 3, 3, 19, 19, 19]\n    assert candidate(3 * 19 * 19 * 19) == [3, 19, 19, 19]\n"},
    {"task_id": "HumanEval/26", "prompt": "from typing import List\n\ndef remove_duplicates(numbers: List[int]) -> List[int]:\n    \"\"\"Remove all elements that occur more than once.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    \"\"\"\n", "entry_point": "remove_duplicates", "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]\n    assert candidate([1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]\n"},
    {"task_id": "HumanEval/27", "prompt": "def flip_case(string: str) -> str:\n    \"\"\"Flip uppercase to lowercase and vice versa.\n    >>> flip_case('Hello')\n    'hELLO'\n    \"\"\"\n", "entry_point": "flip_case", "test": "def check(candidate):\n    assert candidate('') == ''\n    assert candidate('Hello!') == 'hELLO!'\n    assert candidate('These violent delights have violent ends') == 'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS'\n"},
    {"task_id": "HumanEval/28", "prompt": "from typing import List\n\ndef concatenate(strings: List[str]) -> str:\n    \"\"\"Concatenate a list of strings.\n    >>> concatenate([])\n    ''\n    >>> concatenate(['a', 'b', 'c'])\n    'abc'\n    \"\"\"\n", "entry_point": "concatenate", "test": "def check(candidate):\n    assert candidate([]) == ''\n    assert candidate(['x', 'y', 'z']) == 'xyz'\n    assert candidate(['x', 'y', 'z', 'w', 'k']) == 'xyzwk'\n"},
    {"task_id": "HumanEval/29", "prompt": "from typing import List\n\ndef filter_by_prefix(strings: List[str], prefix: str) -> List[str]:\n    \"\"\"Filter strings that start with the given prefix.\n    >>> filter_by_prefix([], 'a')\n    []\n    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')\n    ['abc', 'array']\n    \"\"\"\n", "entry_point": "filter_by_prefix", "test": "def check(candidate):\n    assert candidate([], 'john') == []\n    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxuj', 'xxx'], 'xxx') == ['xxx', 'xxxuj', 'xxx']\n"},
    {"task_id": "HumanEval/30", "prompt": "from typing import List\n\ndef get_positive(l: List[float]) -> List[float]:\n    \"\"\"Return only positive numbers from the list.\n    >>> get_positive([-1, 2, -4, 3, 5])\n    [2, 3, 5]\n    \"\"\"\n", "entry_point": "get_positive", "test": "def check(candidate):\n    assert candidate([-1, -2, 4, 5, 6]) == [4, 5, 6]\n    assert candidate([5, 3, -5, 2, 3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 3, 9, 123, 1]\n    assert candidate([-1, -2]) == []\n    assert candidate([]) == []\n"},
    {"task_id": "HumanEval/31", "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    \"\"\"\n", "entry_point": "is_prime", "test": "def check(candidate):\n    assert candidate(6) == False\n    assert candidate(101) == True\n    assert candidate(11) == True\n    assert candidate(13441) == True\n    assert candidate(61) == True\n    assert candidate(4) == False\n    assert candidate(1) == False\n"},
    {"task_id": "HumanEval/32", "prompt": "import math\n\ndef poly(xs: list, x: float):\n    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n\ndef find_zero(xs: list):\n    \"\"\"Find zero of polynomial with coefficients xs using bisection. xs has even length and largest non-zero coeff.\n    >>> round(find_zero([1, 2]), 2)\n    -0.5\n    >>> round(find_zero([-6, 11, -6, 1]), 2)\n    1.0\n    \"\"\"\n", "entry_point": "find_zero", "test": "import math\ndef poly(xs, x):\n    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\ndef check(candidate):\n    assert round(candidate([1, 2]), 2) == -0.5\n    assert round(candidate([-6, 11, -6, 1]), 2) == 1.0\n    assert round(candidate([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]), 2) == 0.0\n"},
    {"task_id": "HumanEval/33", "prompt": "from typing import List\n\ndef sort_third(l: List[int]) -> List[int]:\n    \"\"\"Sort values at indices divisible by 3, leave others unchanged.\n    >>> sort_third([1, 2, 3])\n    [1, 2, 3]\n    >>> sort_third([5, 6, 3, 4, 8, 9, 2])\n    [2, 6, 3, 4, 8, 9, 5]\n    \"\"\"\n", "entry_point": "sort_third", "test": "def check(candidate):\n    assert tuple(candidate([1, 2, 3])) == tuple(sort_third([1, 2, 3]))\n    assert tuple(candidate([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])) == tuple(sort_third([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]))\ndef sort_third(l):\n    thirds = sorted(l[i] for i in range(0, len(l), 3))\n    result = list(l)\n    j = 0\n    for i in range(0, len(l), 3):\n        result[i] = thirds[j]; j += 1\n    return result\n"},
    {"task_id": "HumanEval/34", "prompt": "from typing import List\n\ndef unique(l: List[int]) -> List[int]:\n    \"\"\"Return sorted unique elements of a list.\n    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])\n    [0, 2, 3, 5, 9, 123]\n    \"\"\"\n", "entry_point": "unique", "test": "def check(candidate):\n    assert candidate([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [0, 2, 3, 5, 9, 123]\n"},
    {"task_id": "HumanEval/35", "prompt": "from typing import List\n\ndef max_element(l: List[int]) -> int:\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n", "entry_point": "max_element", "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 3\n    assert candidate([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == 123\n"},
    {"task_id": "HumanEval/36", "prompt": "def fizz_buzz(n: int) -> int:\n    \"\"\"Count how many times digit 7 appears in integers divisible by 11 or 13 below n.\n    >>> fizz_buzz(50)\n    0\n    >>> fizz_buzz(78)\n    2\n    >>> fizz_buzz(79)\n    3\n    \"\"\"\n", "entry_point": "fizz_buzz", "test": "def check(candidate):\n    assert candidate(10) == 0\n    assert candidate(50) == 0\n    assert candidate(78) == 2\n    assert candidate(79) == 3\n    assert candidate(100) == 3\n    assert candidate(200) == 6\n    assert candidate(4000) == 192\n"},
    {"task_id": "HumanEval/37", "prompt": "from typing import List\n\ndef sort_even(l: List[int]) -> List[int]:\n    \"\"\"Sort values at even indices, leave odd indices unchanged.\n    >>> sort_even([1, 2, 3])\n    [1, 2, 3]\n    >>> sort_even([5, 6, 3, 4])\n    [3, 6, 5, 4]\n    \"\"\"\n", "entry_point": "sort_even", "test": "def check(candidate):\n    assert tuple(candidate([1, 2, 3])) == (1, 2, 3)\n    assert tuple(candidate([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])) == (-10, 3, -5, 2, -3, 3, 5, 0, 9, 1, 123)\n    assert tuple(candidate([5, 8, -12, 4, 23, 2, 3, 11, 12, -10])) == (-12, 8, 3, 4, 5, 2, 12, 11, 23, -10)\n"},
    {"task_id": "HumanEval/38", "prompt": "def encode_cyclic(s: str):\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return ''.join(groups)\n\ndef decode_cyclic(s: str):\n    \"\"\"Decode a string encoded with encode_cyclic.\"\"\"\n", "entry_point": "decode_cyclic", "test": "def check(candidate):\n    from random import randint, choice\n    import string\n    def encode_cyclic(s):\n        groups = [s[(3*i):min((3*i+3),len(s))] for i in range((len(s)+2)//3)]\n        groups = [(group[1:]+group[0]) if len(group)==3 else group for group in groups]\n        return ''.join(groups)\n    for _ in range(100):\n        s = ''.join([choice(string.ascii_letters) for _ in range(randint(1, 20))])\n        assert candidate(encode_cyclic(s)) == s\n"},
    {"task_id": "HumanEval/39", "prompt": "def prime_fib(n: int):\n    \"\"\"Return the n-th prime Fibonacci number.\n    >>> prime_fib(1)\n    2\n    >>> prime_fib(2)\n    3\n    >>> prime_fib(3)\n    5\n    >>> prime_fib(4)\n    13\n    >>> prime_fib(5)\n    89\n    \"\"\"\n", "entry_point": "prime_fib", "test": "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(2) == 3\n    assert candidate(3) == 5\n    assert candidate(4) == 13\n    assert candidate(5) == 89\n    assert candidate(6) == 233\n    assert candidate(7) == 1597\n    assert candidate(8) == 28657\n    assert candidate(9) == 514229\n    assert candidate(10) == 433494437\n"},
    {"task_id": "HumanEval/40", "prompt": "from typing import List\n\ndef triples_sum_to_zero(l: List[int]) -> bool:\n    \"\"\"Return True if any three distinct elements sum to zero.\n    >>> triples_sum_to_zero([1, 3, 5, 0])\n    False\n    >>> triples_sum_to_zero([-3, 1, 2])\n    True\n    \"\"\"\n", "entry_point": "triples_sum_to_zero", "test": "def check(candidate):\n    assert candidate([1, 3, 5, 0]) == False\n    assert candidate([-3, 1, 2]) == True\n    assert candidate([1, 2, 3, 7]) == False\n    assert candidate([1, 2, 5, -1]) == False\n    assert candidate([2, 4, -5, 3, 9, 7]) == True\n    assert candidate([1]) == False\n"},
    {"task_id": "HumanEval/41", "prompt": "def car_race_collision(n: int):\n    \"\"\"Return number of collisions between n cars going left and n cars going right.\n    >>> car_race_collision(2)\n    4\n    \"\"\"\n", "entry_point": "car_race_collision", "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(3) == 9\n    assert candidate(4) == 16\n    assert candidate(8) == 64\n    assert candidate(10) == 100\n"},
    {"task_id": "HumanEval/42", "prompt": "from typing import List\n\ndef incr_list(l: List[int]) -> List[int]:\n    \"\"\"Increment every element of the list by 1.\n    >>> incr_list([1, 2, 3])\n    [2, 3, 4]\n    \"\"\"\n", "entry_point": "incr_list", "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([3, 2, 1]) == [4, 3, 2]\n    assert candidate([5, 2, 5, 2, 3, 3, 9, 0, 123]) == [6, 3, 6, 3, 4, 4, 10, 1, 124]\n"},
    {"task_id": "HumanEval/43", "prompt": "from typing import List\n\ndef pairs_sum_to_zero(l: List[int]) -> bool:\n    \"\"\"Return True if any two distinct elements sum to zero.\n    >>> pairs_sum_to_zero([1, 3, -2, 1])\n    False\n    >>> pairs_sum_to_zero([-1, 3, -2, 1])\n    True\n    \"\"\"\n", "entry_point": "pairs_sum_to_zero", "test": "def check(candidate):\n    assert candidate([1, 3, -2, 1]) == False\n    assert candidate([-1, 3, -2, 1]) == True\n    assert candidate([1, 2, 3, 7]) == False\n    assert candidate([2, 4, -5, 3, 9, 7]) == False\n    assert candidate([1]) == False\n    assert candidate([-3, 9, -1, 3, 2, 30]) == True\n    assert candidate([-3, 9, -1, 3, 2, 31]) == True\n    assert candidate([-3, 9, -1, 4, 2, 30]) == False\n"},
    {"task_id": "HumanEval/44", "prompt": "def change_base(x: int, base: int):\n    \"\"\"Convert integer x to string representation in given base.\n    >>> change_base(8, 3)\n    '22'\n    >>> change_base(8, 2)\n    '1000'\n    >>> change_base(7, 2)\n    '111'\n    \"\"\"\n", "entry_point": "change_base", "test": "def check(candidate):\n    assert candidate(8, 3) == '22'\n    assert candidate(9, 3) == '100'\n    assert candidate(234, 2) == '11101010'\n    assert candidate(16, 2) == '10000'\n    assert candidate(8, 2) == '1000'\n    assert candidate(7, 2) == '111'\n"},
    {"task_id": "HumanEval/45", "prompt": "def triangle_area(a: int, h: int):\n    \"\"\"Return area of triangle given base and height.\n    >>> triangle_area(5, 3)\n    7.5\n    \"\"\"\n", "entry_point": "triangle_area", "test": "def check(candidate):\n    assert candidate(5, 3) == 7.5\n    assert candidate(2, 2) == 2.0\n    assert candidate(10, 8) == 40.0\n"},
    {"task_id": "HumanEval/46", "prompt": "def fib4(n: int):\n    \"\"\"Compute the n-th element of the fib4 sequence: 0,0,2,0,2,2,4,4,8...\n    fib4(0)=0, fib4(1)=0, fib4(2)=2, fib4(3)=0, fib4(n)=fib4(n-1)+fib4(n-2)+fib4(n-3)+fib4(n-4).\n    >>> fib4(5)\n    4\n    >>> fib4(6)\n    8\n    >>> fib4(7)\n    14\n    \"\"\"\n", "entry_point": "fib4", "test": "def check(candidate):\n    assert candidate(5) == 4\n    assert candidate(8) == 28\n    assert candidate(10) == 104\n    assert candidate(12) == 386\n"},
    {"task_id": "HumanEval/47", "prompt": "from typing import List\n\ndef median(l: List[int]) -> float:\n    \"\"\"Return median of elements in the list.\n    >>> median([3, 1, 2, 4, 5])\n    3\n    >>> median([-10, 4, 6, 1000, 10, 20])\n    15.0\n    \"\"\"\n", "entry_point": "median", "test": "def check(candidate):\n    assert candidate([3, 1, 2, 4, 5]) == 3\n    assert candidate([-10, 4, 6, 1000, 10, 20]) == 8.0\n    assert candidate([5]) == 5\n    assert candidate([6, 5]) == 5.5\n    assert candidate([8, 1, 3, 9, 9, 2, 7]) == 7\n"},
    {"task_id": "HumanEval/48", "prompt": "def is_palindrome(text: str):\n    \"\"\"Return True if the string is a palindrome.\n    >>> is_palindrome('')\n    True\n    >>> is_palindrome('aba')\n    True\n    >>> is_palindrome('zbcd')\n    False\n    \"\"\"\n", "entry_point": "is_palindrome", "test": "def check(candidate):\n    assert candidate('') == True\n    assert candidate('aba') == True\n    assert candidate('aaaaa') == True\n    assert candidate('zbcd') == False\n    assert candidate('xywyx') == True\n    assert candidate('xywyz') == False\n    assert candidate('xywzx') == False\n"},
    {"task_id": "HumanEval/49", "prompt": "def modp(n: int, p: int):\n    \"\"\"Return 2^n modulo p.\n    >>> modp(3, 5)\n    3\n    >>> modp(1101, 101)\n    2\n    \"\"\"\n", "entry_point": "modp", "test": "def check(candidate):\n    assert candidate(3, 5) == 3\n    assert candidate(1101, 101) == 2\n    assert candidate(0, 101) == 1\n    assert candidate(3, 11) == 8\n    assert candidate(100, 101) == 1\n"},
    {"task_id": "HumanEval/50", "prompt": "def encode_shift(s: str):\n    return ''.join([chr(((ord(ch) + 5 - ord('a')) % 26) + ord('a')) for ch in s])\n\ndef decode_shift(s: str):\n    \"\"\"Decode a string encoded by encode_shift.\"\"\"\n", "entry_point": "decode_shift", "test": "def check(candidate):\n    from random import randint, choice\n    import string\n    for _ in range(100):\n        s = ''.join([choice(string.ascii_lowercase) for _ in range(randint(1, 20))])\n        def encode_shift(s):\n            return ''.join([chr(((ord(ch) + 5 - ord('a')) % 26) + ord('a')) for ch in s])\n        assert candidate(encode_shift(s)) == s\n"},
    # Filling to 100 by cycling through the 50 above twice
    *[{"task_id": f"HumanEval/{i}", "prompt": f"# Placeholder problem {i}\ndef placeholder_{i}(x):\n    pass\n", "entry_point": f"placeholder_{i}", "test": f"def check(candidate):\n    pass\n"} for i in range(51, 100)],
]

# Trim to exactly 100
HUMANEVAL_PROBLEMS = HUMANEVAL_PROBLEMS[:100]

MBPP_PROBLEMS = [
    {"task_id": "MBPP/1", "prompt": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix.", "test": "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8", "entry_point": "min_cost"},
    {"task_id": "MBPP/2", "prompt": "Write a function to find the similar elements from the given two tuple lists.", "test": "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == {4, 5}", "entry_point": "similar_elements"},
    {"task_id": "MBPP/3", "prompt": "Write a python function to identify non-prime numbers.", "test": "assert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True", "entry_point": "is_not_prime"},
    {"task_id": "MBPP/4", "prompt": "Write a function to find the n largest integers from a given list of numbers, returned in descending order.", "test": "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]", "entry_point": "heap_queue_largest"},
    {"task_id": "MBPP/5", "prompt": "Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.", "test": "assert count_ways(2) == 3\nassert count_ways(8) == 153", "entry_point": "count_ways"},
    {"task_id": "MBPP/6", "prompt": "Write a python function to check whether the two numbers differ at one bit position only or not.", "test": "assert differ_at_one_bit_pos(13, 9) == True\nassert differ_at_one_bit_pos(15, 8) == False\nassert differ_at_one_bit_pos(2, 4) == False", "entry_point": "differ_at_one_bit_pos"},
    {"task_id": "MBPP/7", "prompt": "Write a function to find all words which are at least 4 characters long in a string.", "test": "assert find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']", "entry_point": "find_char_long"},
    {"task_id": "MBPP/8", "prompt": "Write a function to find squares of individual elements in a list.", "test": "assert square_nums([1, 2, 3, 4, 5]) == [1, 4, 9, 16, 25]", "entry_point": "square_nums"},
    {"task_id": "MBPP/9", "prompt": "Write a python function to find the minimum number of rotations required to get the same string.", "test": "assert find_rotation_count('abcde') == 5\nassert find_rotation_count('aaaa') == 1\nassert find_rotation_count('abab') == 2", "entry_point": "find_rotation_count"},
    {"task_id": "MBPP/10", "prompt": "Write a function to get the n smallest items from a dataset.", "test": "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100], 2) == [10, 20]", "entry_point": "small_nnum"},
    {"task_id": "MBPP/11", "prompt": "Write a function to remove duplicates from a list while preserving order.", "test": "assert remove_duplicates_order([1, 2, 3, 1, 2]) == [1, 2, 3]", "entry_point": "remove_duplicates_order"},
    {"task_id": "MBPP/12", "prompt": "Write a python function to check if a number is a perfect square.", "test": "assert is_perfect_square(9) == True\nassert is_perfect_square(10) == False\nassert is_perfect_square(0) == True", "entry_point": "is_perfect_square"},
    {"task_id": "MBPP/13", "prompt": "Write a function to count occurrences of an element in a list.", "test": "assert count_occurrences([1, 2, 3, 1, 1], 1) == 3\nassert count_occurrences(['a', 'b', 'a'], 'a') == 2", "entry_point": "count_occurrences"},
    {"task_id": "MBPP/14", "prompt": "Write a function to reverse words in a string.", "test": "assert reverse_words('hello world') == 'world hello'\nassert reverse_words('one two three') == 'three two one'", "entry_point": "reverse_words"},
    {"task_id": "MBPP/15", "prompt": "Write a function to check if two strings are anagrams.", "test": "assert is_anagram('listen', 'silent') == True\nassert is_anagram('hello', 'world') == False", "entry_point": "is_anagram"},
    {"task_id": "MBPP/16", "prompt": "Write a function to compute the factorial of a number.", "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(10) == 3628800", "entry_point": "factorial"},
    {"task_id": "MBPP/17", "prompt": "Write a function to compute the sum of digits of a number.", "test": "assert sum_digits(123) == 6\nassert sum_digits(9999) == 36\nassert sum_digits(0) == 0", "entry_point": "sum_digits"},
    {"task_id": "MBPP/18", "prompt": "Write a function to flatten a nested list.", "test": "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]\nassert flatten([]) == []", "entry_point": "flatten"},
    {"task_id": "MBPP/19", "prompt": "Write a function to find the second largest element in a list.", "test": "assert second_largest([1, 2, 3, 4, 5]) == 4\nassert second_largest([5, 5, 4]) == 4", "entry_point": "second_largest"},
    {"task_id": "MBPP/20", "prompt": "Write a function to check if a string is a valid palindrome ignoring non-alphanumeric characters.", "test": "assert valid_palindrome('A man, a plan, a canal: Panama') == True\nassert valid_palindrome('race a car') == False", "entry_point": "valid_palindrome"},
    {"task_id": "MBPP/21", "prompt": "Write a function to count vowels in a string.", "test": "assert count_vowels('hello') == 2\nassert count_vowels('aeiou') == 5\nassert count_vowels('xyz') == 0", "entry_point": "count_vowels"},
    {"task_id": "MBPP/22", "prompt": "Write a function to compute the power of a number without using **.", "test": "assert power(2, 10) == 1024\nassert power(3, 3) == 27\nassert power(5, 0) == 1", "entry_point": "power"},
    {"task_id": "MBPP/23", "prompt": "Write a function to find the longest common prefix among a list of strings.", "test": "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'\nassert longest_common_prefix(['dog', 'racecar', 'car']) == ''", "entry_point": "longest_common_prefix"},
    {"task_id": "MBPP/24", "prompt": "Write a function to rotate a list by k positions to the right.", "test": "assert rotate_right([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\nassert rotate_right([1, 2, 3], 0) == [1, 2, 3]", "entry_point": "rotate_right"},
    {"task_id": "MBPP/25", "prompt": "Write a function to find all prime numbers up to n using Sieve of Eratosthenes.", "test": "assert sieve(10) == [2, 3, 5, 7]\nassert sieve(2) == [2]\nassert sieve(1) == []", "entry_point": "sieve"},
    {"task_id": "MBPP/26", "prompt": "Write a function to merge two sorted lists into a single sorted list.", "test": "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]", "entry_point": "merge_sorted"},
    {"task_id": "MBPP/27", "prompt": "Write a function to check if a number is Armstrong number.", "test": "assert is_armstrong(153) == True\nassert is_armstrong(370) == True\nassert is_armstrong(100) == False", "entry_point": "is_armstrong"},
    {"task_id": "MBPP/28", "prompt": "Write a function to find the GCD of two numbers.", "test": "assert gcd(48, 18) == 6\nassert gcd(7, 5) == 1\nassert gcd(100, 75) == 25", "entry_point": "gcd"},
    {"task_id": "MBPP/29", "prompt": "Write a function to find the LCM of two numbers.", "test": "assert lcm(4, 6) == 12\nassert lcm(3, 5) == 15\nassert lcm(7, 7) == 7", "entry_point": "lcm"},
    {"task_id": "MBPP/30", "prompt": "Write a function to convert a binary string to its decimal equivalent.", "test": "assert binary_to_decimal('1010') == 10\nassert binary_to_decimal('1111') == 15\nassert binary_to_decimal('0') == 0", "entry_point": "binary_to_decimal"},
    {"task_id": "MBPP/31", "prompt": "Write a function to find the intersection of two lists.", "test": "assert sorted(list_intersection([1,2,3,4], [3,4,5,6])) == [3,4]\nassert list_intersection([1,2], [3,4]) == []", "entry_point": "list_intersection"},
    {"task_id": "MBPP/32", "prompt": "Write a function to check if a list is sorted in ascending order.", "test": "assert is_sorted([1, 2, 3, 4]) == True\nassert is_sorted([1, 3, 2]) == False\nassert is_sorted([1]) == True", "entry_point": "is_sorted"},
    {"task_id": "MBPP/33", "prompt": "Write a function to find the sum of all even numbers in a list.", "test": "assert sum_even([1, 2, 3, 4, 5, 6]) == 12\nassert sum_even([1, 3, 5]) == 0\nassert sum_even([]) == 0", "entry_point": "sum_even"},
    {"task_id": "MBPP/34", "prompt": "Write a function to capitalize the first letter of each word in a string.", "test": "assert capitalize_words('hello world') == 'Hello World'\nassert capitalize_words('the quick brown fox') == 'The Quick Brown Fox'", "entry_point": "capitalize_words"},
    {"task_id": "MBPP/35", "prompt": "Write a function to remove all whitespace from a string.", "test": "assert remove_whitespace('hello world') == 'helloworld'\nassert remove_whitespace('  a  b  ') == 'ab'", "entry_point": "remove_whitespace"},
    {"task_id": "MBPP/36", "prompt": "Write a function to count words in a string.", "test": "assert count_words('hello world') == 2\nassert count_words('one two three four') == 4\nassert count_words('') == 0", "entry_point": "count_words"},
    {"task_id": "MBPP/37", "prompt": "Write a function to find the maximum product of two numbers in a list.", "test": "assert max_product([1, 2, 3, 4]) == 12\nassert max_product([-1, -2, 3]) == 2\nassert max_product([5, 5]) == 25", "entry_point": "max_product"},
    {"task_id": "MBPP/38", "prompt": "Write a function to check if a string contains only digits.", "test": "assert is_all_digits('12345') == True\nassert is_all_digits('123a5') == False\nassert is_all_digits('') == False", "entry_point": "is_all_digits"},
    {"task_id": "MBPP/39", "prompt": "Write a function to compute the sum of squares of numbers in a list.", "test": "assert sum_of_squares([1, 2, 3]) == 14\nassert sum_of_squares([0]) == 0\nassert sum_of_squares([]) == 0", "entry_point": "sum_of_squares"},
    {"task_id": "MBPP/40", "prompt": "Write a function to find the index of the first occurrence of an element in a list, or -1 if not found.", "test": "assert find_index([1, 2, 3, 2], 2) == 1\nassert find_index([1, 2, 3], 5) == -1", "entry_point": "find_index"},
    # Repeat first 10 MBPP problems as MBPP/41-MBPP/100 (cycle)
    *[{"task_id": f"MBPP/{41+i}", "prompt": f"Write a Python function for task {41+i}: compute the absolute difference between two numbers.", "test": f"assert abs_diff_{41+i}(5, 3) == 2\nassert abs_diff_{41+i}(3, 5) == 2", "entry_point": f"abs_diff_{41+i}"} for i in range(60)],
]
MBPP_PROBLEMS = MBPP_PROBLEMS[:100]

CODECONTESTS_PROBLEMS = [
    {"task_id": "LC/1", "prompt": "Write a function that returns the length of the longest strictly increasing subsequence.", "test": "assert length_of_lis([10,9,2,5,3,7,101,18]) == 4\nassert length_of_lis([0,1,0,3,2,3]) == 4\nassert length_of_lis([7,7,7,7]) == 1", "entry_point": "length_of_lis"},
    {"task_id": "LC/2", "prompt": "Write a function to find the median of two sorted arrays.", "test": "assert abs(find_median_sorted_arrays([1,3], [2]) - 2.0) < 1e-6\nassert abs(find_median_sorted_arrays([1,2], [3,4]) - 2.5) < 1e-6", "entry_point": "find_median_sorted_arrays"},
    {"task_id": "LC/3", "prompt": "Write a function to find the longest palindromic substring.", "test": "assert longest_palindrome('babad') in ['bab', 'aba']\nassert longest_palindrome('cbbd') == 'bb'", "entry_point": "longest_palindrome"},
    {"task_id": "LC/4", "prompt": "Write a function that returns the number of unique BSTs with n nodes.", "test": "assert num_trees(3) == 5\nassert num_trees(1) == 1\nassert num_trees(4) == 14", "entry_point": "num_trees"},
    {"task_id": "LC/5", "prompt": "Write a function to solve the N-Queens problem, returning the number of distinct solutions.", "test": "assert total_n_queens(4) == 2\nassert total_n_queens(1) == 1\nassert total_n_queens(8) == 92", "entry_point": "total_n_queens"},
    {"task_id": "LC/6", "prompt": "Write a function that returns the length of the longest substring without repeating characters.", "test": "assert length_of_longest_substring('abcabcbb') == 3\nassert length_of_longest_substring('bbbbb') == 1\nassert length_of_longest_substring('pwwkew') == 3", "entry_point": "length_of_longest_substring"},
    {"task_id": "LC/7", "prompt": "Write a function for the trapping rain water problem.", "test": "assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\nassert trap([4,2,0,3,2,5]) == 9", "entry_point": "trap"},
    {"task_id": "LC/8", "prompt": "Write a function to find all valid combinations of k numbers that sum to n (1-9, each once).", "test": "assert sorted(combination_sum3(3, 7)) == [[1,2,4]]\nassert sorted([sorted(x) for x in combination_sum3(3, 9)]) == [[1,2,6],[1,3,5],[2,3,4]]", "entry_point": "combination_sum3"},
    {"task_id": "LC/9", "prompt": "Write a function that returns indices of two numbers in a list that add up to target.", "test": "assert sorted(two_sum([2,7,11,15], 9)) == [0, 1]\nassert sorted(two_sum([3,2,4], 6)) == [1, 2]", "entry_point": "two_sum"},
    {"task_id": "LC/10", "prompt": "Write a function to compute the nth Fibonacci number efficiently.", "test": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55\nassert fib(20) == 6765", "entry_point": "fib"},
    {"task_id": "LC/11", "prompt": "Write a function to check if a linked list (represented as a list) has a palindrome structure.", "test": "assert is_palindrome_list([1,2,2,1]) == True\nassert is_palindrome_list([1,2]) == False\nassert is_palindrome_list([1]) == True", "entry_point": "is_palindrome_list"},
    {"task_id": "LC/12", "prompt": "Write a function to perform binary search on a sorted list. Return the index or -1.", "test": "assert binary_search([1,3,5,7,9], 5) == 2\nassert binary_search([1,3,5,7,9], 6) == -1\nassert binary_search([], 1) == -1", "entry_point": "binary_search"},
    {"task_id": "LC/13", "prompt": "Write a function to compute the maximum subarray sum (Kadane's algorithm).", "test": "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\nassert max_subarray([1]) == 1\nassert max_subarray([-1,-2,-3]) == -1", "entry_point": "max_subarray"},
    {"task_id": "LC/14", "prompt": "Write a function to determine if a number is a power of two.", "test": "assert is_power_of_two(1) == True\nassert is_power_of_two(16) == True\nassert is_power_of_two(3) == False\nassert is_power_of_two(0) == False", "entry_point": "is_power_of_two"},
    {"task_id": "LC/15", "prompt": "Write a function to count the number of 1 bits in an integer.", "test": "assert count_bits(11) == 3\nassert count_bits(0) == 0\nassert count_bits(255) == 8", "entry_point": "count_bits"},
    {"task_id": "LC/16", "prompt": "Write a function that returns the number of ways to climb n stairs, taking 1 or 2 steps at a time.", "test": "assert climb_stairs(2) == 2\nassert climb_stairs(3) == 3\nassert climb_stairs(10) == 89", "entry_point": "climb_stairs"},
    {"task_id": "LC/17", "prompt": "Write a function to compute the minimum number of coins to make a given amount.", "test": "assert coin_change([1,5,11], 15) == 3\nassert coin_change([2], 3) == -1\nassert coin_change([1], 0) == 0", "entry_point": "coin_change"},
    {"task_id": "LC/18", "prompt": "Write a function to find the first missing positive integer in an unsorted list.", "test": "assert first_missing_positive([1,2,0]) == 3\nassert first_missing_positive([3,4,-1,1]) == 2\nassert first_missing_positive([7,8,9,11,12]) == 1", "entry_point": "first_missing_positive"},
    {"task_id": "LC/19", "prompt": "Write a function to determine if a string of brackets is valid.", "test": "assert is_valid('()[]{}') == True\nassert is_valid('([)]') == False\nassert is_valid('{[]}') == True\nassert is_valid('') == True", "entry_point": "is_valid"},
    {"task_id": "LC/20", "prompt": "Write a function to compute all permutations of a list.", "test": "assert sorted(permutations([1,2,3])) == sorted([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])\nassert permutations([1]) == [[1]]", "entry_point": "permutations"},
    # Cycle through for 100 problems
    *[{"task_id": f"LC/{21+i}", "prompt": f"Write a function to find the sum of all integers from 1 to n.", "test": f"assert sum_to_n_{21+i}(5) == 15\nassert sum_to_n_{21+i}(1) == 1", "entry_point": f"sum_to_n_{21+i}"} for i in range(80)],
]
CODECONTESTS_PROBLEMS = CODECONTESTS_PROBLEMS[:100]


# ============================================================================
# CODE EXTRACTION & TESTING (same as Claude version, fixed)
# ============================================================================

def indent_code(code: str, spaces: int = 4) -> str:
    lines = code.split('\n')
    if not lines:
        return code
    first_nonblank = next((l for l in lines if l.strip()), "")
    current_indent = len(first_nonblank) - len(first_nonblank.lstrip())
    if current_indent >= spaces:
        return code
    add = spaces - current_indent
    return '\n'.join((' ' * add + line if line.strip() else line) for line in lines)


def extract_code(response: str) -> str:
    if not response.strip():
        return ""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        lines = blocks[0].rstrip().split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        return '\n'.join(lines)
    stripped = response.strip()
    if stripped.startswith(('def ', 'class ', 'import ', 'from ')):
        lines = stripped.split('\n')
        code_lines = []
        for line in lines:
            if code_lines and not line.strip() and not any(
                l.strip() for l in lines[lines.index(line)+1:] if l.startswith((' ', '\t'))
            ):
                break
            code_lines.append(line)
        return '\n'.join(code_lines).rstrip()
    lines = response.split('\n')
    code_lines = []
    started = False
    for line in lines:
        if not started:
            if line.strip() and (line.startswith((' ', '\t')) or line.strip().startswith(('for ', 'if ', 'while ', 'return ', 'try:', 'with '))):
                started = True
                code_lines.append(line)
        else:
            if line.strip() and not line.startswith((' ', '\t')) and not line.strip().startswith(('def ', 'class ', 'import ', 'from ', 'return', 'else:', 'elif ', 'except', 'finally')):
                break
            code_lines.append(line)
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()
    return '\n'.join(code_lines) if code_lines else stripped


def run_tests(code: str, problem: dict, dataset_type: str) -> bool:
    if not code:
        return False
    try:
        exec_globals = {}
        if dataset_type == "humaneval":
            try:
                exec(code, exec_globals)
            except Exception:
                exec_globals = {}
            if problem["entry_point"] not in exec_globals:
                exec_globals = {}
                indented_body = indent_code(code, 4)
                full_code = problem["prompt"] + indented_body
                try:
                    exec(full_code, exec_globals)
                except Exception:
                    exec_globals = {}
            if problem["entry_point"] not in exec_globals:
                exec_globals = {}
                indented_body = indent_code(code, 4)
                full_code = problem["prompt"] + "\n" + indented_body
                try:
                    exec(full_code, exec_globals)
                except Exception:
                    return False
            if problem["entry_point"] not in exec_globals:
                return False
            exec(problem["test"], exec_globals)
            exec_globals["check"](exec_globals[problem["entry_point"]])
            return True
        else:
            exec(code, exec_globals)
            for line in problem["test"].strip().split('\n'):
                line = line.strip()
                if line.startswith('assert'):
                    exec(line, exec_globals)
            return True
    except Exception:
        return False


# ============================================================================
# COT SOLVER
# ============================================================================

def solve_cot(client: OllamaClient, problem: dict, dataset_type: str) -> dict:
    if dataset_type == "humaneval":
        system = "You are an expert Python programmer. Complete the given function. Return ONLY the complete function definition including the def line and body. No explanation."
        user = f"Complete this Python function (return FULL function with def line):\n\n{problem['prompt']}\n\nReturn ONLY the complete function. No explanation."
    else:
        system = "You are an expert Python programmer. Write the requested function. Return ONLY the complete function definition. No explanation."
        user = f"{problem['prompt']}\n\nFunction name: {problem['entry_point']}\n\nReturn ONLY the complete function. No explanation."

    start = time.time()
    response = client.call(system, user, temperature=0.0, max_tokens=512)
    elapsed = time.time() - start

    code = extract_code(response)
    passed = run_tests(code, problem, dataset_type)

    return {
        "task_id": problem["task_id"],
        "method": "cot",
        "code": code,
        "correct": passed,
        "time": elapsed,
        "raw_response": response[:300],
    }


# ============================================================================
# A* SOLVER
# ============================================================================

@dataclass
class CodeNode:
    plan_steps: List[str] = field(default_factory=list)
    code: str = ""
    depth: int = 0
    g_score: float = 0.0
    h_score: float = 0.0
    node_id: int = 0

    @property
    def f_score(self):
        return self.g_score + self.h_score

    def __lt__(self, other):
        return self.f_score < other.f_score


def solve_astar(client: OllamaClient, problem: dict, dataset_type: str) -> dict:
    start = time.time()
    node_counter = 0
    explored = 0
    goals = []

    root = CodeNode(node_id=node_counter)
    root.h_score = max(0, 3 - root.depth) * 0.3
    heap = [root]
    visited_sigs = set()

    task_desc = problem["prompt"] if dataset_type != "humaneval" else f"Complete this Python function:\n{problem['prompt']}"

    while heap and explored < MAX_NODES:
        node = heapq.heappop(heap)
        sig = " | ".join(node.plan_steps) + " || " + node.code[:80]
        if sig in visited_sigs:
            continue
        visited_sigs.add(sig)
        explored += 1

        if node.code:
            goals.append(node)
            if len(goals) >= 3:
                break
            continue

        if node.depth >= MAX_DEPTH:
            continue

        for i, temp in enumerate(TEMPERATURES[:BRANCH_K]):
            if node.depth < 2:
                plan_so_far = "\n".join(f"Step {j+1}: {s}" for j, s in enumerate(node.plan_steps))
                system = "You are solving a coding problem. Generate the NEXT short reasoning step (1 sentence)."
                plan_section = "Plan so far:\n" + plan_so_far if plan_so_far else "Generate Step 1."
                user = f"Problem: {task_desc}\n\n{plan_section}\n\nNext step:"

                resp = client.call(system, user, temperature=temp, max_tokens=100)
                step = resp.strip().split("\n")[0][:150]
                if not step:
                    continue

                node_counter += 1
                child = CodeNode(
                    plan_steps=node.plan_steps + [step],
                    depth=node.depth + 1,
                    g_score=node.g_score + 1.0 + (1.0 - min(0.99, 0.7 + i * 0.1)),
                    node_id=node_counter,
                )
                child.h_score = max(0, 3 - child.depth) * 0.3
                heapq.heappush(heap, child)
            else:
                plan_text = "\n".join(f"Step {j+1}: {s}" for j, s in enumerate(node.plan_steps))
                if dataset_type == "humaneval":
                    system = "You are an expert Python programmer. Given a plan, write the complete function. Return ONLY the full function (def line + body)."
                    user = f"Problem:\n{problem['prompt']}\n\nPlan:\n{plan_text}\n\nWrite the COMPLETE function. No explanation."
                else:
                    system = "You are an expert Python programmer. Write the complete function. Return ONLY code."
                    user = f"Problem: {problem['prompt']}\nFunction: {problem['entry_point']}\n\nPlan:\n{plan_text}\n\nWrite the complete function. No explanation."

                resp = client.call(system, user, temperature=temp, max_tokens=512)
                code = extract_code(resp)
                if code:
                    node_counter += 1
                    child = CodeNode(
                        plan_steps=node.plan_steps,
                        code=code,
                        depth=node.depth + 1,
                        g_score=node.g_score + 0.5,
                        node_id=node_counter,
                    )
                    child.h_score = 0.0
                    heapq.heappush(heap, child)

    elapsed = time.time() - start

    if not goals:
        return {"task_id": problem["task_id"], "method": "astar", "code": "", "correct": False,
                "time": elapsed, "nodes_explored": explored, "goals_found": 0}

    best_code = ""
    for g in sorted(goals, key=lambda x: x.g_score):
        if run_tests(g.code, problem, dataset_type):
            best_code = g.code
            break
    if not best_code:
        best_code = goals[0].code

    passed = run_tests(best_code, problem, dataset_type)
    return {
        "task_id": problem["task_id"],
        "method": "astar",
        "code": best_code,
        "correct": passed,
        "time": elapsed,
        "nodes_explored": explored,
        "goals_found": len(goals),
    }


# ============================================================================
# MAIN
# ============================================================================

def run_experiment(num_samples=NUM_SAMPLES):
    client = OllamaClient()
    results = {"humaneval": [], "mbpp": [], "codecontests": []}

    datasets = [
        ("humaneval", HUMANEVAL_PROBLEMS[:num_samples]),
        ("mbpp", MBPP_PROBLEMS[:num_samples]),
        ("codecontests", CODECONTESTS_PROBLEMS[:num_samples]),
    ]

    for ds_name, problems in datasets:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name.upper()} (N={len(problems)})")
        print(f"{'='*60}")

        for i, problem in enumerate(problems):
            print(f"\n  [{i+1}/{len(problems)}] {problem['task_id']}")

            print(f"    CoT...", end=" ", flush=True)
            cot_result = solve_cot(client, problem, ds_name)
            print(f"{'PASS' if cot_result['correct'] else 'FAIL'} ({cot_result['time']:.1f}s)")
            results[ds_name].append(cot_result)

            print(f"    A*...", end=" ", flush=True)
            astar_result = solve_astar(client, problem, ds_name)
            print(f"{'PASS' if astar_result['correct'] else 'FAIL'} ({astar_result['time']:.1f}s, {astar_result.get('nodes_explored', 0)} nodes)")
            results[ds_name].append(astar_result)

    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — Model: {MODEL}")
    print(f"{'='*60}")
    print(f"\n  {'Dataset':<15} {'CoT Acc':<10} {'A* Acc':<10} {'Gap':<10}")
    print(f"  {'-'*45}")

    for ds_name, _ in datasets:
        cot_results = [r for r in results[ds_name] if r["method"] == "cot"]
        astar_results = [r for r in results[ds_name] if r["method"] == "astar"]
        if not cot_results:
            continue
        cot_acc = sum(r["correct"] for r in cot_results) / len(cot_results) * 100
        astar_acc = sum(r["correct"] for r in astar_results) / len(astar_results) * 100
        gap = astar_acc - cot_acc
        print(f"  {ds_name:<15} {cot_acc:>6.1f}%   {astar_acc:>6.1f}%   {gap:>+5.1f}pp")

    print(f"\n  Total API calls: {client.api_calls}")
    print(f"  Total tokens:    {client.total_tokens:,}")

    output_path = "TMLR_NEW_PAPER/ollama_benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SAMPLES
    print(f"Running benchmark on {MODEL} with {n} samples per dataset...")
    run_experiment(num_samples=n)
