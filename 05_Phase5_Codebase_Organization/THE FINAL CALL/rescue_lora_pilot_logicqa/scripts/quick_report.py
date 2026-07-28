#!/usr/bin/env python3
import json
from pathlib import Path

root = Path('/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa')
summary = root / 'data/rescue_pairs/summary.json'
filter_stats = root / 'data/rescue_pairs/filter_stats.json'

if summary.exists():
    print('=== Summary ===')
    print(summary.read_text())
if filter_stats.exists():
    print('\n=== Filter Stats ===')
    print(filter_stats.read_text())
