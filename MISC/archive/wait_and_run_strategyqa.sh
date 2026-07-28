#!/bin/bash
LOGICQA_PID=49577
HOTPOTQA_PID=52105

echo "[$(date)] Waiting for LogicQA (PID $LOGICQA_PID) and HotpotQA (PID $HOTPOTQA_PID) to finish..."

while kill -0 $LOGICQA_PID 2>/dev/null || kill -0 $HOTPOTQA_PID 2>/dev/null; do
    # print status every 5 minutes
    REMAINING=""
    kill -0 $LOGICQA_PID 2>/dev/null && REMAINING+="LogicQA "
    kill -0 $HOTPOTQA_PID 2>/dev/null && REMAINING+="HotpotQA "
    echo "[$(date)] Still running: $REMAINING"
    sleep 300
done

echo "[$(date)] Both done! Starting StrategyQA CoT..."

CUDA_VISIBLE_DEVICES=0 python "/home/dibyanayan/tanzeel/COT Reasoning qwen/strategyqa_cot.py" \
    --model "qwen2.5:0.5b" \
    --limit 500 \
    --output "/home/dibyanayan/tanzeel/COT Reasoning qwen/qwen_results/qwen0.5b_strategyqa_cot_results.json" \
    2>&1 | tee "/home/dibyanayan/tanzeel/COT Reasoning qwen/qwen_results/strategyqa_cot_log.txt"

echo "[$(date)] StrategyQA CoT done!"
