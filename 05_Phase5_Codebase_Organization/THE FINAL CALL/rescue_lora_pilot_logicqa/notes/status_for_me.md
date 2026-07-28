# Status For Me (Beginner Version)

1. We are trying to fix cases where normal CoT reasoning gives a wrong answer.
2. We used A* reasoning as a teacher because A* got some of those cases correct.
3. We took only the useful rescue cases, cleaned them, and made a small training set.
4. Day 1 was data work: bucket, rewrite, filter, and train/dev file creation.
5. Day 2 was model work: train LoRA once, evaluate, then run one small fix.
6. We trained two main Qwen adapters: Run-1 and Fix-1.
7. Rescue improved from 0.00 to 33.33 (so the idea works at least partially).
8. Preservation improved from 0.00 to 12.50 after Fix-1, but it is still low.
9. So this is a successful pilot signal, not a final production model.
10. Next step is to improve preservation without losing rescue.

Quick file map:
- Day 1 details: notes/day1_log.md
- Day 2 details: notes/day2_log.md
- Best current adapter: outputs/adapters/rescue_lora_logicqa_qwen05b_fix1/final
- Final metrics: outputs/metrics/rescue_eval_qwen05b_fix1.json
