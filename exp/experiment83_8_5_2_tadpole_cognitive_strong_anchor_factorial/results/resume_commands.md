# Recovery command

Run from the authoritative experiment directory. The sealed runner verifies the source, data, split, configuration and preflight fingerprints. Folds with verified audit records are skipped. An interrupted fold resumes from its atomic Latest checkpoint.

```bash
cd /model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial
PYTHONPATH=audits python3 code/run_e8385.py \
  --groups S01 S02 S03 S04 S05 S06 S07 S08 S09 C01 C02 C03 \
  --workers 8
```
