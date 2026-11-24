# TADPOLE experiment data

The generated feature matrix, modality dictionary, and subject visit table are
kept out of the public repository because they contain subject-level TADPOLE
records. The experiment expects the following files in this directory:

- `processed_standard_data.csv`
- `modal_feat_dict.npy`
- `samples.csv`

To recreate them, place `TADPOLE_D1_D2.csv` and `TADPOLE_D1_D2_Dict.csv` from
the TADPOLE challenge under `/model/spromptgl/Tadpole/tadpole_challenge`, then
run:

```bash
python exp/tadpole_repair/prepare.py
```

The script follows the deterministic MMGL preprocessing protocol and writes
the feature and preprocessing manifests in this directory. The default
experiment uses the fixed 598-subject NC/MCI/AD cohort with modality dimensions
145, 148, 3, 36, 24, and 9.
