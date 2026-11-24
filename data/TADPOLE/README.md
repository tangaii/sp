# TADPOLE input

The files in this directory are the six-modal TADPOLE input expected by
`main.py`. They were generated from the supplied
`/model/spromptgl/Tadpole/tadpole_challenge/TADPOLE_D1_D2.csv` and
`TADPOLE_D1_D2_Dict.csv` using the MMGL TADPOLE preprocessing protocol.

The input contains 598 unique subjects and 365 standardized features.
The MMGL protocol keeps the last qualifying visit per subject among visits
from baseline through month 48. Feature selection and standardization are
performed on the cohort before cross-validation, followed by mean imputation.

The modality dimensions are 145, 148, 3, 36, 24, and 9. The last CSV column
encodes 209 NC, 315 MCI, and 74 AD subjects as 1, 2, and 3, respectively.
