# Energy Data Cleaning Report


## Initial Dataset Statistics

- Total rows: 37656673

- Total columns: 44

- Missing values per column:

  - ID: 0

  - TimeStamp: 0

  - Station: 0

  - rTotalActivePower: 0

  - rTotalReactivePower: 0

  - rTotalApparentPower: 0

  - rTotalPowerFactorPF: 0

  - rTotalActiveEnergy: 0

  - rTotalReactiveEnergy: 0

  - rTotalApparentEnergy: 0

  - aCurrentL1: 0

  - aCurrentL2: 0

  - aCurrentL3: 0

  - aVoltage_L1_N: 0

  - aVoltage_L2_N: 0

  - aVoltage_L3_N: 0

  - aActivePower_L1: 0

  - aActivePower_L2: 0

  - aActivePower_L3: 0

  - aReactivePower_L1: 0

  - aReactivePower_L2: 0

  - aReactivePower_L3: 0

  - aApparentPower_L1: 0

  - aApparentPower_L2: 0

  - aApparentPower_L3: 0

  - aCosPhi_L1: 0

  - aCosPhi_L2: 0

  - aCosPhi_L3: 0

  - aPowerFactorPF_L1: 0

  - aPowerFactorPF_L2: 0

  - aPowerFactorPF_L3: 0

  - aFrequency_L1: 0

  - aFrequency_L2: 0

  - aFrequency_L3: 0

  - airTotal: 0

  - airDurchfluss: 0

  - airTempeatur: 0

  - airDruck: 0

  - coolTotal: 0

  - coolDurchfluss: 0

  - coolTempeatur: 0

  - coolDruck: 0

  - celTemp: 0

  - celHum: 0


## Removed Zero Variance Columns

- Columns removed: rTotalPowerFactorPF, rTotalActiveEnergy, aPowerFactorPF_L3, aFrequency_L1, aFrequency_L2, aFrequency_L3, coolTotal, coolDurchfluss, coolTempeatur, coolDruck


## Duplicate Removal

- Initial rows: 37656673

- Rows after deduplication: 34721669

- Duplicates removed: 2935004
