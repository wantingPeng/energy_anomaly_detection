# Energy Data Cleaning Report


## Initial Dataset Statistics

- Total rows: 37654789

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

- Columns removed: rTotalPowerFactorPF, rTotalActiveEnergy, aPowerFactorPF_L3, aFrequency_L1, aFrequency_L2, aFrequency_L3, airDruck, coolTotal, coolDurchfluss, coolTempeatur, coolDruck, celTemp, celHum


## Duplicate Removal

- Initial rows: 37654789

- Rows after deduplication: 34720278

- Duplicates removed: 2934511


## Outlier Detection

- Total records marked as outliers: 19062261

- Percentage of outliers: 54.90%


### ID

- Lower bound: 185751710.12

- Upper bound: 1239000149.12

- Number of outliers: 0


### rTotalActivePower

- Lower bound: 205.44

- Upper bound: 1408.36

- Number of outliers: 231252


### rTotalReactivePower

- Lower bound: -87.47

- Upper bound: 1097.72

- Number of outliers: 5156374


### rTotalApparentPower

- Lower bound: 316.31

- Upper bound: 1969.83

- Number of outliers: 461400


### rTotalReactiveEnergy

- Lower bound: 1118065.00

- Upper bound: 1571545.00

- Number of outliers: 0


### rTotalApparentEnergy

- Lower bound: 23578750.00

- Upper bound: 47060750.00

- Number of outliers: 0


### aCurrentL1

- Lower bound: -0.04

- Upper bound: 3.20

- Number of outliers: 1125049


### aCurrentL2

- Lower bound: 0.93

- Upper bound: 1.83

- Number of outliers: 9009578


### aCurrentL3

- Lower bound: -0.08

- Upper bound: 4.15

- Number of outliers: 71669


### aVoltage_L1_N

- Lower bound: 216.06

- Upper bound: 233.95

- Number of outliers: 1674621


### aVoltage_L2_N

- Lower bound: 218.69

- Upper bound: 234.47

- Number of outliers: 207243


### aVoltage_L3_N

- Lower bound: 218.21

- Upper bound: 234.60

- Number of outliers: 196576


### aActivePower_L1

- Lower bound: 73.90

- Upper bound: 393.25

- Number of outliers: 695923


### aActivePower_L2

- Lower bound: 137.51

- Upper bound: 270.55

- Number of outliers: 8836321


### aActivePower_L3

- Lower bound: -90.45

- Upper bound: 783.27

- Number of outliers: 35692


### aReactivePower_L1

- Lower bound: -208.86

- Upper bound: 516.05

- Number of outliers: 2325023


### aReactivePower_L2

- Lower bound: 47.10

- Upper bound: 227.68

- Number of outliers: 7100483


### aReactivePower_L3

- Lower bound: 63.10

- Upper bound: 263.26

- Number of outliers: 13692901


### aApparentPower_L1

- Lower bound: 19.27

- Upper bound: 698.40

- Number of outliers: 1381740


### aApparentPower_L2

- Lower bound: 210.63

- Upper bound: 412.66

- Number of outliers: 8929036


### aApparentPower_L3

- Lower bound: -21.26

- Upper bound: 942.38

- Number of outliers: 68563


### aCosPhi_L1

- Lower bound: -0.21

- Upper bound: 1.59

- Number of outliers: 7009929


### aCosPhi_L2

- Lower bound: 0.61

- Upper bound: 1.02

- Number of outliers: 5201101


### aCosPhi_L3

- Lower bound: 0.76

- Upper bound: 0.96

- Number of outliers: 6393732


### aPowerFactorPF_L1

- Lower bound: 0.12

- Upper bound: 1.09

- Number of outliers: 0


### aPowerFactorPF_L2

- Lower bound: 0.50

- Upper bound: 0.78

- Number of outliers: 4923102


### airTotal

- Lower bound: 74215.50

- Upper bound: 330307.50

- Number of outliers: 0


### airDurchfluss

- Lower bound: -4.15

- Upper bound: 24.25

- Number of outliers: 8709


### airTempeatur

- Lower bound: 24.10

- Upper bound: 29.70

- Number of outliers: 4451613
