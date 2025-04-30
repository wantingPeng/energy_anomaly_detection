# Energy Data Cleaning Report


## Initial Dataset Statistics

- Total rows: 37865180

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

- Columns removed: rTotalPowerFactorPF, rTotalActiveEnergy, aPowerFactorPF_L1, aPowerFactorPF_L3, aFrequency_L1, aFrequency_L2, aFrequency_L3, airDruck, coolTotal, coolDurchfluss, coolTempeatur, coolDruck, celTemp, celHum


## Duplicate Removal

- Initial rows: 37865180

- Rows after deduplication: 32193340

- Duplicates removed: 5671840


## Timestamp Standardization


## Outlier Detection


### ID

- Lower bound: 202408250.00

- Upper bound: 1183664918.00

- Number of outliers: 0


### rTotalActivePower

- Lower bound: 1325.09

- Upper bound: 8898.13

- Number of outliers: 25497


### rTotalReactivePower

- Lower bound: 347.59

- Upper bound: 5936.15

- Number of outliers: 1842406


### rTotalApparentPower

- Lower bound: 4915.37

- Upper bound: 11654.41

- Number of outliers: 1616306


### rTotalReactiveEnergy

- Lower bound: 1896745.00

- Upper bound: 2166305.00

- Number of outliers: 3360565


### rTotalApparentEnergy

- Lower bound: 143030000.00

- Upper bound: 306990000.00

- Number of outliers: 0


### aCurrentL1

- Lower bound: 7.40

- Upper bound: 20.46

- Number of outliers: 1518768


### aCurrentL2

- Lower bound: 5.60

- Upper bound: 15.99

- Number of outliers: 1841623


### aCurrentL3

- Lower bound: 6.75

- Upper bound: 17.03

- Number of outliers: 1835721


### aVoltage_L1_N

- Lower bound: 217.22

- Upper bound: 233.59

- Number of outliers: 991043


### aVoltage_L2_N

- Lower bound: 218.13

- Upper bound: 233.69

- Number of outliers: 607782


### aVoltage_L3_N

- Lower bound: 217.61

- Upper bound: 233.61

- Number of outliers: 769193


### aActivePower_L1

- Lower bound: 829.39

- Upper bound: 3228.64

- Number of outliers: 547927


### aActivePower_L2

- Lower bound: -82.86

- Upper bound: 2863.86

- Number of outliers: 152218


### aActivePower_L3

- Lower bound: 423.51

- Upper bound: 2929.94

- Number of outliers: 302046


### aReactivePower_L1

- Lower bound: 315.78

- Upper bound: 2378.39

- Number of outliers: 1667014


### aReactivePower_L2

- Lower bound: -197.67

- Upper bound: 1944.46

- Number of outliers: 1704920


### aReactivePower_L3

- Lower bound: -37.34

- Upper bound: 1959.57

- Number of outliers: 2009566


### aApparentPower_L1

- Lower bound: 1697.89

- Upper bound: 4580.77

- Number of outliers: 1524869


### aApparentPower_L2

- Lower bound: 1290.43

- Upper bound: 3584.67

- Number of outliers: 1858292


### aApparentPower_L3

- Lower bound: 1559.78

- Upper bound: 3808.42

- Number of outliers: 1869438


### aCosPhi_L1

- Lower bound: 0.66

- Upper bound: 1.00

- Number of outliers: 1450045


### aCosPhi_L2

- Lower bound: 0.65

- Upper bound: 1.03

- Number of outliers: 2136340


### aCosPhi_L3

- Lower bound: 0.75

- Upper bound: 0.98

- Number of outliers: 3590767


### aPowerFactorPF_L2

- Lower bound: 0.31

- Upper bound: 0.88

- Number of outliers: 767290


### airTotal

- Lower bound: 207356.00

- Upper bound: 764316.00

- Number of outliers: 0


### airDurchfluss

- Lower bound: -41.80

- Upper bound: 91.00

- Number of outliers: 0


### airTempeatur

- Lower bound: 22.90

- Upper bound: 30.10

- Number of outliers: 52239
