# Data Cleaning Report
Generated on: 2025-04-28 06:17:09

## Initial Data Overview
- Total rows: 2696875
- Total columns: 44
- Memory usage: 905.32 MB

## Cleaning Steps and Results

### 1. Removed Useless Columns
- Initial column count: 44
- Final column count: 33
- Removed columns (11):
```
- Station
- rTotalActiveEnergy
- aFrequency_L2
- aFrequency_L3
- airDruck
- coolTotal
- coolDurchfluss
- coolTempeatur
- coolDruck
- celTemp
- celHum
```

### 2. Duplicate Rows Removal
- Initial row count: 2696875
- Duplicates removed: 204826
- Final row count: 2492049
- Percentage reduced: 7.59%

### 3. Timestamp Standardization
- Format: Converted to datetime64[ns, UTC]
- Timezone: Standardized to UTC
- Time range: 2024-04-01 00:00:00+00:00 to 2024-04-30 23:59:59+00:00

### 4. Station Column Normalization
- Standardized text formatting (uppercase, stripped whitespace)
- Unique values: Column not found

### 5. Outlier Analysis

#### ID
- Range: 567514206.00 to 602844039.00
- Q1: 576566428.00
- Q3: 594126670.00
- IQR: 17560242.00
- Outlier bounds: [550226065.00, 620467033.00]
- Number of outliers: 0 (0.00%)

#### rTotalActivePower
- Range: 2253.52 to 11940.20
- Q1: 4921.39
- Q3: 6575.76
- IQR: 1654.37
- Outlier bounds: [2439.84, 9057.32]
- Number of outliers: 5331 (0.21%)

#### rTotalReactivePower
- Range: -1897.00 to 6990.76
- Q1: 2647.14
- Q3: 4033.94
- IQR: 1386.80
- Outlier bounds: [566.94, 6114.14]
- Number of outliers: 62944 (2.53%)

#### rTotalApparentPower
- Range: 2930.48 to 14997.90
- Q1: 7930.87
- Q3: 9642.47
- IQR: 1711.60
- Outlier bounds: [5363.47, 12209.87]
- Number of outliers: 6462 (0.26%)

#### rTotalPowerFactorPF
- Range: 0.27 to 0.95
- Q1: 0.61
- Q3: 0.69
- IQR: 0.07
- Outlier bounds: [0.50, 0.80]
- Number of outliers: 117017 (4.70%)

#### rTotalReactiveEnergy
- Range: 1997820.00 to 1998800.00
- Q1: 1998340.00
- Q3: 1998760.00
- IQR: 420.00
- Outlier bounds: [1997710.00, 1999390.00]
- Number of outliers: 0 (0.00%)

#### rTotalApparentEnergy
- Range: 204049000.00 to 210365000.00
- Q1: 205761000.00
- Q3: 208959000.00
- IQR: 3198000.00
- Outlier bounds: [200964000.00, 213756000.00]
- Number of outliers: 0 (0.00%)

#### aCurrentL1
- Range: 4.69 to 29.35
- Q1: 13.79
- Q3: 17.30
- IQR: 3.51
- Outlier bounds: [8.53, 22.57]
- Number of outliers: 2193 (0.09%)

#### aCurrentL2
- Range: 3.25 to 21.40
- Q1: 9.68
- Q3: 12.05
- IQR: 2.37
- Outlier bounds: [6.13, 15.61]
- Number of outliers: 31833 (1.28%)

#### aCurrentL3
- Range: 4.49 to 26.33
- Q1: 11.36
- Q3: 14.01
- IQR: 2.65
- Outlier bounds: [7.38, 17.98]
- Number of outliers: 21678 (0.87%)

#### aVoltage_L1_N
- Range: 215.38 to 233.40
- Q1: 222.91
- Q3: 225.88
- IQR: 2.97
- Outlier bounds: [218.46, 230.33]
- Number of outliers: 30562 (1.23%)

#### aVoltage_L2_N
- Range: 216.51 to 234.33
- Q1: 224.03
- Q3: 226.97
- IQR: 2.94
- Outlier bounds: [219.62, 231.39]
- Number of outliers: 35818 (1.44%)

#### aVoltage_L3_N
- Range: 215.90 to 234.16
- Q1: 223.57
- Q3: 226.57
- IQR: 3.00
- Outlier bounds: [219.07, 231.07]
- Number of outliers: 36586 (1.47%)

#### aActivePower_L1
- Range: 671.13 to 5274.56
- Q1: 1899.14
- Q3: 2636.74
- IQR: 737.60
- Outlier bounds: [792.74, 3743.14]
- Number of outliers: 1103 (0.04%)

#### aActivePower_L2
- Range: 333.90 to 3892.14
- Q1: 1265.66
- Q3: 1820.88
- IQR: 555.22
- Outlier bounds: [432.83, 2653.71]
- Number of outliers: 37228 (1.49%)

#### aActivePower_L3
- Range: 651.42 to 4332.13
- Q1: 1626.13
- Q3: 2186.05
- IQR: 559.92
- Outlier bounds: [786.25, 3025.93]
- Number of outliers: 30692 (1.23%)

#### aReactivePower_L1
- Range: -694.05 to 3136.61
- Q1: 1117.13
- Q3: 1654.27
- IQR: 537.14
- Outlier bounds: [311.42, 2459.98]
- Number of outliers: 13419 (0.54%)

#### aReactivePower_L2
- Range: -708.42 to 2494.53
- Q1: 615.66
- Q3: 1053.27
- IQR: 437.61
- Outlier bounds: [-40.75, 1709.68]
- Number of outliers: 90140 (3.62%)

#### aReactivePower_L3
- Range: -692.43 to 2913.67
- Q1: 857.92
- Q3: 1366.01
- IQR: 508.09
- Outlier bounds: [95.78, 2128.14]
- Number of outliers: 38061 (1.53%)

#### aApparentPower_L1
- Range: 1052.95 to 6532.43
- Q1: 3103.26
- Q3: 3875.09
- IQR: 771.83
- Outlier bounds: [1945.52, 5032.84]
- Number of outliers: 3405 (0.14%)

#### aApparentPower_L2
- Range: 727.70 to 4970.58
- Q1: 2177.02
- Q3: 2718.35
- IQR: 541.33
- Outlier bounds: [1365.03, 3530.34]
- Number of outliers: 30630 (1.23%)

#### aApparentPower_L3
- Range: 1017.79 to 6036.61
- Q1: 2548.48
- Q3: 3150.86
- IQR: 602.38
- Outlier bounds: [1644.91, 4054.43]
- Number of outliers: 21249 (0.85%)

#### aCosPhi_L1
- Range: -1.00 to 1.00
- Q1: 0.79
- Q3: 0.90
- IQR: 0.10
- Outlier bounds: [0.64, 1.05]
- Number of outliers: 2373 (0.10%)

#### aCosPhi_L2
- Range: -1.00 to 1.00
- Q1: 0.84
- Q3: 0.91
- IQR: 0.06
- Outlier bounds: [0.75, 1.01]
- Number of outliers: 105855 (4.25%)

#### aCosPhi_L3
- Range: -1.00 to 1.00
- Q1: 0.84
- Q3: 0.89
- IQR: 0.06
- Outlier bounds: [0.75, 0.98]
- Number of outliers: 129328 (5.19%)

#### aPowerFactorPF_L1
- Range: 0.17 to 0.95
- Q1: 0.62
- Q3: 0.68
- IQR: 0.07
- Outlier bounds: [0.52, 0.79]
- Number of outliers: 30783 (1.24%)

#### aPowerFactorPF_L2
- Range: 0.20 to 0.96
- Q1: 0.57
- Q3: 0.68
- IQR: 0.11
- Outlier bounds: [0.40, 0.85]
- Number of outliers: 177191 (7.11%)

#### aPowerFactorPF_L3
- Range: 0.31 to 0.97
- Q1: 0.63
- Q3: 0.70
- IQR: 0.07
- Outlier bounds: [0.52, 0.80]
- Number of outliers: 140102 (5.62%)

#### aFrequency_L1
- Range: 49.83 to 50.12
- Q1: 49.99
- Q3: 50.02
- IQR: 0.03
- Outlier bounds: [49.94, 50.06]
- Number of outliers: 45952 (1.84%)

#### airTotal
- Range: 415151.00 to 435899.00
- Q1: 420138.00
- Q3: 431008.00
- IQR: 10870.00
- Outlier bounds: [403833.00, 447313.00]
- Number of outliers: 0 (0.00%)

#### airDurchfluss
- Range: 0.00 to 54.10
- Q1: 9.50
- Q3: 41.50
- IQR: 32.00
- Outlier bounds: [-38.50, 89.50]
- Number of outliers: 0 (0.00%)

#### airTempeatur
- Range: 23.20 to 31.40
- Q1: 25.20
- Q3: 26.80
- IQR: 1.60
- Outlier bounds: [22.80, 29.20]
- Number of outliers: 4640 (0.19%)

## Final Data Overview
- Total rows: 2492049
- Total columns: 33
- Memory usage: 646.44 MB

## Remaining Columns
```
- ID
- TimeStamp
- rTotalActivePower
- rTotalReactivePower
- rTotalApparentPower
- rTotalPowerFactorPF
- rTotalReactiveEnergy
- rTotalApparentEnergy
- aCurrentL1
- aCurrentL2
- aCurrentL3
- aVoltage_L1_N
- aVoltage_L2_N
- aVoltage_L3_N
- aActivePower_L1
- aActivePower_L2
- aActivePower_L3
- aReactivePower_L1
- aReactivePower_L2
- aReactivePower_L3
- aApparentPower_L1
- aApparentPower_L2
- aApparentPower_L3
- aCosPhi_L1
- aCosPhi_L2
- aCosPhi_L3
- aPowerFactorPF_L1
- aPowerFactorPF_L2
- aPowerFactorPF_L3
- aFrequency_L1
- airTotal
- airDurchfluss
- airTempeatur
```
