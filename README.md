# ASST

This is the code of Aligning Synchrosqueezing Transform (ASST), which is the submethod of the Statistic Synchrosqueezing Transform (Stat-SST) in the paper: Chen S, Wang S, An B, et al. Instantaneous Frequency Band and Synchrosqueezing in Time-Frequency Analysis[J]. IEEE Transactions on Signal Processing, 2023, 71: 539-554.

This code is based on pytorch, which can generate the highly-concentrated time-frequency representation in parallel.

Func "ASST":           original ASST
Func "SST_parellel":   SST in parallel
Func "ASST_parellel":  ASST in parallel
Func "iASST":          the inverse ASST when the hop is greater than 1
Func "iASST_2":        the inverse ASST when the hop is equal to 1
