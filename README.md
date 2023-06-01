# ASST

This is the code of Aligning Synchrosqueezing Transform (ASST), which is the submethod of the Statistic Synchrosqueezing Transform (Stat-SST) in the paper: Chen S, Wang S, An B, et al. Instantaneous Frequency Band and Synchrosqueezing in Time-Frequency Analysis[J]. IEEE Transactions on Signal Processing, 2023, 71: 539-554.

We provide the codes based on pytorch and Matlab, which can generate the highly-concentrated time-frequency representation in parallel.
If you want to reconstruct the signal components, respectively, you can use any Ridge search method to search the index of the component in ASST result and extract the TF coefficients. Then, the signal components can be reconstructed by the reconstruction code.



# Citations 
Please cite this paper:

@article{chen2023instantaneous,
  title={Instantaneous Frequency Band and Synchrosqueezing in Time-Frequency Analysis},
  author={Chen, Shaowen and Wang, Shibin and An, Botao and Yan, Ruqiang and Chen, Xuefeng},
  journal={IEEE Transactions on Signal Processing},
  volume={71},
  pages={539--554},
  year={2023},
  publisher={IEEE}
}

# Mail:
shaowen.chen@g.sp.m.is.nagoya-u.ac.jp
or 
ShaowenChen1995@gmail.com
