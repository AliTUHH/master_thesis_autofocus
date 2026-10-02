| Candidate | Fr error [%] | b = sqrt(abs(e)/Fr) [px] (mean) | n div. | NRMSE phase (median) | NRMSE offset-free (median) | NRMSE gradient (median) | Pearson (median) | SSIM (median) | Core loss (median) | Data residual at Fr_true (median) | NRMSE rel. true (paired median) | NRMSE grad rel. true | Residual rel. true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| true | +0.00 | 0.0 | 0/20 | 0.723 | 0.776 | 1.385 | 0.717 | 0.111 | 1.438e-04 | 0.0623 | 1.00 | 1.00 | 1.00 |
| ml_cnn | -1.49 | 2.8 | 0/20 | 0.748 | 0.796 | 1.378 | 0.695 | 0.108 | 1.433e-04 | 0.1471 | 1.00 | 1.01 | 2.27 |
| ml_cnn_hologram_spectrum | +0.40 | 2.9 | 0/20 | 0.763 | 0.806 | 1.391 | 0.698 | 0.107 | 1.432e-04 | 0.1381 | 1.00 | 1.02 | 2.10 |
| ringfit_cos | +0.43 | 4.6 | 0/20 | 0.771 | 0.762 | 1.400 | 0.718 | 0.113 | 1.608e-04 | 0.1071 | 1.00 | 1.01 | 2.20 |
| npe_radial_ens5_median | +4.06 | 3.8 | 0/20 | 0.752 | 0.802 | 1.400 | 0.676 | 0.085 | 1.538e-04 | 0.1528 | 1.00 | 1.02 | 2.54 |
| holowizard_find_focus | -4.54 | 2.2 | 0/20 | 0.753 | 0.769 | 1.360 | 0.735 | 0.107 | 1.529e-04 | 0.1254 | 0.99 | 0.99 | 2.00 |
| classical_tv | +3.77 | 1.8 | 0/20 | 0.760 | 0.790 | 1.409 | 0.686 | 0.111 | 1.384e-04 | 0.0780 | 1.00 | 1.00 | 1.14 |
