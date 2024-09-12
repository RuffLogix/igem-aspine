# igem-aspine

Breast cancer remains a primary global health concern. Dendritic Cell (DC) vaccines are a promising immunotherapy that targets tumors using neoantigens. However, personalizing these vaccines requires mRNA sequencing, which is likely unavailable. ASPINE aims to address this issue by developing a machine-learning model to predict neoantigen expression using DNA sequences.
We conducted four experiments using the Xpresso model, the first two using an obtained genome sequence and the latter two using an acquired patient data. The second and the fourth experiments have a shorter input sequence length to control for our shorter WES data in experiment four. With acquired patient data, the third experiment yielded an R-squared of 0.329 (RMSE: 0.440). The fourth experiment had an R-squared of 0.294 (RMSE: 0.443), showing improved performance compared to non-patient data. Future work will focus on optimizing the model and refining transcription start site classification to improve prediction accuracy.

[Team Wiki](https://2024.igem.wiki/bangkok-nmh/)
