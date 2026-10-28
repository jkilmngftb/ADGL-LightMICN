# Dataset

## Files

- `CMA_Jiangsu_daily_2017_2026.xlsx`: daily meteorological observations used in the paper.
- `adjacency_matrix.csv`: fixed 15 x 15 station adjacency matrix used by fixed-graph experiments.

## Source and coverage

The meteorological observations are from the **China Meteorological Administration (CMA)**. The workbook contains observations for 15 stations in Jiangsu Province from 2017-01-01 to 2026-09-20. The experiments use eight daily meteorological variables and predict next-day mean temperature.

The data are split chronologically into 70% training, 10% validation, and 20% test subsets. Normalization statistics are computed from the training subset only. The common comparison window is 48 days, while the forecasting input window is 12 days.

## Station order and adjacency

Station order follows the workbook and is documented in `../supplementary_data/README.md`. For adjacency matrix `A`, row `i` is the receiving station and column `j` is the source station. Model-specific self-loop and normalization operations are applied in code.
