# ADGL-LightMICN Supplementary Data

This directory contains the machine-readable data associated with the supplementary materials. The underlying meteorological observations were obtained from the China Meteorological Administration (CMA). All derived files were generated from the frozen best-performing model or the formal experiment outputs; no model was retrained when preparing these files.

## Files

- `Data_S1_dynamic_adjacency.npz`: 15 × 15 dynamic adjacency matrices and target dates for 662 test windows, provided for direct use in Python.
- `Data_S1_dynamic_adjacency_long.csv`: The same dynamic adjacency matrices in long-table format.
- `Data_S2_directed_edge_statistics.csv`: Selection frequency, mean normalized weight, weight standard deviation, and inter-station distance for all 210 directed non-self edges.
- `Data_S3a_neighbor_turnover.csv`: Changes in Top-k neighbor sets between consecutive test dates.
- `Data_S3b_daily_vs_mean_adjacency.csv`: Differences between each daily adjacency matrix and the mean adjacency matrix over the test period.
- `Data_S4_fixed_adjacency.csv`: The 15 × 15 binary adjacency matrix used in the fixed-graph comparison.
- `Data_S5_test_predictions.csv`: 9,930 target-date–station records containing observed temperature, predicted temperature, and residuals.
- `station_metadata.csv`: Station IDs, names, coordinates, English names, and abbreviations for the 15 stations.
- `seasonal_metrics.json` and `year_specific_winter_summer_metrics.csv`: Metrics describing winter–summer and interannual differences in the dynamic graphs.
- `SHA256SUMS.txt`: SHA-256 checksums for files in this directory.

## Adjacency matrix definition

The station order is provided in `station_metadata.csv`. Matrix rows represent receiving stations and columns represent source stations; therefore, A_ij denotes the directed connection j → i. The dynamic adjacency matrix A is row-normalized after Top-k selection and is not a temperature-correlation matrix. The exported matrices do not include self-loops. For GCN propagation, the model constructs A_bar = A + I and applies D^(-1/2) A_bar D^(-1/2). The fixed adjacency matrix also has a zero diagonal, and self-loops are added only during GCN propagation. The available project materials do not document how the original fixed adjacency matrix was constructed.

## Frequency, weight, and distance

Selection frequencies use the 662 test windows as the denominator. Mean weights and weight standard deviations are calculated from the row-normalized dynamic adjacency weights for individual windows. Inter-station distances are great-circle distances calculated from station coordinates and are reported in kilometres.

## Prediction records and metric reproduction

`Data_S5_test_predictions.csv` is the standard observation-level export for reproducing the overall prediction metrics reported in the main text. The test target dates range from 28 November 2024 to 20 September 2026, covering 662 dates. Each date contains records for 15 stations, giving 9,930 records in total. Residuals are defined as `residual_c = y_pred_c - y_true_c`. Overall MAE, RMSE, and R² are calculated by pooling all 9,930 station-date records, rather than by calculating station-level metrics first and then averaging them.

## Missing values and numerical precision

The formal test predictions and the eight selected variables for all 15 stations contain no missing values. If a missing value occurs in a CSV file, it is represented by an empty field; no special missing-value string is used. Predicted temperatures and residuals are reported to eight decimal places, while the other files retain the precision of the formal diagnostic outputs.
