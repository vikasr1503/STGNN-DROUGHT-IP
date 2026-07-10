# Data Directory

## Layout

```
data/
  raw/            Original datasets — NOT tracked by Git
  interim/        Cleaned, merged monthly CSV
  processed/      Node feature matrix, drought indices
  tensors/        X.npy [T,N,F], Y.npy [T,N,3], dates.npy
```

## Expected raw/ structure

```
data/raw/
  subbasin.csv                  130 sub-basin attributes (PolygonId, Area, Subbasin)
  discharge/
    station_001.csv             columns: date, subbasin_id, discharge_m3s
    ...
  atmosphere/
    Temperature_All_Stations_*_ma.csv
    Atmospheric_Pressure_All_Stations_*_ma.csv
    Solar_Radiation_All_Stations_*_ma.csv
    Relative_Humidity_All_Stations_*_ma.csv
    Wind_Direction_All_Stations_*_ma.csv
  swat_outputs/
    swat_monthly.csv            columns: date, subbasin_id, runoff_mm,
                                         soil_moisture_mm, et_mm,
                                         groundwater_mm, streamflow_m3s
  groundwater/
    gw_station_001.csv          columns: date, subbasin_id, groundwater_m
  reservoir/
    reservoir_001.csv           columns: date, reservoir_id, subbasin_id,
                                         storage_mcm, release_mcm
```

## Tensor shapes

| File            | Shape          | Description                        |
|-----------------|----------------|------------------------------------|
| X.npy           | [T, N, F]      | Node feature tensor                |
| Y.npy           | [T, N, 3]      | SRI-3, SRI-6, SRI-12 targets       |
| dates.npy       | [T]            | Monthly timestamps                 |
| metadata.json   | —              | Feature names, subbasin IDs, dims  |
