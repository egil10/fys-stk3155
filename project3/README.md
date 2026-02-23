# project 3: player value prediction

predicting football player market values using machine learning models on transfermarkt data.

## project overview
this project processes historical player data (biographical, events, performance) to predict valuations. it explores ridge regression and neural networks (mlp/rnn) with progressively complex feature sets.

## structure
```
project 3/
├── code/
│   ├── data/                # raw csvs (see code/data/readme.md)
│   ├── data_processed/      # output datasets
│   ├── implementations/     # processing scripts
│   └── notebooks/           # analysis & models
└── readme.md
```

## workflows & datasets

### 1. core feature set
- **source**: `code/implementations/generate_player_core_features.ipynb`
- **output**: `player_core_features.csv`
- **content**: player stats, cumulative performance, and 10-game lag features.

### 2. extended feature set (nationality)
- **source**: `code/implementations/generate_player_extended_features.ipynb`
- **output**: `player_extended_features.parquet`
- **content**: includes all core features plus 184 one-hot encoded nationality features. saved as parquet for performance.

## getting started

1. **setup data**:
   - download `players.csv`, `player_valuations.csv`, `game_events.csv` from [kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores).
   - place them in `code/data/`.

2. **generate data**:
   - run the notebooks in `code/implementations/` to build the datasets.

3. **run analysis**:
   - explore models in `code/notebooks/`.

## dependencies
`pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`, `pyarrow`, `jupyter`
