-- DDL for feature_binned_samples table
CREATE TABLE feature_binned_samples (
    feature_id INTEGER NOT NULL,
    bin_index INTEGER NOT NULL,
    bin_min_activation REAL,
    bin_max_activation REAL,
    example_id INTEGER,
    activation_value REAL,
    rank_in_bin INTEGER NOT NULL,
    PRIMARY KEY (feature_id, bin_index, rank_in_bin),
    FOREIGN KEY (example_id) REFERENCES examples(id)
);

-- Index for feature_id and bin_index on feature_binned_samples
CREATE INDEX idx_fbs_feature_bin ON feature_binned_samples (feature_id, bin_index);

-- Index for example_id on feature_binned_samples
CREATE INDEX idx_fbs_example ON feature_binned_samples (example_id);

-- DDL for feature_stats_new table
CREATE TABLE feature_stats_new (
    feature_id INTEGER PRIMARY KEY NOT NULL,
    overall_min_activation REAL,
    overall_max_activation REAL,
    estimated_mean REAL,
    estimated_median REAL,
    num_sampled_examples INTEGER,
    num_bins INTEGER,
    sampled_histogram_data TEXT -- For storing JSON data
);
