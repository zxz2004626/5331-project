"""Names and experiment dimensions shared by command-line workflows."""

DATASETS = ["movielens", "ucimessages"]
SEEDS = [11, 23, 37, 51, 71]
GENERATORS = ["G0", "G1", "G2", "G3"]
CORE_FEATURES = ["tgf", "extended"]
ABLATIONS = ["local", "temporal", "tgf_local", "tgf_temporal", "local_temporal"]
SENSITIVITIES = ["ratio01", "ratio10", "window500", "window2000", "past_only"]
SCHEMA_VERSION = 1
