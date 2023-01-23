import subprocess

# Automated grid search experiments
experiments = [
    {
        "distance": "emd",
        "weight_dist": "uniform",
        "temporal_type": None,
        "reg1": 0.015,
        "reg2": None,
        "nit": 100,
    },
    {
        "distance": "emd",
        "weight_dist": "idf",
        "temporal_type": None,
        "reg1": 0.015,
        "reg2": None,
        "nit": 100,
    },
    {
        "distance": "seq",
        "weight_dist": "uniform",
        "temporal_type": "TCOT",
        "reg1": 0.015,
        "reg2": None,
        "nit": 100,
    },
    {
        "distance": "seq",
        "weight_dist": "idf",
        "temporal_type": "TCOT",
        "reg1": 0.015,
        "reg2": None,
        "nit": 100,
    },
    {
        "distance": "seq",
        "weight_dist": "uniform",
        "temporal_type": "OPW",
        "reg1": 0.1,
        "reg2": 1,
        "nit": 100,
    },
    {
        "distance": "seq",
        "weight_dist": "idf",
        "temporal_type": "OPW",
        "reg1": 0.1,
        "reg2": 1,
        "nit": 100,
    },
]

# Iterate over all combinations of hyperparameter values.
for exp in experiments:
    # Execute "dvc exp run --queue --set-param model.reg1=<reg1> --set-param model.reg2=<reg2>".
    subprocess.run(
        [
            "dvc",
            "exp",
            "run",
            "--queue",
            "--set-param",
            f"model.distance={exp['distance']}",
            "--set-param",
            f"model.weight_dist={exp['weight_dist']}",
            "--set-param",
            f"model.temporal_type={exp['temporal_type']}",
            "--set-param",
            f"model.reg1={exp['reg1']}",
            "--set-param",
            f"model.reg2={exp['reg2']}",
            "--set-param",
            f"model.nit={exp['nit']}",
        ]
    )
