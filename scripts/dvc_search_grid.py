import itertools
import subprocess

# Automated grid search experiments
reg1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
reg2_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Iterate over all combinations of hyperparameter values.
for reg1, reg2 in itertools.product(reg1_values, reg2_values):
    # Execute "dvc exp run --queue --set-param model.reg1=<reg1> --set-param model.reg2=<reg2>".
    subprocess.run(
        [
            "dvc",
            "exp",
            "run",
            "--queue",
            "--set-param",
            f"model.reg1={reg1}",
            "--set-param",
            f"model.reg2={reg2}",
        ]
    )
