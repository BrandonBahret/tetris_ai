import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np

from GA.GeneticTrainer import GeneticModelTrainer
from GA import block_configs


if __name__ == "__main__":
    blocks_set = []
    blocks = block_configs.broken.copy()
    blocks[:10].fill(0)
    blocks_set.append(blocks)
    # blocks_set.append(block_configs.overhang_1.copy())
    # blocks_set.append(block_configs.overhang_2.copy())
    # blocks_set.append(block_configs.well_1.copy())
    # blocks_set.append(block_configs.well_2.copy())

    metrics_count = 2
    metric_length = 5
    parameter_count = metric_length * metrics_count
    parameter_affect = 1

    # trainer = GeneticModelTrainer(
    #     blocks_set=blocks_set,
    #     base_pop_size=(parameter_count / (1 - np.round(parameter_affect/parameter_count, decimals=5))) * 10,
    #     mutation_rate=np.round(parameter_affect/parameter_count, decimals=5),
    #     weight_step=1/metrics_count
    # )

    trainer = GeneticModelTrainer(
        blocks_set=blocks_set,
        base_pop_size=(parameter_count / (1 - np.round(parameter_affect/parameter_count, decimals=5))) * 4,
        mutation_rate=np.round(parameter_affect/parameter_count, decimals=5),
        weight_step=0.01,
        iteration_depth=75
    )

    print("Training started!")
    trained_model = trainer.train(generations=1000)

    import winsound
    winsound.Beep(500, 1000)
    print("training done!")
