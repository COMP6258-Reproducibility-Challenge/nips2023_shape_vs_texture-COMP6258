import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template_AlexNet


def run_evaluation():
    # models = ["AlexNet_topK_50","AlexNet_topK_40","AlexNet_topK_30", "AlexNet_topK_20","AlexNet_topK_10","AlexNet_topK_5","AlexNet_normal"]
    models = ["AlexNet_normal", "AlexNet_topK_5"]
    datasets = ["cue-conflict"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 14}
    Evaluate()(models, datasets, **params)


def run_plotting():
    # plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plot_types=["shape-bias"]
    plotting_def = plotting_definition_template_AlexNet
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
