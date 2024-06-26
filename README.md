# BrainOps
This is a repo for reproducing the result of the paper Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity.

We have recreated Figures 2 (texture-synthesis), 3 (reconstruction) and 5 (benchmarking) and have placed the associated code in the named directories.

## Figure 2
To recreate figure 2, see the associated [Jupyter notebook](texture-synthesis/example.ipynb) and run the code
Adjust the following hyper-parameters according your need 

- synthesizer.TARGET_LAYERS = [1, 4, 9, 16, 22, 30] 
- synthesizer.IMAGE_SIZE = [256,256]
- synthesizer.EPHOCS = 2000
- synthesizer.TOPK = 0.05
- synthesizer.REVERSE = True (True for non-topk and False for topk mask)
  
The image will be generated in the notebook

## Figure 3
To recreate figure 3, see the associated [Jupyter notebook](reconstruction/reconstruction.ipynb) and run the code. 

The resulting figures will be in the [results](reconstruction/results) directory, in all three shapes.

## Figure 5
To run the model-vs-human benchmark, first follow the instruction in the readme file under that, pointing towards https://github.com/bethgelab/model-vs-human/ (export and install requirements).

To recreate the results, run the evaluate.py file under model-vs-human/examples/evaluate.py.

To recreate the results using our custom implementation of the top-K layer, you would need to change the import statement under model-vs-human/modelvshuman/models/pytorch/model_zoo.py. Further information about that can be found in that file.
