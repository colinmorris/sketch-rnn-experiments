A playground for experiments with the [Quick Draw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and [Sketch-RNN](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn). Sorry about the mess.

`get_perplexities`, the function in the iPython notebook for calculating the loss per sketch in the dataset relies on a one-line local modification I made to Magenta to make it possible to grab the pre-aggregated loss of a batch. I added the following line after the call to `get_lossfunc` in sketch_rnn/model.py: `self.lossfunc = lossfunc`.

If you don't want to bother building Magenta from source, you can use `_get_perplexities` with a model having `hps.batch_size = 1`, but it's an order of magnitude slower.
