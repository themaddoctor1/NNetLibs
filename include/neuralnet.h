#ifndef _NEURALNET_H_
#define _NEURALNET_H_

#include "matrix.h"

/****************/
/* NEURON LAYER */
/****************/

typedef Matrix (*TransFunc)(Matrix);

Matrix linearTransfer(Matrix m); // f(X) = X
Matrix linearTransferGradient(Matrix m); // d/dX X = 1

Matrix sigmoidTransfer(Matrix m); // f(X) = 1 / (1 + e^(-X))
Matrix sigmoidTransferGradient(Matrix m); // d/dX f = f (1 - f)

Matrix unitStepTransfer(Matrix m); //f(x) = x >= 0 ? 1 : 0
Matrix competeTransfer(Matrix m); //f(x) = v | v_i = v_i >= v_j forall j ? 1 : 0
Matrix zeroMatrix(Matrix m); //d/dx c = o

struct neuron_layer;

typedef struct neuron_layer* NeuronLayer;

/* NeuronLayer factories */
NeuronLayer makeBlankNeuronLayer(int in, int out, TransFunc func);
NeuronLayer maleBlankRecurrentLayer(int in, int out, int r, TransFunc func);
NeuronLayer makePresetNeuronLayer(Matrix W, Matrix R, int r, TransFunc func);

void freeNeuronLayer(NeuronLayer layer);

/* Getters for NeuronLayer */
Matrix getLayerWeights(NeuronLayer layer);
Matrix getLayerRecurrentWeights(NeuronLayer layer);
int getLayerRecurrence(NeuronLayer layer);
TransFunc getLayerFunc(NeuronLayer layer);

/* Setter methods */
void setLayerWeights(NeuronLayer layer, Matrix m);
void setLayerRecurrentWeights(NeuronLayer layer, Matrix r);
void setLayerRecurrence(NeuronLayer layer, int r);
void setLayerFunc(NeuronLayer layer, TransFunc func);

/* Runs a single layer, assuming it is not recurrent. */
Matrix layerFunction(NeuronLayer layer, Matrix x);

/* Runs a single layer, allowing it to be recurrent. */
Matrix* layerRecurrentFunction(NeuronLayer layer, Matrix *x);
Matrix layerRaw(NeuronLayer layer, Matrix x);

/******************/
/* NEURAL NETWORK */ 
/******************/

struct neural_net;
typedef struct neural_net* NeuralNet;

/* Neural Net factory */
NeuralNet makeNeuralNet(int sizes[]);
void freeNeuralNet(NeuralNet net);

/* Getter methods */
NeuronLayer getNetLayer(NeuralNet net, int layer);
Matrix getNetWeights(NeuralNet net, int layer);
int getNetDepth(NeuralNet net);

/* Runs network on an input Matrix */
Matrix netFunction(NeuralNet net, Matrix x);

/* Runs a recurrent network on a set of input matrices. */
Matrix* netRecurrentFunction(NeuralNet net, Matrix *xs);

#endif


