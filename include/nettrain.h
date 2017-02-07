#ifndef _NETTRAIN_H_
#define _NETTRAIN_H_

#include "neuralnet.h"
#include "matrix.h"

struct nettrainkit {
    TransFunc* functions;
    TransFunc* derivatives;
    Matrix **data;
    double learnRate; // A constant that dictates network change speed.
    double momentum; // A constant that allows some of a previous change to be applied.
    double decay;
    int maxCycles;
};
typedef struct nettrainkit* NetTrainKit;

/**
 * Defines a function template for training Neural Nets.

 * NeuralNet   - A Neural Net that meets the rule's criteria.
 * NetTrainKit - A set of parameters used for network training.
 */
typedef void (*NetTrainRule)(NeuralNet, NetTrainKit);

/**
 * Applies the supervised version of the Hebbian rule to a Neural Net to train. 
 * This method modifies the intensity of connections between active inputs and
 * active outputs in order to categorize inputs.
 *
 * precondition: The given Neural Network should only have one layer.
 */
void supervisedHebbRuleTrain(NeuralNet net, NetTrainKit kit);

void deltaRuleTrain(NeuralNet net, NetTrainKit kit);

void backpropagationTrain(NeuralNet net, NetTrainKit kit);

/**
 * Applies unsupervised Hebbian rule to a Neural Network. This
 * causes activation of neurons links that will increase the
 * network output. This causes particular network inputs to become
 * associated with a positive network output. Since the rule is
 * unsupervised, it does not require output vectors in the training data.
 *
 * If learning rate is equal to decay, the rule becomes Instar.
 *
 * precondition: The given Neural Network should only have one layer.
 */
void unsupervisedHebbRuleTrain(NeuralNet net, NetTrainKit kit);

/**
 * Uses the Kohonen competitive training rule to train a Neural
 * Network. This rule is unsupervised and therefore does not
 * require training outputs.
 */
void kohonenTrain(NeuralNet net, NetTrainKit kit);

#endif

