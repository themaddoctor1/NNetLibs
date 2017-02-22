#include "nettrain.h"

#include "matrix.h"
#include "neuralnet.h"

#include <stdlib.h>
#include <stdio.h>

/**
 * Computes the error of a Neural Net on given data.
 *
 * conditions - Data set is non-empty and valid.
 *              Neural Net is valid and compatible with given data.
 */
Matrix computeError(NeuralNet net, Matrix* data) {
    
    int output = data[1]->ROWS;

    Matrix x = data[0]; //Input for data point
    Matrix y = data[1]; //Output for data point (expected)
    Matrix z = netFunction(net, x); //Actual output on x
    
    Matrix err = subMtrx(y, z); //The derivative of the error.
    freeMatrix(z);
    
    //Integrate the derivative.
    int j = 0;
    while(j < output) {
        double d = getMtrxVal(err, j, 0);
        setMtrxVal(err, j++, 0, d * d / 2);
    }

    return err;

}

void supervisedHebbRuleTrain(NeuralNet net, NetTrainKit kit) {
    
    int cycles = kit->maxCycles;
    Matrix **data = kit->data;
    
    double decay = kit->decay;

    while (cycles) {

        int i = 0;
        while(data[i]) {
            Matrix *unit = data[i];

            //Compute the output of the net on input x.
            Matrix x = unit[0];
            Matrix y = unit[1];

            Matrix x_t = transpose(x);
            
            //The change in weights is the product of y and x_t
            Matrix dW = mulMtrxM(y, x_t);
            freeMatrix(x_t);

            //Applies the change
            Matrix oldW = mulMtrxC(getNetWeights(net, 0), 1 - decay);
            freeMatrix(getNetWeights(net, 0));
            Matrix newW = addMtrx(dW, oldW);
            freeMatrix(oldW);
            setLayerWeights(getNetLayer(net, 0), newW);

            i++;
        }

        cycles--;
    }

}


void deltaRuleTrain(NeuralNet net, NetTrainKit kit) {
    
    Matrix **data = kit->data;
    double rate = kit->learnRate;
    int cycles = kit->maxCycles;
    TransFunc transGrad = kit->derivatives ? kit->derivatives[0] : linearTransferGradient;

    while (cycles) {

        int i = 0;
        while(data[i]) {
            Matrix *unit = data[i];
            
            Matrix x = unit[0]; //Input for data point
            Matrix y = unit[1]; //Output for data point (expected)

            Matrix z = netFunction(net, x); //Actual output on x

            Matrix err = subMtrx(y, z); //The error in the particular case
            freeMatrix(z);

            Matrix d = mulMtrxC(err, rate);

            //The gradient

            Matrix tmp = layerRaw(getNetLayer(net, 0), x);
            Matrix g = transGrad(tmp);
            freeMatrix(tmp);
            
            //The transpose of the input
            Matrix x_t = transpose(x);
            
            //Compute aEg
            tmp = mulMtrxM(d, g);

            d = mulMtrxM(err, x_t); //The change to M
            
            freeMatrix(x_t);
            freeMatrix(tmp);
            
            Matrix M = addMtrx(getNetWeights(net, 0), d); //The new M
            freeMatrix(getNetWeights(net, 0));
            setLayerWeights(getNetLayer(net, 0), M);
            
            freeMatrix(d);
            
            freeMatrix(g);
            freeMatrix(err);
            
            i++;
        }
        
        cycles--;

    }
    
}

void backpropagationTrain(NeuralNet net, NetTrainKit kit) {
    
    Matrix **data = kit->data;
    if(!kit || !net) {
        //printf("Backpropagation training could not be performed.\n");
        return;
    }

    double rate = kit->learnRate;
    double momentum = kit->momentum;
    double decay = kit->decay;
    int cycles = kit->maxCycles;

    //Layer functions and their derivatives.
    TransFunc* f = kit->functions;
    TransFunc* g = kit->derivatives;
    
    //Number of layers.
    int numLayers = getNetDepth(net);
    
    //Layers.
    NeuronLayer layer[numLayers];
    int i = numLayers;
    while (i--)
        layer[i] = getNetLayer(net, i);
    
    //Sums and outputs.
    Matrix a[numLayers];
    Matrix s[numLayers];

    //Needed derivatives.
    Matrix d[numLayers];

    //Holds the previous changes to the net weights
    Matrix dW[numLayers];
    i = numLayers;
    while (i--) {
        Matrix W = getLayerWeights(layer[i]);
        dW[i] = makeMatrix(W->ROWS, W->COLS);
    }

    Matrix tmp;

    while (cycles) {

        i = 0;
        while (data[i]) {
            Matrix *unit = data[i];
            
            //The input vector.
            Matrix x = unit[0];
            Matrix t = unit[1];
            int j;
            
            //Forward propagate the sums and outputs.
            s[0] = mulMtrxM(getLayerWeights(layer[0]), x);
            j = 0;
            while (j < numLayers - 1) {
                a[j] = f[j](s[j]);
                s[j+1] = mulMtrxM(getLayerWeights(layer[j+1]), a[j]);
                j++;
            }
            a[j] = f[j](s[j]);
            
            //Error
            Matrix dErr = subMtrx(a[j], t);
            
            Matrix grad = g[j](s[j]);
            
            d[j] = mulMtrxM(grad, dErr);
            
            freeMatrix(dErr);
            freeMatrix(grad);
            while (j--) {
                grad = g[j](s[j]);
                Matrix W_t = transpose(getLayerWeights(layer[j+1]));
                tmp = mulMtrxM(grad, W_t);
                d[j] = mulMtrxM(tmp, d[j+1]);
                
                freeMatrix(grad);
                freeMatrix(W_t);
                freeMatrix(tmp);
            }
            
            j = numLayers;
            while (j--) {
                Matrix a_t = transpose(j ? a[j-1] : x);
                tmp = mulMtrxM(d[j], a_t);
                freeMatrix(a_t);
                
                Matrix delta = mulMtrxC(tmp, rate); //-rate * dE/dW
                freeMatrix(tmp);
                
                //Apply momentum
                tmp = mulMtrxC(delta, 1 - momentum);
                a_t = mulMtrxC(dW[j], momentum);
                freeMatrix(delta);
                freeMatrix(dW[j]);

                dW[j] = addMtrx(tmp, a_t);
                freeMatrix(tmp);
                freeMatrix(a_t);
            }

            j = numLayers;
            while (j--) {
                Matrix W = getLayerWeights(layer[j]);
                Matrix decW = mulMtrxC(W, 1 - decay);
                freeMatrix(W);
                setLayerWeights(layer[j], subMtrx(decW, dW[j]));

                freeMatrix(s[j]);
                freeMatrix(a[j]);
                freeMatrix(d[j]);
                freeMatrix(decW);
            }

            i++;
        }

        cycles--;
    }

    i = numLayers;
    while (i--)
        freeMatrix(dW[i]);

}

void unsupervisedHebbRuleTrain(NeuralNet net, NetTrainKit kit) {
    
    int cycles = kit->maxCycles;
    Matrix **data = kit->data;
    
    double decay = kit->decay;

    while (cycles) {

        int i = 0;
        while(data[i]) {
            Matrix *unit = data[i];

            //Compute the output of the net on input x.
            Matrix x = unit[0];
            Matrix y = netFunction(net, x);

            Matrix x_t = transpose(x);
            
            //The change in weights is the product of y and x_t
            Matrix dW = mulMtrxM(y, x_t);
            freeMatrix(x_t);

            //Applies the change
            Matrix oldW = mulMtrxC(getNetWeights(net, 0), 1 - decay);
            freeMatrix(getNetWeights(net, 0));
            Matrix newW = addMtrx(dW, oldW);
            freeMatrix(oldW);
            setLayerWeights(getNetLayer(net, 0), newW);

            i++;
        }

        cycles--;
    }

}

void kohonenTrain(NeuralNet net, NetTrainKit kit) {
 
    int cycles = kit->maxCycles;
    Matrix **data = kit->data;

    double rate = kit->learnRate;
    double decay = kit->decay;
    Matrix W = getNetWeights(net, 0);

    while (cycles) {

        int i = 0;
        while(data[i]) {
            Matrix *unit = data[i];

            //Compute the output of the net on input x.
            Matrix x = unit[0];
            Matrix y = netFunction(net, x);

            int j = y->ROWS;
            while (getMtrxVal(y, j--, 0) <= 0);
            freeMatrix(y);

            int k = W->COLS;
            while (k--) {
                int h = W->ROWS;
                while (h--) {
                    double w = h == j
                               ? (1 - rate) * getMtrxVal(W, j, k) + rate
                               : decay * getMtrxVal(W, j, k);
                    setMtrxVal(W, h, k, w);
                }
            }

            i++;
        }

        cycles--;
    }


}


