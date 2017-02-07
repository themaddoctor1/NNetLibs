#include "neuralnet.h"

#include <stdlib.h>
#include <stdio.h>

struct neuron_layer {
    Matrix W; //Non-recurrent layer weight matrix.
    Matrix R; //Recurrent layer weight matrix, if applicable
    int r; //Number of recurrences
    TransFunc f;
};

struct neural_net {
    NeuronLayer *layers;
};

NeuronLayer makeBlankNeuronLayer(int in, int out, TransFunc func) {
    return makePresetNeuronLayer(
                   makeMatrix(out, in),
                   NULL,
                   0,
                   func);
}

NeuronLayer makeBlankRecurrentLayer(int in, int out, int r, TransFunc func) {
    return makePresetNeuronLayer(
                   makeMatrix(out, in),
                   makeMatrix(out, out),
                   r,
                   func);
}

NeuronLayer makePresetNeuronLayer(Matrix W, Matrix R, int r, TransFunc func) {
    
    NeuronLayer layer = (NeuronLayer) malloc(sizeof(struct neuron_layer));

    layer->W = W;
    layer->R = R;
    layer->r = r;

    layer->f = func;
    
    return layer;
}

Matrix getLayerWeights(NeuronLayer layer) {
    return layer->W;
}

Matrix getLayerRecurrentWeights(NeuronLayer layer) {
    return layer->R;
}

int getLayerRecurrence(NeuronLayer layer) {
    return layer->r;
}

TransFunc getLayerFunc(NeuronLayer layer) {
    return layer->f;
}

void setLayerWeights(NeuronLayer layer, Matrix m) {
    layer->W = m;
}

void setLayerRecurrentWeights(NeuronLayer layer, Matrix r) {
    layer->R = r;
}

void setLayerRecurrence(NeuronLayer layer, int r) {
    layer->r = r;
}

void setLayerFunc(NeuronLayer layer, TransFunc f) {
    layer->f = f;
}

Matrix layerFunction(NeuronLayer layer, Matrix x) {
    Matrix z = layerRaw(layer, x);
    Matrix y = layer->f(z);
    freeMatrix(z);
    return y;
}

Matrix* layerRecurrentFunction(NeuronLayer layer, Matrix *xs) {
    Matrix *zs = (Matrix*) malloc((layer->r + 1) * sizeof(Matrix)); makeMatrix(xs[0]->ROWS, 1);
    int i = 0;
    while (i < layer->r) {
        Matrix tmp1 = layerRaw(layer, xs[i]);
        Matrix tmp2 = i ? mulMtrxM(layer->R, zs[i-1]) : 
                          makeMatrix(layer->R->ROWS, layer->R->COLS);
        Matrix s = addMtrx(tmp1, tmp2);
        freeMatrix(tmp1);
        freeMatrix(tmp2);

        zs[i] = layer->f(s);
        freeMatrix(s);
        i++;
    }

    zs[i] = layerFunction(layer, zs[i-1]);
    return zs;
}

Matrix layerRaw(NeuronLayer layer, Matrix x) {
    Matrix Wx = mulMtrxM(layer->W, x);
    return Wx;
}

NeuralNet makeNeuralNet(int sizes[]) {
    
    NeuralNet net = (NeuralNet) malloc(sizeof(struct neural_net));
    
    //Get the number of layers
    int size = 0;
    while(sizes[size] > 0) size++;
    
    //Make the list
    net->layers = (NeuronLayer*) malloc((size) * sizeof(NeuronLayer));

    int i = 0;
    while(i < size-1) {
        net->layers[i] = makeBlankNeuronLayer(sizes[i], sizes[i+1], NULL);
        i++;
    }
    net->layers[i] = NULL;

    return net;

}

NeuronLayer getNetLayer(NeuralNet net, int layer) {
    return net->layers[layer];
}

Matrix getNetWeights(NeuralNet net, int layer) {
    return net->layers[layer]->W;
}

int getNetDepth(NeuralNet net) {
    int depth = 0;
    while (net->layers[depth])
        depth++;

    return depth;
}

Matrix netFunction(NeuralNet net, Matrix x) {
    
    Matrix z = mulMtrxC(x, 1); //Duplicate the input
    int i = 0;
    while(net->layers[i]) {
        Matrix tmp = z;
        z = layerFunction(net->layers[i], z);
        freeMatrix(tmp);
        i++;
    }

    return z;

}

Matrix* netRecurrentFunction(NeuralNet net, Matrix *xs) {
    
    //Build duplicate of input set.
    int len = 0;
    while (xs[len]) len++;
    Matrix *zs = (Matrix*) malloc(len * sizeof(Matrix));
    int i = len;
    while (i--) {
        zs[i] = cloneMatrix(xs[i] ? xs[i] : makeMatrix(xs[0]->ROWS, xs[0]->COLS));
    }

    i = 0;
    while (net->layers[i]) {
        //If the layer is recurrent, apply recurrent properties to it.
            Matrix *tmp = zs;
        if (net->layers[i]->R) {
            zs = layerRecurrentFunction(net->layers[i], zs);
        } else {
            zs = (Matrix*) malloc(len * sizeof(Matrix));
            
            //Compute each time epoch
            int k = len;
            while (k--)
                zs[k] = layerFunction(net->layers[i], tmp[k]);
        }
        
        //Deallocation
        int j = len;
        while (j--)
            freeMatrix(tmp[j]);
        free(tmp);

        i++;
    }
    
    return zs;

}


