#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "neuralnet.h"
#include "nettrain.h"

void backpropXorDemo() {
    /* The layer sizes. This indicates that we want a network with
       two layers with two inputs and one output. We use three inputs
       because one input will be a bias value that is always equal to one. */
    int sizes[] = {3, 7, 1, 0};
    
    //Builds a blank network with the given sizes.
    printf("Building %i-%i-%i network...\n", sizes[0], sizes[1], sizes[2]);
    NeuralNet network = makeNeuralNet(sizes);
    
    srand(time(NULL));

    int i = 2;
    while (i--) {
        Matrix M = getNetWeights(network, i);
        int r = M->ROWS;
        while(r--) {
            int c = M->COLS;
            while (c--) {
                double d = ((double)rand()) / ((double) RAND_MAX);
                setMtrxVal(M, r, c, 2 * d - 1);
            }
        }
    }

    printf("Initial weight matrix:\n");
    printf("Layer 0: "); printMatrix(getNetWeights(network, 0));
    printf("Layer 1: "); printMatrix(getNetWeights(network, 1));
    
    //In order to train, we need a training kit.
    printf("Allocating training kit...\n");
    NetTrainKit kit = (NetTrainKit) malloc(sizeof(struct nettrainkit));
    
    //We might need to have the transfer functions and their derivatives.
    printf("Building training kit...\n");
    kit->functions = (TransFunc*) malloc(2 * sizeof(TransFunc));
    kit->functions[0] = sigmoidTransfer;
    kit->functions[1] = linearTransfer;
    kit->derivatives = (TransFunc*) malloc(2 * sizeof(TransFunc));
    kit->derivatives[0] = sigmoidTransferGradient;
    kit->derivatives[1] = linearTransferGradient;
    
    printf("Applying transfer functions...\n");
    setLayerFunc(getNetLayer(network, 0), kit->functions[0]);
    setLayerFunc(getNetLayer(network, 1), kit->functions[1]);
    
    //We also need a learning rate, as well as a maximum number of cycles.
    kit->learnRate = 0.01;
    kit->momentum = 0.05;
    kit->decay = 0; // Decay rate is not needed.
    kit->maxCycles = 65536;

    //We will also need training data.
    printf("Building training data...\n");
    kit->data = (Matrix**) malloc(5 * sizeof(Matrix*));
    kit->data[4] = NULL;

    int p = 0;
    while (p <= 1) {
        int q = 0;
        while (q <= 1) {
            printf("Item %i of 4...\n", 2*p + q + 1);
            //Each unit of training data is a pair of matrices.
            kit->data[2*p + q] = (Matrix*) malloc(2 * sizeof(Matrix));

            //The input is size 3, so create a 3x1 input vector.
            Matrix x = makeMatrix(3, 1);
            setMtrxVal(x, 0, 0, p);
            setMtrxVal(x, 1, 0, q);
            setMtrxVal(x, 2, 0, 1);
            kit->data[2*p + q][0] = x;

            //The output is size 1, so create a 1x1 output vector.
            Matrix y = makeMatrix(1, 1);
            setMtrxVal(y, 0, 0, p ^ q); //It's value reflects our ideal result.
            kit->data[2*p + q][1] = y;
            
            q++;
        }

        p++;
    }
    
    printf("Training...\n");
    //Finally, train the Neural Net via Delta Rule using the tool kit.
    backpropagationTrain(network, kit);

    printf("Complete!\n\n");
    
    printf("W[0]: "); printMatrix(getNetWeights(network, 0));
    printf("W[1]: "); printMatrix(getNetWeights(network, 1));
    
    p = 0;
    while (p < 4) {
        Matrix res = netFunction(network, kit->data[p][0]);
        printf("\n%i XOR %i = %lf\n", p/2, p%2, getMtrxVal(res, 0, 0));
        freeMatrix(res);
        p++;
    }


}

void hebbianXODemo() {
    //We will say that images are drawn on a 5x5 canvas. This requires a net with 25 inputs.

    int sizes[] = {26, 1, 0};

    //Builds a blank network with the given sizes.
    printf("Building 26-1 network...\n");
    NeuralNet network = makeNeuralNet(sizes);

    printf("Initial weight matrix:\n");
    printMatrix(getNetWeights(network, 0));
    
    //Disregarding optimality, set the only layer to use a sigmoid function.
    printf("Setting transfer function to sigmoid...\n");
    setLayerFunc(getNetLayer(network, 0), linearTransfer);

    //In order to train, we need a training kit.
    printf("Allocating training kit...\n");
    NetTrainKit kit = (NetTrainKit) malloc(sizeof(struct nettrainkit));
    
    //We might need to have the transfer functions and their derivatives.
    printf("Building training kit...\n");
    kit->functions = (TransFunc*) malloc(sizeof(TransFunc));
    *(kit->functions) = linearTransfer;
    kit->derivatives = (TransFunc*) malloc(sizeof(TransFunc));
    *(kit->derivatives) = linearTransferGradient;

    //We also need a learning rate, as well as a maximum number of cycles.
    kit->learnRate = 0.0625; //Set the learn rate coefficient to 0.0625.
    kit->maxCycles = 1; //Max 64 cycles.
    kit->decay = 0;

    //We will also need training data.
    printf("Building training data...\n");
    kit->data = (Matrix**) malloc(3 * sizeof(Matrix*));
    kit->data[2] = NULL;
    
    //Input X
    kit->data[0] = (Matrix*) malloc(2*sizeof(Matrix));
    kit->data[0][0] = makeMatrix(26, 1);
    setMtrxVal(kit->data[0][0], 25, 0, 1);
    int i = 0;
    while (i < 5) {
        setMtrxVal(kit->data[0][0], 6*i, 0, 1);
        setMtrxVal(kit->data[0][0], 4*(i+1), 0, 1);
        i++;
    }
    kit->data[0][1] = identityMatrix(1);
    printf("X should output %lf\n", getMtrxVal(kit->data[0][1], 0, 0));
    
    //Input O
    kit->data[1] = (Matrix*) malloc(2*sizeof(Matrix));
    kit->data[1][0] = makeMatrix(26, 1);
    setMtrxVal(kit->data[1][0], 25, 0, 1);
    i = 0;
    while (i < 4) {
        setMtrxVal(kit->data[1][0], i, 0, 1);
        setMtrxVal(kit->data[1][0], 4 +5*i, 0, 1);
        setMtrxVal(kit->data[1][0], 5*(i+1), 0, 1);
        setMtrxVal(kit->data[1][0], 21+i, 0, 1);
        i++;
    }
    kit->data[1][1] = mulMtrxC(kit->data[0][1], -1);
    printf("O should output %lf\n", getMtrxVal(kit->data[1][1], 0, 0));
    
    printf("Training...\n");
    supervisedHebbRuleTrain(network, kit);
    printf("Complete!\n\n");

    printf("Input: X\n");
    Matrix m = netFunction(network, kit->data[0][0]);
    printf("Output: %lf\n\n", getMtrxVal(m, 0, 0));
    freeMatrix(m);

    printf("Input: O\n");
    m = netFunction(network, kit->data[1][0]);
    printf("Output: %lf\n\n", getMtrxVal(m, 0, 0));
    freeMatrix(m);

}

void deltaOrGateDemo() {
    //Create a single-layer network with two inputs and one output.

    /* The layer sizes. This indicates that we want a network with a
       single layer with two inputs and one output. We use three inputs
       because one input will be a bias value that is always equal to one. */
    int sizes[] = {3, 1, 0};
    
    //Builds a blank network with the given sizes.
    printf("Building 3-1 network...\n");
    NeuralNet network = makeNeuralNet(sizes);

    printf("Initial weight matrix:\n");
    printMatrix(getNetWeights(network, 0));
    
    //Disregarding optimality, set the only layer to use a sigmoid function.
    printf("Setting transfer function to sigmoid...\n");
    setLayerFunc(getNetLayer(network, 0), sigmoidTransfer);

    //In order to train, we need a training kit.
    printf("Allocating training kit...\n");
    NetTrainKit kit = (NetTrainKit) malloc(sizeof(struct nettrainkit));
    
    //We might need to have the transfer functions and their derivatives.
    printf("Building training kit...\n");
    kit->functions = (TransFunc*) malloc(sizeof(TransFunc));
    *(kit->functions) = sigmoidTransfer;
    kit->derivatives = (TransFunc*) malloc(sizeof(TransFunc));
    *(kit->derivatives) = sigmoidTransferGradient;

    //We also need a learning rate, as well as a maximum number of cycles.
    kit->learnRate = 0.0625; //Set the learn rate coefficient to 0.0625.
    kit->maxCycles = 64; //Max 64 cycles.
    kit->decay = 0;

    //We will also need training data.
    printf("Building training data...\n");
    kit->data = (Matrix**) malloc(5 * sizeof(Matrix*));
    kit->data[4] = NULL;

    int p = 0;
    while (p <= 1) {
        int q = 0;
        while (q <= 1) {
            printf("Item %i of 4...\n", 2*p + q + 1);
            //Each unit of training data is a pair of matrices.
            kit->data[2*p + q] = (Matrix*) malloc(2 * sizeof(Matrix));

            //The input is size 2, so create a 2x1 input vector.
            Matrix x = makeMatrix(3, 1);
            setMtrxVal(x, 0, 0, p);
            setMtrxVal(x, 1, 0, q);
            setMtrxVal(x, 2, 0, 1);
            kit->data[2*p + q][0] = x;

            //The output is size 1, so create a 1x1 output vector.
            Matrix y = makeMatrix(1, 1);
            setMtrxVal(y, 0, 0, p | q); //It's value reflects our ideal result.
            kit->data[2*p + q][1] = y;
            
            q++;
        }

        p++;
    }
    
    printf("Training...\n");
    //Finally, train the Neural Net via Delta Rule using the tool kit.
    deltaRuleTrain(network, kit);
    //backpropagationTrain(network, kit);

    printf("Complete!\n\n");
    
    printf("W: "); printMatrix(getNetWeights(network, 0));
    
    p = 0;
    while (p < 4) {
        printf("\nInput: "); printMatrix(kit->data[p][0]);
        printf("Expected: "); printMatrix(kit->data[p][1]);
        Matrix res = netFunction(network, kit->data[p][0]);
        printf("Yields: "); printMatrix(res);
        freeMatrix(res);
        p++;
    }

}

/**
 * As seen in Neural Network Design, 2nd Edition.
 */
void hebbianBananaDemo() {
    
    //In our example, we have two inputs: the sight of a
    //banana (unconditioned) and the smell (conditioned).
    //The output is the response to the banana.
    int sizes[] = {2, 1, 0};

    //Builds a blank network with the given sizes.
    printf("Building 3-1 network...\n");
    NeuralNet network = makeNeuralNet(sizes);
    
    setMtrxVal(getNetWeights(network, 0), 0, 0, 1);
    setMtrxVal(getNetWeights(network, 0), 0, 1, 0);
    
    printf("Initial weight matrix:\n");
    printMatrix(getNetWeights(network, 0));
    
    //Disregarding optimality, set the only layer to use a sigmoid function.
    printf("Setting transfer function to linear...\n");
    setLayerFunc(getNetLayer(network, 0), linearTransfer);

    //In order to train, we need a training kit.
    printf("Allocating training kit...\n");
    NetTrainKit kit = (NetTrainKit) malloc(sizeof(struct nettrainkit));
    
    //We might need to have the transfer functions and their derivatives.
    printf("Building training kit...\n");
    kit->functions = (TransFunc*) malloc(sizeof(TransFunc));
    *(kit->functions) = linearTransfer;
    kit->derivatives = (TransFunc*) malloc(sizeof(TransFunc));
    *(kit->derivatives) = linearTransferGradient;

    //We also need a learning rate, as well as a maximum number of cycles.
    //kit->learnRate = 0.0625; //Set the learn rate coefficient to 0.0625.
    kit->maxCycles = 4; //Max 64 cycles.
    kit->decay = 0.5;

    //We will also need training data. The first index will be the sight,
    //and the second will be the smell. The network is preconfigured to
    //respond to the sight, so that the association can be made.
    printf("Building training data...\n");
    kit->data = (Matrix**) malloc(2 * sizeof(Matrix*));
    kit->data[1] = NULL;
      
    Matrix before = makeMatrix(2, 1);
    setMtrxVal(before, 0, 0, 1); //Already conditioned stimulus
    setMtrxVal(before, 1, 0, 0); //What we want to condition.

    Matrix during = makeMatrix(2, 1);
    setMtrxVal(during, 0, 0, 1); //Already conditioned stimulus
    setMtrxVal(during, 1, 0, 1); //What we want to condition.

    Matrix after = makeMatrix(2, 1);
    setMtrxVal(after, 0, 0, 0); //Already conditioned stimulus
    setMtrxVal(after, 1, 0, 1); //What we want to condition.
    
    kit->data[0] = (Matrix*) malloc(2 * sizeof(Matrix));
    kit->data[0][0] = during;
    
    Matrix y;
    
    //Since the sight (index 0) is already conditioned,
    //the result should be a positive number.
    printf("Unconditioned stimulus only (before): ");
    y = netFunction(network, before);
    printf("%lf\n", getMtrxVal(y, 0, 0));
    freeMatrix(y);

    //Since the smell (index 1) is not conditioned,
    //the result should be zero.
    printf("Conditioned stimulus only (before): ");
    y = netFunction(network, after);
    printf("%lf\n", getMtrxVal(y, 0, 0));
    freeMatrix(y);

    printf("\nTraining...\n");
    unsupervisedHebbRuleTrain(network, kit);
    printf("Complete!\n\n");

    //The original stimulus should still be present.
    printf("Unconditioned stimulus only (after): ");
    y = netFunction(network, before);
    printf("%lf\n", getMtrxVal(y, 0, 0));
    freeMatrix(y);

    //However, the new stimulus should now yield a positive
    //number, since it now associates the smell with bananas.
    printf("Conditioned stimulus only (after): ");
    y = netFunction(network, after);
    printf("%lf\n", getMtrxVal(y, 0, 0));
    freeMatrix(y);
}





