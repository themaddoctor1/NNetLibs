# Overview
This repository contains a neural network library that I wrote on my free time. The library comes with the necessary components to perform linear algebra, train neural networks, and execute them on input vectors. The library also has some support for recurrent neural networks, which still needs work.

# Installation
In the main directory, there is a makefile that can be used to make the libraries. To build, simply run ```make```. This will generate ```libnnet.a```, which can be used in your C compiler to use and compile with the libraries.

# Features
The library has several features that can be used for training and simulating neural networks. These include:

### Linear Algebra
The libraries contain source code that allows for basic linear algebra. Here is an example of a basic usage of matrices:

```
// 3x3 matrix of zeros.
Matrix m = makeMatrix(3, 3);

// Sets the value at index (1,2) to 3.
setMtrxVal(m, 1, 2, 3);

//Creates a 3x3 identity matrix
Matrix I = identityMatrix(3);

//Multiplies m and I, and stores it in mI.
Matrix mI = mulMtrxM(m, I);

//Multiplies mI by 3.
Matrix n = mulMtrxC(mI, 3);

//Adds n to mI.
Matrix n_plus_mI = addMtrx(n, mI);

//Prints a Matrix
printMatrix(I);

```

The library can also be used for vector mathematics, for instance:

```
Matrix i = makeMatrix(3, 1);
setMtrxVal(m, 0, 0, 1);

Matrix j = makeMatrix(3, 1);
setMtrxVal(m, 1, 0, 1);

Matrix j = makeMatrix(3, 1);
setMtrxVal(m, 2, 0, 1);

//Magnitude should be one.
printf("The magnitude of i is %lf\n", vecNorm(i));

//Dot product should be zero.
printf("The dot product of i and j is %lf\n", dotProd(i, j));

```

These libraries are used as the basis of the network implementation, since the theory behind neural networks is based on a linear algebra approach. Note that calling `free()` on a matrix is an unsafe operation, as it will cause a memory leak unless the contained `double*` is preserved. To properly free a matrix, use `freeMatrix(Matrix)`, which will zero and free the matrix and its contents.

### Neural Networks
One of the primary features of the library is the ability to create and modify neural networks. This is done by providing the size of the network and then modifying each layer with the desired transition functions.

The following functions and derivatives are offered:

| Name | Formula | Function |
|:---- |:-------:|:--------:|
| Linear | f(x) = x | linearTransfer(Matrix) |
| Sigmoid | f(x) = 1 / (1 + e^(-x)) | sigmoidTransfer(Matrix) |
| Unit Step | f(x) = x >= 0 ? 1 : 0 | unitStepTransfer(Matrix) |
| Competitive Function | f(x) = max(x_i) | competeTransfer(Matrix) |

These functions have header initializations in `neuralnet.h` can be viewed in `transfunc.c`. Functions can be assigned to layers using function pointers, and an appropriate pointer typdef is made available.

```
//Gets the linear transfer functions.
Transfunc func = linearTransfer;
```

Keep in mind that for some training algorithms, it is necessary to have a derivative function that returns the derivative of your function. This output should be in the form of a Jacobian matrix, since the derivative of a function of vectors is a Jacobian matrix. Several of these gradients are made available alongside the original functions in `neuralnet.h` and `transfunc.c`.

Creation of a neural net requires that the user have the sizes for a network and the functions that are needed. Consider the case in the XOR demo (see `backpropXorDemo()` in `test.c`), where a 2-layer neural network is required. In the algorithm, the first layer takes 3 inputs, returns 7 outputs, and utilizes a sigmoid transfer algorithm. The second layer takes 7 inputs and returns 1, which is returned to the user. Without worrying about the training kit, I can do the following to construct the network:

```
//List of sizes with a null terminator.
int sizes[] = {3, 7, 1, 0};

NeuralNet net = makeNeuralNet(sizes);

//I can get any layer that's in the network.
NeuronLayer layer = getNetLayer(net, 0);

//Change the layer to use the new transfer function.
setLayerFunc(layer, sigmoidTransfer);

//However, I can do them both in one line if I choose.
setLayerFunc(getNetLayer(net, 1), linearTransfer);
```

Now that I have a network, I am able to modify the layers and run the network. The library comes with setter and getter functions that allow for retrieval of the network weights and the transfer functions. One can also modify the weights of the network using the matrix functionality. The network can be run by providing an input vector (a matrix with 1 column) by calling `netFunction(NeuralNet, Matrix)`, which will return a vector in the form of a matrix.

### Training Algorithms

The library comes with several prewritten functions for training the networks, which was the original purpose of the library. The available functionality can be viewed in `nettrain.h`, which has all of the function and struct names. For any training function, one will require a network and a training kit, which can be used to tailor fit the training procedures to your needs. Not all of the algoritms will use any given field, but it is recommended that as many fields are filled as possible. Sample usage is available in `test.c`.

Here is an example of a training kit construction:

```
//Initialize the kit value.
struct nettrainkit kit;

//I add the function for a linear transfer. I need one slot for those functions.
kit.functions = (TransFunc*) malloc(2 * sizeof(TransFunc));
kit.functions[0] = linearTransfer;
kit.derivatives = (TransFunc*) malloc(2 * sizeof(TransFunc));
kit.derivatives[0] = linearTransferGradient;

//Values that may or may not be used.
kit.learnRate = 0.01;
kit.momentum = 0.05;
kit.decay = 0;

kit.maxCycles = 65536; //The number of training rounds

//Perhaps I want ten input points.
kit.data = (Matrix**) malloc(11 * sizeof(Matrix*));

//I have to null terminate.
kit.data[10] = NULL;

for (int i = 0; i < 10; i++) {
    //Allocate memory
    kit.data[i] = (Matrix*) malloc(2 * sizeof(Matrix));

    //Input vector
    Matrix x = ...
    
    //Output target vector
    Matrix t = ...
    
    kit.data[i][0] = x;
    kit.data[i][1] = t;
}

//Train with backpropagation
backpropXorTrain(network, &kit);

```

Note that `malloc()` is required to dynamically allocate memory for the kit. Furthermore, freeing the training kit is the responsibility of the user, as the kit is not modified by the training functions in order to enable reuse.

The library has several training functions for use, which are listed below.

| Algorithm | Function |
| --------- |:--------:|
| Backpropagation | backpropagationTrain() |
| Delta Rule | deltaRuleTrain() |
| Kohonen Rule | kohonenTrain() |
| Supervised Hebbian | supervisedHebbRuleTrain() |
| Unsupervised Hebbian | unsupervisedHebbRuleTrain() |

The lack of Perceptron and ADALINE are due to the fact that Delta Rule and Backpropagation can be modified to act exactly like Perceptron and ADALINE. All of these functions require a network and an appropriate training kit.

# Demos
During the creation of the network, I wrote several sample applications that build training kits and train networks on input. These can be viewed in `test.c` and `conway.c`, as well as their respective header files.

