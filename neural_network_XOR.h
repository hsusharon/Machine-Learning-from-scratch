#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple nn to learn XOR

#define numTrainSets 4

double init_weights(){
    return ((double)rand()) / ((double)RAND_MAX);
}

double ReLu(double x){  //ReLu activation function
    if(x>0){
        return x;
    }else{
        return 0.0;
    }
}

double dReLu(double x){  // derivative of ReLu function
    if(x>0){
        return 1.0;
    }else{
        return 0.0;
    }
}

double sigmoid(double x){  // Sigmoid function
    return 1/(1+exp(-1*x));
}

double dsigmoid(double x){   // derivatiove of Sigmoid function
    return x*(1-x);
}

double shuffle(int *array, size_t n){  // shuffle the input data
    if(n>1){
        size_t i;
        for(int i=0;i<n-1;i++){
            size_t j = i + rand()/(RAND_MAX / (n-i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
} 

void train_model(int inputNodes, int hiddenLayerNodes, int outputNodes, int show_progress){
    const double lr = 0.1f;  // learning rate

    //specify the structure of the model
    double hiddenlayer[hiddenLayerNodes];
    double outputLayer[outputNodes];
    double hiddenLayerBias[hiddenLayerNodes];
    double outputLayerBias[outputNodes];
    double hiddenLayer_weight[inputNodes][hiddenLayerNodes];
    double outputLayer_weight[hiddenLayerNodes][outputNodes];

    //Specify the training inputs
    double training_inputs[numTrainSets][inputNodes];
    training_inputs[0][0] = 0.0; training_inputs[0][1] = 0.0; 
    training_inputs[1][0] = 1.0; training_inputs[1][1] = 0.0; 
    training_inputs[2][0] = 0.0; training_inputs[2][1] = 1.0; 
    training_inputs[3][0] = 1.0; training_inputs[3][1] = 1.0; 
    double training_outputs[numTrainSets][outputNodes];
    training_outputs[0][0] = 0.0; training_outputs[1][0] = 1.0;
    training_outputs[2][0] = 1.0; training_outputs[3][0] = 0.0;


    // Initialize model values
    for(int i=0;i<inputNodes;i++){
        for(int j=0;j<hiddenLayerNodes;j++){
            hiddenLayer_weight[i][j] = init_weights();  //initiate hidden-layer weight
        }
    }
    for(int i=0;i<hiddenLayerNodes;i++){
        for(int j=0;j<outputNodes;j++){
            outputLayer_weight[i][j] = init_weights();  // initiate outpu-layer weight
        }
    }
    for(int i=0;i<hiddenLayerNodes;i++){
        hiddenLayerBias[i] = init_weights();  // initiate hidden-layer bias
    }
    for(int i=0;i<outputNodes;i++){
        outputLayerBias[i] = init_weights();  // initiate output-layer bias
    }

    // Train network
    int trainingSetOrder[] = {0,1,2,3};
    int numEpoch = 10000;

    for(int epoch=0; epoch<numEpoch; epoch++){
        shuffle(trainingSetOrder, numTrainSets);  //shuffle the training dataset order

        for(int x=0;x<numTrainSets;x++){
            int idx = trainingSetOrder[x];   // the number of the index that will be used
            double input[2];
            input[0] = training_inputs[idx][0]; 
            input[1] = training_inputs[idx][1];

            //forward pass
            for(int i=0;i<hiddenLayerNodes;i++){  //intput layer -> hidden layer
                for(int j=0;j<inputNodes;j++){
                    hiddenlayer[i] = hiddenlayer[i] + (input[j] * hiddenLayer_weight[j][i]);  // node * weights
                }
                hiddenlayer[i] += hiddenLayerBias[i];  // add bias
                hiddenlayer[i] = sigmoid(hiddenlayer[i]);  // activation function
            }
            for(int i=0;i<outputNodes;i++){
                for(int j=0;j<hiddenLayerNodes;j++){  // hidden layer -> output layer
                    outputLayer[i] = outputLayer[i] + (hiddenlayer[j] * outputLayer_weight[j][i]); // node * weights
                }
                outputLayer[i] += outputLayerBias[i];  // add bias
                outputLayer[i] = sigmoid(outputLayer[i]);  // activation function
            }

            if(show_progress){
                printf("Input:%f %f Predicted:%f  Expected:%f\n", input[0], input[1], outputLayer[0], training_outputs[idx][0]);
            }

            //Back propagation
            //Compute change in output weights
            double deltaOutput[outputNodes];
            for(int j=0;j<outputNodes;j++){
                double error = training_outputs[idx][j] - outputLayer[j];
                deltaOutput[j] = error * dsigmoid(outputLayer[j]);
            }

            double deltaHidden[hiddenLayerNodes];
            for(int j=0;j<hiddenLayerNodes;j++){
                double error = 0.0;
                for(int k=0;k<outputNodes;k++){
                    error += deltaOutput[k] * outputLayer_weight[j][k];
                }
                deltaHidden[j] = error * dsigmoid(hiddenlayer[j]);
            }
            //Update weights and bias
            for(int j=0;j<outputNodes;j++){  //update output layer
                outputLayerBias[j] += deltaOutput[j];
                for(int k=0;k<hiddenLayerNodes;k++){
                    outputLayer_weight[k][j] += hiddenlayer[k] * deltaOutput[j] * lr;
                }
            }
            for(int j=0;j<hiddenLayerNodes;j++){  // update hidden layer
                hiddenLayerBias[j] += deltaHidden[j];
                for(int k=0;k<inputNodes;k++){
                    hiddenLayer_weight[k][j] += training_inputs[idx][k] * deltaHidden[j] * lr;
                }
            }


        }

    }

}
