#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int input_size = 28*28;  //28*28
int layer1_size = 128;  //128
int layer2_size = 128;  //128
int layer3_size = 128;  //128
int output_size = 10;  //10


//define a struct for 5 layer model
typedef struct model{
    double *input;
    double *layer1;
    double *layer2;
    double *layer3;
    double *output;

    double *layer1_bias;
    double *layer2_bias;
    double *layer3_bias;
    double *output_bias;

    double **layer1_weight;
    double **layer2_weight;
    double **layer3_weight;
    double **output_weight;

    double *delta_layer1;
    double *delta_layer2;
    double *delta_layer3;
    double *delta_output;
} model;

void print_model_info(int inputsize, int layer1_size, int layer2_size, int layer3_size, int outputsize){
    printf("\n--------------------------------------------------\n");
    printf("5 Layer Feed Forward Model Created\n");
    printf("Input layer size: %d\n", inputsize);
    printf("Layer size: %d  Activation function: ReLu\n", layer1_size);
    printf("Layer size: %d  Activation function: ReLu\n", layer2_size);
    printf("Layer size: %d  Activation function: ReLu\n", layer3_size);
    printf("Output layer size: %d\n", outputsize);
    printf("---------------------------------------------------\n\n");

}

double** create_2D_double_arr(int row, int col){
    double **arr = (double**)malloc(row * sizeof(double *));
    if(arr == NULL){
        printf("Run out of memory\n");
        exit(0);
    }
    for(int i=0;i<row;i++){
        arr[i] = (double*)malloc(col * sizeof(double));
        if(arr[i] == NULL){
            printf("Run out of memeory");
            exit(0);
        }
    }
    return arr;
}

double* create_1D_double_arr(int col){
    double *arr = (double*)malloc(col * sizeof(double));
    if(arr == NULL){
        printf("Run out of memory\n");
        exit(0);
    }
    return arr;
}

double** unsigned_char2double_2D(unsigned char** arr, int row, int col){
    double **new_arr;
    new_arr = create_2D_double_arr(row, col);
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            new_arr[i][j] = (double)arr[i][j];
        }
    }
    return new_arr;
}

double* unsigned_char2double_1D(unsigned char* arr,  int col){
    double *new_arr;
    new_arr = create_1D_double_arr(col);
    for(int i=0;i<col;i++){
        new_arr[i] = (double)arr[i];
    }
    return new_arr;
}

double init_weights(){  // initialize weight
    return ((double)rand()) / ((double)RAND_MAX);
    //return 0.0;
}

int find_max_idx(double *arr, int size){
    int max_idx = 0;
    double max_val = 0.0;
    for(int i=0;i<size;i++){
        if(arr[i] >= max_val){
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

double sigmoid(double x){  // Sigmoid function
    return 1/(1+exp(-1*x));
}

double dsigmoid(double x){   // derivative of Sigmoid function
    return x*(1-x);
}

double ReLu(double x){
    if(x>0.0){
        return x;
    }else{
        return 0.0;
    }
}

double dReLu(double x){
    if(x>0.0){
        return 1.0;
    }else{
        return 0.0;
    }
}


double CrossEntropy(double *predict, double *label){
    double loss;
    for(int i=0;i<output_size;i++){
        if(predict[i] < 0.00001){
            loss += 100000 * label[i];
        }else{
            loss += -1 * log(predict[i]) * label[i];
        }
    }
    //printf("loss:%d %f\n",label, loss);
    return loss;
}

double* createLayer(int nodes){  
    double *arr;
    arr = create_1D_double_arr(nodes);
    for(int i=0;i<nodes;i++){
        arr[i] = init_weights();  // initialize weights
    }
    return arr;
}

double** createWeight(int prev_nodes, int nodes){
    double **arr;
    arr = create_2D_double_arr(prev_nodes, nodes);
    for(int i=0;i<prev_nodes;i++){
        for(int j=0;j<nodes;j++){
            arr[i][j] = init_weights();  // initialize weights
        }
    }
    return arr;
}

model forwardPass(double *img, model training_model, int input_size, int layer1_size, int layer2_size, int layer3_size, int output_size){
    for(int i=0;i<input_size;i++){
        training_model.input[i] = img[i];
    }
    // input layer -> hidden-layer1
    for(int i=0;i<layer1_size;i++){
        for(int j=0;j<input_size;j++){
            training_model.layer1[i] += training_model.input[j] * training_model.layer1_weight[j][i];
        }
        training_model.layer1[i] += training_model.layer1_bias[i];
        training_model.layer1[i] = sigmoid(training_model.layer1[i]);
    }

    //hidden layer1 -> hidden layer2
    for(int i=0;i<layer2_size;i++){
        for(int j=0;j<layer1_size;j++){
            training_model.layer2[i] += training_model.layer1[j] * training_model.layer2_weight[j][i];
        }
        training_model.layer2[i] += training_model.layer2_bias[i];
        training_model.layer2[i] = sigmoid(training_model.layer2[i]);
    }

    //hidden layer2 -> hidden layer3
    for(int i=0;i<layer3_size;i++){
        for(int j=0;j<layer2_size;j++){
            training_model.layer3[i] += training_model.layer2[j] * training_model.layer3_weight[j][i];
        }
        training_model.layer3[i] += training_model.layer3_bias[i];
        training_model.layer3[i] = sigmoid(training_model.layer3[i]);
    }

    //hidden layer3 -> output layer
    for(int i=0;i<output_size;i++){
        for(int j=0;j<layer3_size;j++){
            training_model.output[i] += training_model.layer3[j] * training_model.output_weight[j][i];
        }
        training_model.output[i] += training_model.output_bias[i];
        training_model.output[i] = sigmoid(training_model.output[i]);
    }
    return training_model;
}

model delta_val(model training_model, double label){
    int i=0, j=0;
    double *label_temp = (double*)malloc(output_size*sizeof(double));
    for(int i=0;i<output_size;i++){
        if(i == (int)label){
            label_temp[i] = 1.0;
        }else{
            label_temp[i] = 0.0;
        }
    }

    // output layer
    for(i=0;i<output_size;i++){  
        //double error = label - training_model.output[i];
        double error = CrossEntropy(training_model.output, label_temp);
        training_model.delta_output[i] = error * dsigmoid(training_model.output[i]);
    }

    //layer3 
    for(i=0;i<layer3_size;i++){
        double error = 0.0;
        for(j=0;j<output_size;j++){
            error += training_model.delta_output[j] * training_model.output_weight[i][j];
        }
        training_model.delta_layer3[i] = error * dsigmoid(training_model.layer3[i]);
    }

    // layer2 
    for(i=0;i<layer2_size;i++){
        double error = 0.0;
        for(j=0;j<layer3_size;j++){
            error += training_model.delta_layer3[j] * training_model.layer3_weight[i][j];
        }
        training_model.delta_layer2[i] = error * dsigmoid(training_model.layer2[i]);
    }

    //layer1 
    for(i=0;i<layer1_size;i++){
        double error = 0.0;
        for(j=0;j<layer2_size;j++){
            error += training_model.delta_layer2[j] * training_model.layer2_weight[i][j];
        }
        training_model.delta_layer1[i] = error * dsigmoid(training_model.layer1[i]);
    }

    free(label_temp);
    return training_model;
}

model update_model(model training_model, double lr){
    //layer3 <- output
    for(int i=0;i<output_size;i++){
        training_model.output_bias[i] += training_model.delta_output[i];
        for(int j=0;j<layer3_size;j++){
            training_model.output_weight[j][i] += training_model.layer3[j] * training_model.delta_output[i] * lr;
        } 
    }

    //layer2 <- layer3
    for(int i=0;i<layer3_size;i++){
        training_model.layer3_bias[i] += training_model.delta_layer3[i];
        for(int j=0;j<layer2_size;j++){
            training_model.layer3_weight[j][i] += training_model.layer2[j] * training_model.delta_layer3[i] * lr;
        } 
    }

    //layer1 <- layer2
    for(int i=0;i<layer2_size;i++){
        training_model.layer2_bias[i] += training_model.delta_layer2[i];
        for(int j=0;j<layer1_size;j++){
            training_model.layer2_weight[j][i] += training_model.layer1[j] * training_model.delta_layer2[i] * lr;
        } 
    }

    //input <- layer1
    for(int i=0;i<layer1_size;i++){
        training_model.layer1_bias[i] += training_model.delta_layer1[i];
        for(int j=0;j<input_size;j++){
            training_model.layer1_weight[j][i] += training_model.input[j] * training_model.delta_layer1[i] * lr;
        } 
    }

    return training_model;
}

model backProp(model training_model, double label, double lr){
    // Get delta values
    training_model = delta_val(training_model, label);

    //update weight and bias
    training_model = update_model(training_model, lr);

    return training_model;
}

model trainModel(float lr, int epochNum, int inputNodes, int outputNodes, int show_progress, double **train_data, double *train_label){
    int i=0, j=0, k=0, epoch=0;
    int training_size = 10; //54000
    
    /* Create 5 layer model 
    ---------------------------------------------------
    input size : (28*28, 1)
    layer 1: 128 nodes    Activation function: ReLu
    layer 2: 128 nodes    Activation function: ReLu
    layer 3: 128 nodes    Activation function: ReLu
    output size: (1, 1)
    ---------------------------------------------------
    */
    model MLmodel; // Create model
    MLmodel.input = createLayer(input_size);
    MLmodel.layer1 = createLayer(layer1_size); MLmodel.layer2 = createLayer(layer2_size); 
    MLmodel.layer3 = createLayer(layer3_size); MLmodel.output = createLayer(outputNodes);
    MLmodel.layer1_bias = createLayer(layer1_size); MLmodel.layer2_bias = createLayer(layer2_size);  
    MLmodel.layer3_bias = createLayer(layer3_size); MLmodel.output_bias = createLayer(outputNodes);
    MLmodel.delta_layer1 = createLayer(layer1_size); MLmodel.delta_layer2 = createLayer(layer2_size);
    MLmodel.delta_layer3 = createLayer(layer3_size); MLmodel.delta_output = createLayer(output_size);
    MLmodel.layer1_weight = createWeight(inputNodes, layer1_size);  // 784 * 128
    MLmodel.layer2_weight = createWeight(layer1_size, layer2_size);  // 128 * 128
    MLmodel.layer3_weight = createWeight(layer2_size, layer3_size);  // 128 * 128
    MLmodel.output_weight = createWeight(layer3_size, outputNodes);  //128 * 1
    print_model_info(inputNodes, layer1_size, layer2_size, layer3_size, outputNodes);
    
    printf("Model Training...\n");
    for(epoch=0;epoch<epochNum;epoch++){
        for(i=0;i<training_size;i++){
            double *input_temp = (double*)malloc(inputNodes * sizeof(double));
            for(j=0;j<inputNodes;j++){
                input_temp[j] = train_data[i][j]; // load the image into temp for forward passing
            }

            //forward passing
            MLmodel = forwardPass(input_temp, MLmodel, inputNodes, layer1_size, layer2_size, layer3_size, outputNodes);
            // if(i%1000 == 0 && show_progress){
            //     printf("Epoch:%d Predict label:%d Actual Lable:%d \n",epoch, MLmodel.output[0], (int)train_label[i]);
            // }

            //Back propagation
            MLmodel = backProp(MLmodel, train_label[i], lr);

            free(input_temp);
        }
        if(show_progress){
            printf("Epoch:%d \n",epoch+1);
        }

    }
    
    printf("Training Process End\n");

    return MLmodel;

}


double predictByModel(double *data, double label, model MLmodel){
    MLmodel = forwardPass(data, MLmodel, input_size, layer1_size, layer2_size, layer3_size, output_size);
    double prediction = MLmodel.output[0];
    
    return prediction;
}

void write_data(FILE *fp, model ML_model){

}

void accuracy(model trained_model, double **data, double *label){
    float test_size = 10.0; // 10000
    float positive = 0, negative = 0;
    for(int i=0;i<test_size;i++){
        double *data_temp = (double *)malloc(input_size * sizeof(double));
        for(int j=0;j<input_size;j++){
            data_temp[j] = data[i][j];
        } 
        double predict = predictByModel(data_temp, label[i], trained_model);

        if(predict<label[i]+0.5 && predict>label[i]-0.5){
            positive += 1.0;
        }else{
            negative += 1.0;
        }
        free(data_temp);
    }
    
    printf("Accuracy: %f Miss classified:%f\n", positive/test_size, negative/test_size);
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

model model_train_XOR(double lr, int epochNum, double **trainData, double *trainingLabel){
    model MLmodel; // Create model
    MLmodel.input = createLayer(input_size);
    MLmodel.layer1 = createLayer(layer1_size); MLmodel.layer2 = createLayer(layer2_size); 
    MLmodel.layer3 = createLayer(layer3_size); MLmodel.output = createLayer(output_size);
    MLmodel.layer1_bias = createLayer(layer1_size); MLmodel.layer2_bias = createLayer(layer2_size);  
    MLmodel.layer3_bias = createLayer(layer3_size); MLmodel.output_bias = createLayer(output_size);
    MLmodel.delta_layer1 = createLayer(layer1_size); MLmodel.delta_layer2 = createLayer(layer2_size);
    MLmodel.delta_layer3 = createLayer(layer3_size); MLmodel.delta_output = createLayer(output_size);
    MLmodel.layer1_weight = createWeight(input_size, layer1_size);  // 2 * 10
    MLmodel.layer2_weight = createWeight(layer1_size, layer2_size);  // 10 * 10
    MLmodel.layer3_weight = createWeight(layer2_size, layer3_size);  // 10 * 10
    MLmodel.output_weight = createWeight(layer3_size, output_size);  // 10 * 1
    print_model_info(input_size, layer1_size, layer2_size, layer3_size, output_size);
    
    int trainingSetOrder[] = {0,1,2,3};
    for(int epoch=0;epoch<epochNum;epoch++){
        shuffle(trainingSetOrder, 4);
        int idx;
        for(int x=0;x<4;x++){
            idx = trainingSetOrder[x];
            double *input_temp = malloc(input_size * sizeof(double));   
            input_temp[0] = trainData[idx][0];
            input_temp[1] = trainData[idx][1];

            //forward passing
            MLmodel = forwardPass(input_temp, MLmodel, input_size, layer1_size, layer2_size, layer3_size, output_size);
            //Back propagation
            MLmodel = backProp(MLmodel, trainingLabel[idx], lr);

            free(input_temp);
        }

        printf("Epoch:%d Input:%f %f Predicted:%f  Expected:%f\n", epoch, trainData[idx][0], trainData[idx][1], MLmodel.output[0], trainingLabel[idx]);

    }
    
    return MLmodel;
}

