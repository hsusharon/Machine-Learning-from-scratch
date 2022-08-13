#include <stdio.h>
#include <stdlib.h>
#include "data_handler.h"
#include "neural_network.h"
#include <math.h>

/*
This is a modified version of a youtube tutorial
https://www.youtube.com/watch?v=LA4I3cWkp1E&ab_channel=NicolaiNielsen-ComputerVision%26AI
Instead of a small structure, we now use the MNist dataset to train a feed forward model
Activation function: ReLu, Cost function: Cross Entropy
Epoch: 1000, Learning rate:0.01
*/

int main(){

    // Read Training Data
    int train_size = 60000, img_size = 28;
    int pixel_per_img = img_size * img_size;

    char training_data_path[] = "./train-images.idx3-ubyte";
    unsigned char **training_data;
    training_data = create_2D_arr(pixel_per_img, train_size);
    read_feature_vector(training_data, training_data_path);

    //Read Training Labels
    char training_label_path[] = "./train-labels.idx1-ubyte";
    unsigned char *training_label;
    training_label = create_1D_arr(train_size);
    read_feature_label(training_label, training_label_path);
    printf("------ Loaded training dataset ------\n\n");
 
    //Read Testing Data
    int test_size = 10000;
    char test_data_path[] = "./t10k-images.idx3-ubyte";
    unsigned char **testing_data;
    testing_data = create_2D_arr(pixel_per_img, test_size);
    read_feature_vector(testing_data, test_data_path);

    //Read Testing Lables
    char test_label_path[] = "./t10k-labels.idx1-ubyte";
    unsigned char *testing_label;
    testing_label = create_1D_arr(test_size);
    read_feature_label(testing_label, test_label_path); 
    printf("------ Loaded testing dataset ------\n\n");

    // Count the number of classes
    unsigned char *label_counter;
    label_counter = count_classes(label_counter, training_label, train_size);
    int classes = 10;

    // Separate Training and Validation data
    float VALID_PERCENT = 0.1;
    int valid_size = (int)train_size*VALID_PERCENT;
    unsigned char **valid_data, **train_data;
    valid_data = create_2D_arr(pixel_per_img, valid_size);
    train_data = create_2D_arr(pixel_per_img, train_size-valid_size);
    get_validation_data(training_data, train_data, valid_data, valid_size, pixel_per_img);

    unsigned char *train_label, *valid_label;
    train_label = create_1D_arr(train_size-valid_size);
    valid_label = create_1D_arr(valid_size);
    get_validation_label(training_label, train_label,  valid_label, valid_size);
    train_size = train_size - valid_size;

    printf("\n--------------------------------------------------------------\n");
    // train_data valid_data testing_data // train_label valid_label testing_label
    printf("Training size:%d  Validation size:%d  Testing size:%d \n", train_size, valid_size, test_size);
    printf("--------------------------------------------------------------\n\n");
    
    free(training_data); free(training_label);

    //convert unsigned char to double
    double **trainData, **validData, **testData;
    double *trainLabel, *validLabel, *testLabel;
    trainData = unsigned_char2double_2D(train_data, train_size, pixel_per_img);
    validData = unsigned_char2double_2D(valid_data, valid_size, pixel_per_img);
    testData = unsigned_char2double_2D(testing_data, test_size, pixel_per_img);
    trainLabel = unsigned_char2double_1D(train_label, train_size);
    validLabel = unsigned_char2double_1D(valid_label, valid_size);
    testLabel = unsigned_char2double_1D(testing_label, test_size); 
    free(testing_data); free(testing_label);
    free(valid_data); free(train_data);
    free(train_label); free(valid_label);

    //Create model
    double lr; //learning rate
    int epochNum;  // Epoch number
    int show_progress = 1;  // print progress in console
    int training_on = 1, XOR = 0;
    model ML_model;

    if(training_on){  // Model training mode
        
        if(XOR){   //Test XOR problem to make sure model is working correctly
            
            //Specify the training inputs
            lr = 0.01;
            epochNum = 10000;
            double **training_inputs = create_2D_double_arr(4,2);
            training_inputs[0][0] = 0.0; training_inputs[0][1] = 0.0; 
            training_inputs[1][0] = 1.0; training_inputs[1][1] = 0.0; 
            training_inputs[2][0] = 0.0; training_inputs[2][1] = 1.0; 
            training_inputs[3][0] = 1.0; training_inputs[3][1] = 1.0; 
            double *training_outputs = create_1D_double_arr(4);
            training_outputs[0] = 0.0; training_outputs[1] = 1.0;
            training_outputs[2] = 1.0; training_outputs[3] = 0.0;
            
            // Train Model
            model ML_model;
            ML_model = model_train_XOR(lr, epochNum, training_inputs, training_outputs);

            // Calculate accuracy

            free(training_inputs);
            free(training_outputs);

        }else{    // Train model for MNist dataset
            //Train Model
            lr = 0.01;
            epochNum = 1;
            ML_model = trainModel(lr, epochNum, pixel_per_img, classes, show_progress, trainData, trainLabel);

            // Calculate accuracy
            accuracy(ML_model, testData, testLabel);
        }
        
    
    }else{  // Model reading mode;
        
        // read model from file

        // prediction with model

        // calculate accuracy
    }

    free(trainData); free(validData); free(testData);
    free(trainLabel); free(validLabel); free(testLabel);
    printf("----- Code end -----\n");
    return 0;
}
