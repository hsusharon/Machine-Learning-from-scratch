#include <stdio.h>
#include <stdlib.h>


unsigned int convert_to_little_endian(const unsigned char *bytes){
    return(unsigned int) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3])); 
}


unsigned char** create_2D_arr(int col, int row){
    unsigned char **arr = (unsigned char**)malloc(row * sizeof(unsigned char *));
    if(arr == NULL){
        printf("Run out of memory\n");
        exit(0);
    }
    for(int i=0;i<row;i++){
        arr[i] = (unsigned char*)malloc(col*sizeof(unsigned char));
        if(arr[i] == NULL){
            printf("Run out of memeory");
            exit(0);
        }
    }
    return arr;
}


unsigned char* create_1D_arr(int col){
    unsigned char *arr = (unsigned char*)malloc(col * sizeof(unsigned char));
    if(arr == NULL){
        printf("Run out of memory\n");
        exit(0);
    }
    return arr;
}


void read_feature_vector(unsigned char **data,char* path){
    int header_size = 4;
    unsigned int header[header_size];
    unsigned char bytes[header_size];
    int i=0, j=0, k=0;
    FILE *fp = fopen(path, "rb");
    if(fp){
        for(i=0;i<header_size;i++){  // Read header 
            if(fread(bytes, sizeof(bytes), 1, fp)){ 
                header[i] = convert_to_little_endian(bytes);
            }
        }
        int img_num = header[1];
        int img_size = header[2]*header[3];
        printf("Total images: %d Image Size: %d \n", img_num, img_size);

        // read image
        unsigned char element[1];
        int flag = 0;
        for(i=0;i<img_num;i++){  //read each image 
            for(j=0;j<img_size;j++){  //read every pixel in the image
                if(fread(element, sizeof(element), 1, fp)){
                    data[i][j] = element[0];
                }else{
                    printf("Unable to read file until image #%d, elemnt #%d \n",i, j);
                    exit(1);
                }
            }
        }
        printf("Successfully read %d images\n", img_num);

    }else{
        printf("Error opening feature file\n");
        exit(1);
    }
}


void read_feature_label(unsigned char *data, char *path){
    int header_size = 2;
    unsigned int header[header_size];
    unsigned char bytes[4];
    int i=0, j=0, k=0;
    FILE *fp = fopen(path, "rb");

    if(fp){
        for(i=0;i<header_size;i++){  // Read header 
            if(fread(bytes, sizeof(bytes), 1, fp)){ 
                header[i] = convert_to_little_endian(bytes);
            }else{
                printf("Unable to read header\n");
                exit(1);
            }
        }
        int label_num = header[1];
        printf("Total number of labels: %d \n", label_num);

        unsigned char element[1];
        // read image
        for(i=0;i<label_num;i++){
            if(fread(element, sizeof(element), 1, fp)){
                data[i] = element[0];
            }else{
                printf("Unable to read until label #%d\n",i);
                exit(1);
            }
        }
        printf("Successfully read %d labels\n", label_num);

    }else{
        printf("Error opening label file\n");
        exit(1);
    }
}


unsigned char* count_classes(unsigned char *target_label, unsigned char *label, int length){
    int i,j;
    int *label_counter = malloc(100*sizeof(unsigned char));
    int counter = 1;
    label_counter[0] = label[0];
    for(i=1;i<length;i++){
        int flag = 0;
        for(j=0;j<counter+1;j++){
            if(label_counter[j] == label[i]){
                flag = 1;
                break;
            }
        }
        if(flag != 1){
            label_counter[counter] = label[i];
            counter++;
        }
    }
    
    target_label = malloc(counter*sizeof(unsigned int));
    for(i=0;i<counter;i++){
        target_label[i] = label_counter[i];
    }
    free(label_counter);
    printf("Total of %d classes\n", counter);

    return target_label;
}


void get_validation_data(unsigned char** training_data, unsigned char** train_data, unsigned char** valid_data, int valid_size, int img_size){
    int i=0, j=0;
    int train_size = 60000-valid_size;
    
    for(i=0;i<60000;i++){
        if(i<valid_size){
            for(j=0;j<img_size;j++){
                valid_data[i][j] = training_data[i][j];
            }
        }else{
            for(j=0;j<img_size;j++){
                train_data[i-valid_size][j] = training_data[i][j];
            }
        }
    }
}


void get_validation_label(unsigned char *training_label, unsigned char *train_label, unsigned char *valid_label, int valid_size){
    for(int i=0;i<60000;i++){
        if(i<valid_size){
            valid_label[i] = training_label[i];
        }else{
            train_label[i-valid_size] = training_label[i];
        }
    } 
}


