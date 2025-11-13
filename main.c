#include<raylib.h>
#include<stdio.h>
#include<time.h>
#include"include/matrix.h"

#define SCREENWIDTH 1400
#define SCREENHEIGHT 1400
#define PIXELSIZE 50

int main(){
    srand(time(0));
    FILE* fpImages = fopen("train-images.idx3-ubyte","rb");
    FILE* fplabels = fopen("train-labels.idx1-ubyte","rb");

    unsigned char img[28*28];
    unsigned char label;


    int inputNodes = 28*28;
    int hiddenNodes = 16;
    int outputNodes = 10;
    int epochs = 20;
    int trainingSample = 20000;
    float learningRate = 0.01;
    float totalLoss = 0;
    int i,j;
    matrix* inputLayer;
    matrix* hiddenLayer;
    matrix* outputLayer;
    matrix* weightsIH = Creatematrix(hiddenNodes,inputNodes);
    matrix* weightsHO = Creatematrix(outputNodes,hiddenNodes);
    matrix* biasH = Creatematrix(hiddenNodes,1);
    matrix* biasO = Creatematrix(outputNodes,1);


    Randommatrix(weightsIH);
    Randommatrix(weightsHO);

    Randommatrix(biasH);
    Randommatrix(biasO);

    for(i = 0; i< epochs; i++){
        printf("Epoch:%d\n",i+1);
        fseek(fpImages,16,SEEK_SET);
        fseek(fplabels,8,SEEK_SET);
        totalLoss = 0;
        for(j = 0; j<trainingSample; j++){
            fread(&img,1,28*28,fpImages);
            fread(&label,1,1,fplabels);

            //Forward Pass
            float input[28*28];
            for(int a = 0; a<28*28; a++){
                input[a] = (float)img[a] / 255.0f;
            }
            inputLayer = ArrayToMat(input,28*28,1);
            hiddenLayer = Multiplymatrix(weightsIH,inputLayer);
            hiddenLayer = Addmatrix(hiddenLayer, biasH);
            matrix* tmp = hiddenLayer;
            hiddenLayer = Relu(hiddenLayer);
            Freematrix(tmp);

            outputLayer = Multiplymatrix(weightsHO,hiddenLayer);
            outputLayer = Addmatrix(outputLayer,biasO);
            tmp = outputLayer;
            outputLayer = Softmaxmatrix(outputLayer);
            Freematrix(tmp);
            
            //BackPropagation
            
            float loss = -log(outputLayer->data[label][0]);
            matrix* trainingOutput = Creatematrix(outputNodes,1);
            trainingOutput->data[label][0] = 1;
            totalLoss += loss;

            matrix* dLoss = Subtractmatrix(outputLayer,trainingOutput);
            tmp = Transposematrix(hiddenLayer);
            matrix* dWeightsHO = Multiplymatrix(dLoss,tmp);
            Freematrix(tmp);
            
            tmp = Transposematrix(weightsHO);
            matrix* dHidden = Multiplymatrix(tmp,dLoss);
            Freematrix(tmp);
            
            tmp = dHidden;
            matrix* dHiddenlLayerRelu = ReluDerivative(hiddenLayer);
            dHidden = Dotmatrix(dHidden,dHiddenlLayerRelu);
            Freematrix(dHiddenlLayerRelu);
            Freematrix(tmp);
            
            tmp = Transposematrix(inputLayer);
            matrix* dWeightsIH = Multiplymatrix(dHidden,tmp);
            Freematrix(tmp);
            
            tmp = weightsHO;
            Scalematrix(learningRate,dWeightsHO);
            weightsHO = Subtractmatrix(weightsHO,dWeightsHO);
            Freematrix(tmp);
            
            tmp = biasO;
            Scalematrix(learningRate,dLoss);
            biasO = Subtractmatrix(biasO,dLoss);
            Freematrix(tmp);
            
            tmp = weightsIH;
            Scalematrix(learningRate,dWeightsIH);
            weightsIH = Subtractmatrix(weightsIH,dWeightsIH);
            Freematrix(tmp);
            
            tmp = biasH;
            Scalematrix(learningRate,dHidden);
            biasH = Subtractmatrix(biasH,dHidden);
            Freematrix(tmp);
            
            Freematrix(dLoss);
            Freematrix(dWeightsHO);
            Freematrix(dWeightsIH);
            Freematrix(dHidden);
            Freematrix(inputLayer);
            Freematrix(hiddenLayer);
            Freematrix(outputLayer);
            Freematrix(trainingOutput);
            
            
            totalLoss += loss;
            if(j == trainingSample-1){
                Printmatrix(outputLayer);
                printf("\nLabel: %d\nAvg Loss: %.6f\n", label, totalLoss/trainingSample);
            }
        }
        totalLoss /= trainingSample;
    }   

    InitWindow(SCREENWIDTH,SCREENHEIGHT,"MNIST DATASET");
    matrix* screen = Creatematrix(28*28,1);
    int xPos,yPos;
    SetTargetFPS(60);
    while(!WindowShouldClose()){
        BeginDrawing();
        ClearBackground(BLACK);
        for (int i = 0; i < 28 * 28; i++) {
            int x = (i % 28) * PIXELSIZE;
            int y = (i / 28) * PIXELSIZE;
            float value = screen->data[i][0];
            //float value = 0.5f; // example
            unsigned char v = (unsigned char)(value * 255.0f);
            Color col = (Color){ v, v, v, 255 }; 
            if(screen->data[i][0] > 0) DrawRectangle(x, y, PIXELSIZE,PIXELSIZE, col);
        }
        if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
            xPos = GetMouseX() / PIXELSIZE;
            yPos = GetMouseY() / PIXELSIZE;
            int index = yPos * 28 + xPos;
            screen->data[index][0] += 0.3;
            if(yPos > 1)
            {
                screen->data[index-28][0] += 0.1;
                screen->data[index-28][0] = (screen->data[index-28][0] > 1) ? 1 : screen->data[index-28][0];
            }
            if(yPos < 27)
            { 
                screen->data[index+28][0] += 0.1;
                screen->data[index+28][0] = (screen->data[index+28][0] > 1) ? 1 : screen->data[index+28][0];
            }
            if(xPos > 1)
            { 
                screen->data[index-1][0] += 0.1;
                screen->data[index-1][0] = (screen->data[index-1][0] > 1) ? 1 : screen->data[index-1][0];
            }
            if(xPos < 27){ 
                screen->data[index+1][0] += 0.1;
                screen->data[index+1][0] = (screen->data[index+1][0] > 1) ? 1 : screen->data[index+1][0];
            }
            
            screen->data[index][0] = (screen->data[index][0] > 1) ? 1 : screen->data[index][0];  
            
        }
        if(IsMouseButtonDown(MOUSE_BUTTON_RIGHT)){
            xPos = GetMouseX() / PIXELSIZE;
            yPos = GetMouseY() / PIXELSIZE;
            int index = yPos * 28 + xPos;
            screen->data[index][0] -= 0.3;
            if(yPos > 1)
            {
                screen->data[index-28][0] -= 0.1;
                screen->data[index-28][0] = (screen->data[index-28][0] < 0) ? 0 : screen->data[index-28][0];
            }
            if(yPos < 27)
            { 
                screen->data[index+28][0] -= 0.1;
                screen->data[index+28][0] = (screen->data[index+28][0] < 0) ? 0 : screen->data[index+28][0];
            }
            if(xPos > 1)
            { 
                screen->data[index-1][0] -= 0.1;
                screen->data[index-1][0] = (screen->data[index-1][0] < 0) ? 0 : screen->data[index-1][0];
            }
            if(xPos < 27){ 
                screen->data[index+1][0] -= 0.1;
                screen->data[index+1][0] = (screen->data[index+1][0] < 1) ? 0 : screen->data[index+1][0];
            }
            
            screen->data[index][0] = (screen->data[index][0] < 0) ? 0 : screen->data[index][0];  
            
        }
        
        hiddenLayer = Multiplymatrix(weightsIH,screen);
        hiddenLayer = Addmatrix(hiddenLayer, biasH);
        matrix* tmp = hiddenLayer;
        hiddenLayer = Relu(hiddenLayer);
        Freematrix(tmp);

        outputLayer = Multiplymatrix(weightsHO,hiddenLayer);
        outputLayer = Addmatrix(outputLayer,biasO);
        tmp = outputLayer;
        outputLayer = Softmaxmatrix(outputLayer);
        Freematrix(tmp);
        char label[16];   
        char loss[16];               

        int predicted = 0;
        float maxProb = outputLayer->data[0][0];

        for (int i = 1; i < outputLayer->r; i++) {
            if (outputLayer->data[i][0] > maxProb) {
                maxProb = outputLayer->data[i][0];
                predicted = i;
            }
        }            
        sprintf(label, "label: %d", predicted);
        sprintf(loss, "loss: %f", totalLoss);

        DrawText(label, 0, 0, 32, WHITE);
        DrawText(loss, 0, 64, 32, WHITE);

        EndDrawing();
    }


    CloseWindow();

    Freematrix(weightsIH);
    Freematrix(weightsHO);
    fclose(fpImages);
    fclose(fplabels);

    return 0;
}