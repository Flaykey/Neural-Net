#pragma once
#include<stdio.h>
#include<stdlib.h.>
#include<time.h>
#include<math.h>

typedef struct mat{
    int r,c;
    float**data;
} matrix;
float ReluDerivativf(float x){
    return (x > 0) ? 1 : 0;
}
float Max(float a, float b){
    return (a>b)? a : b;
}
matrix* Creatematrix(int r ,int c){
   matrix *m = (matrix*)malloc(sizeof(matrix));
    if (!m) {
        printf("Memory allocation failed for matrix struct\n");
        return NULL;
    }
    m->r = r;
    m->c = c;
    m->data = (float**)malloc(sizeof(float*) * r);
    if (!m->data) {
        printf("Memory allocation failed for row pointers\n");
        free(m);
        return NULL;
    }

    float *block = (float*)malloc(sizeof(float) * r * c);
    if (!block) {
        printf("Memory allocation failed for data block\n");
        free(m->data);
        free(m);
    return NULL;
    }

    for (int i = 0; i < r; i++) {
        m->data[i] = block + i * c;
    }
    for(int i = 0; i < m->r; i++){
        for (int j = 0; j < m->c; j++)
        {
           m->data[i][j] = 0;
        }
        
    }
    
    return m;
}
void Freematrix(matrix* m){
    if(m){
        free(m->data[0]);
        free(m->data);
        free(m);

    }
}
void Printmatrix(matrix *m){
    if(!m || !m->data) return;
    for(int i = 0; i < m->r; i++){
        for (int j = 0; j < m->c; j++)
        {
            // if(m->data[i][j] > 0.0) printf("1");
            // else printf(" ");
            printf("%f ",m->data[i][j]);
        }
        
        printf("\n");
    }
}
matrix* Addmatrix(matrix* m1, matrix* m2){
    if(m1->r != m2->r || m1->c != m2->c) return NULL;
    matrix* m = Creatematrix(m1->r,m1->c);
    for(int i = 0; i < m->r; i++){
        for (int j = 0; j < m->c; j++)
        {
           m->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
        
    }
    return m;

}
matrix* Subtractmatrix(matrix* m1, matrix* m2){
    if(m1->r != m2->r || m1->c != m2->c) return NULL;
    matrix* m = Creatematrix(m1->r,m1->c);
    for(int i = 0; i < m->r; i++){
        for (int j = 0; j < m->c; j++)
        {
           m->data[i][j] = m1->data[i][j] -m2->data[i][j];
        }
        
    }
        return m;

}
matrix* ArrayToMat(float* arr,int r, int c){
    matrix* m = Creatematrix(r,c);
    for(int i =0; i< r; i++){
        for(int j =0; j< c; j++){
            m->data[i][j] = arr[j+i*c];
        }
    }
    return m;
}
matrix* Multiplymatrix(matrix* m1, matrix* m2){
    if(m1->c != m2->r) return NULL;
    int r = m1->r, c = m2->c;
    matrix* m = Creatematrix(r,c);

    int i,j,k;

    for(i =0; i<r; i++){
        for(j =0; j<c; j++){
            m->data[i][j] = 0;
            for(k =0; k<m1->c; k++){
                m->data[i][j] += m1->data[i][k] * m2->data[k][j]    ;
            }   
        }
    }
    return m;
}
matrix* Dotmatrix(matrix* m1, matrix* m2){

    if(m1->r != m2->r || m1->c != m2->c) return NULL;
    matrix* m = Creatematrix(m1->r,m1->c);

    int i,j;

    for(i =0; i<m1->r; i++){
        for(j =0; j<m1->c; j++){
            m->data[i][j] = m1->data[i][j] * m2->data[i][j]    ;
        }   
    }
    return m;
}
matrix* Transposematrix(matrix* m){
    matrix* a = Creatematrix(m->c,m->r);
    for(int i = 0; i < m->c; i++){
        for (int j = 0; j < m->r; j++)
        {
           a->data[i][j] = m->data[j][i];
        }
        
    }
    return a;
}
void Scalematrix(float s, matrix* m){
    if (!m || !m->data) return;
    for(int i = 0; i < m->r; i++){
        for (int j = 0; j < m->c; j++)
        {
           m->data[i][j] *= s;
        }
        
    }
}
void Randommatrix(matrix* m ){
    if (!m || !m->data) return;
    for(int i = 0; i < m->r; i++){
        for (int j = 0; j < m->c; j++)
        {
           m->data[i][j] = (float)rand() / RAND_MAX;
        }
        
    }
}
matrix* Relu(matrix *m){
    matrix* a = Creatematrix(m->r,m->c);
    for(int i= 0 ;i< m->r; i++){
        for(int j= 0 ;j< m->c; j++){
            a->data[i][j] = Max(0,m->data[i][j]);
        }   
    }
    return a;
}
matrix* ReluDerivative(matrix *m){
    matrix* a = Creatematrix(m->r,m->c);
    for(int i= 0 ;i< m->r; i++){
        for(int j= 0 ;j< m->c; j++){
            a->data[i][j] = ReluDerivativf(m->data[i][j]);
        }   
    }
    return a;
}
matrix* Softmaxmatrix(matrix *m){
    matrix* a = Creatematrix(m->r, m->c);
    float max_val = m->data[0][0];
    for(int i=1;i<m->r;i++) if(m->data[i][0] > max_val) max_val = m->data[i][0];

    float sum = 0;
    for(int i=0;i<m->r;i++){
        a->data[i][0] = exp(m->data[i][0] - max_val);
        sum += a->data[i][0];
    }
    for(int i=0;i<m->r;i++){
        a->data[i][0] /= sum;
    }
    return a;
}
