#include <stdio.h>

void zeraMat (int (*mat), int limit) {
    /*
        Recives a adress matrix that has the same number of lines and colums and that number
        Puts 0 in all elements of this matrix
    */
    int i, j;
    for (i=0; i<(limit*limit); i++){
        *(mat+i) = 0;
    }
}

void printMat (int (*mat), int limit) {
    /*
        Recives a adress matrix that has the same number of lines and colums and that number
        Prints the entire matrix
    */
    int i;
    for (i=0; i<(limit*limit); i++){
        printf("%d" ,*(mat+i));
        if (i<((limit*limit)-1)) printf(",");
    }
}

void runMat (int (*mat), int limit) {
    /*
        Recives a adress matrix that has the same number of lines and colums and that number
        Prints all the possible cross in a matrix of that size with a class "1" at the and meaning that this particular matrix has a cross
    */
int count = 0, i, j;
    for (i=1; i<limit; i++) {
        for (j=1; j<limit; j++) {
            if (i+1 < limit && j+1 < limit) {
                mat[i * limit + j] = 1;
                mat[(i-1) * limit + j] = 1;
                mat[(i+1) * limit + j] = 1;
                mat[i * limit + (j-1)] = 1;
                mat[i * limit + (j+1)] = 1;
                printMat(mat, limit);
                printf(",1\n"); // class
                zeraMat(mat, limit);
                count++;
            }
        }
    }
}

int main () {
    // Predetermines which matrix's sizes are allowed
    int mat4[4][4], mat8[8][8], mat16[16][16], mat32[32][32], size, i;
    
    // printf("Qual o tamaho de linhas da matriz desejada?\nOpcoes: 4, 8, 16, 32\n");
    scanf("%d", &size);

    for(i=0; i<(size*size); i++){
        printf("%d,", i);
    }
    printf("cruz\n");
    
    switch (size) {
        /*
            Controls which matrix's sizes are allowed and make the magic happen
        */
        case 4:
            zeraMat(&mat4[0][0], 4);
            runMat(&mat4[0][0], 4);
            break;
        case 8:
            zeraMat(&mat8[0][0], 8);
            runMat(&mat8[0][0], 8);
            break;
        case 16:
            zeraMat(&mat16[0][0], 16);
            runMat(&mat16[0][0], 16);
            break;
        case 32:
            zeraMat(&mat32[0][0], 32);
            runMat(&mat32[0][0], 32);
            break;
         
        default: printf("Tamanho invalido!");
            break;
    }

return 0;
}