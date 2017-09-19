//
// Created by nw on 19.09.17.
//

#include <iostream>
#include <vector>
#include "mpi.h"


int main(int argc, char **argv)
{
    int my_rank;    //rank of process
    int p;          //number of MPI processes
    int tag=50;     //Tag for message

    int X = 4;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    std::vector<int> rcvvec(X);
    std::vector<int> sndvec(X/p);

    int rcounts[p];
    int rdisp[p];

    for(int i=0; i<p; ++i) {
        rcounts[i] = X/p;
        rdisp[i] = i*(X/p);
    }

    for (int i = 0; i < X/p; ++i)
        sndvec[i] = my_rank+1;

    MPI_Gatherv(&sndvec.front(), rcounts[my_rank], MPI_INT, &rcvvec.front(), rcounts, rdisp, MPI_INT, 0, MPI_COMM_WORLD);

    if (!my_rank) {
        for (int i = 0; i < rcvvec.size(); ++i) {
            std::cout<<rcvvec[i]<<" ";
        } std::cout<<std::endl;
    }

    MPI_Finalize();
}