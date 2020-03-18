#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <optional>
#include <array>
#include <algorithm>

#include <mpi.h>

#include "linalg.h"

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    int row_block_size = 2, col_block_size = 3;
    int proc_rows = 2, proc_cols = 2;
    int ctxt, myid, myrow, mycol, numproc;

    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    Cblacs_gridinit(&ctxt, "R", proc_rows, proc_cols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    for (int p = 0; p < numproc; ++p) {
        if (p == myid)
            printf("Proc %d: %d, %d\n", p, myrow, mycol);
        Cblacs_barrier(ctxt, "A");
    }

    Cblacs_gridexit(ctxt);

    MPI_Finalize();
}

