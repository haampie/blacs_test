#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <optional>
#include <array>
#include <algorithm>

#include <mpi.h>

#include "linalg.h"

// gets MPI_Comm from the grid blacs context
MPI_Comm get_communicator(const int grid_context) {
    int comm_context;
    Cblacs_get(grid_context, 10, &comm_context);
    MPI_Comm comm = Cblacs2sys_handle(comm_context);
    return comm;
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    int proc_rows = 2, proc_cols = 2;
    int ctxt, myid, myrow, mycol, numproc;

    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    Cblacs_gridinit(&ctxt, "C", proc_rows, proc_cols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    int nprow, npcol, myrow_gridinfo, mycol_gridinfo;
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow_gridinfo, &mycol_gridinfo);


    MPI_Comm comm = get_communicator(ctxt);
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (myid == 0) {
        std::cout << "=============================================" << std::endl;
        std::cout << "Output with Cblacs_pinfo and Cblacs_gridinfo:" << std::endl;
        std::cout << "=============================================" << std::endl;
    }
    for (int p = 0; p < numproc; ++p) {
        if (p == myid) {
            printf("Proc %d: %d, %d\n", p, myrow_gridinfo, mycol_gridinfo);
        }
        Cblacs_barrier(ctxt, "A");
    }

    if (rank == 0) {
        std::cout << "=============================================" << std::endl;
        std::cout << "Output with MPI_Comm and Cblacs_pcoord:" << std::endl;
        std::cout << "=============================================" << std::endl;
    }
    for (int p = 0; p < numproc; ++p) {
        if (p == rank) {
            printf("Proc %d: %d, %d\n", rank, myrow, mycol);
        }
        MPI_Barrier(comm);
    }

    Cblacs_gridexit(ctxt);

    MPI_Finalize();
}

