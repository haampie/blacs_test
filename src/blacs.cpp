#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <optional>
#include <array>
#include <algorithm>

#include <mpi.h>

#include "linalg.h"

template<class T>
struct Matrix {
    std::vector<T> data;
    std::array<int, 2> extends;

    Matrix(int m, int n) : data(n * m), extends({m, n}){}

    T const &operator()(int i, int j) const {
        return data[extends[0] * j + i];
    }

    T &operator()(int i, int j) {
        return data[extends[0] * j + i];
    }

    int size(int dim) const {
        return extends[dim];
    }
};

Matrix<double> from_stream(std::istream &istr) {
    int m, n;
    istr >> m;
    istr >> n;

    Matrix<double> matrix(m, n);

    for (int y = 0; y < matrix.size(0); ++y)
        for (int x = 0; x < matrix.size(1); ++x)
            istr >> matrix(y, x);

    return matrix;
}

template<class T>
std::ostream &operator<<(std::ostream &os, Matrix<T> const &mat) {
    for (int y = 0; y < mat.size(0); ++y) {
        for (int x = 0; x < mat.size(1); ++x)
            os << std::setw(3) << mat(y, x);
        os << '\n';
    }

    return os;
}

int main(int argc, char ** argv) {
    if (argc != 4)
        return -1;

    int mpirank;
    int row_block_size = 2, col_block_size = 3;
    int proc_rows = 2, proc_cols = 2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    bool mpiroot = mpirank == 0;
    int ctxt, myid, myrow, mycol, numproc;

    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    Cblacs_gridinit(&ctxt, "C", proc_rows, proc_cols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    // Read the whole matrix locally into memory on the root process
    auto full_matrix = [=]() -> std::optional<Matrix<double>> {
        if (!mpiroot)
            return {};

        std::ifstream input{argv[1]};
        return from_stream(input);
    }();

    // Copy over the total matrix size to all processors
    int dimensions[2];

    if (mpiroot) {
        dimensions[0] = full_matrix->size(0);
        dimensions[1] = full_matrix->size(1);
    }

    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int m = dimensions[0], n = dimensions[1];

    // Initialize the local sub matrix
    int iZERO = 0;
    int nrows = numroc_(&m, &row_block_size, &myrow, &iZERO, &proc_rows);
    int ncols = numroc_(&n, &col_block_size, &mycol, &iZERO, &proc_cols);

//    for (int p = 0; p < 4; ++p) {
//        if (mpirank == p)
//            std::printf("%d x %d\n", nrows, ncols);
//        Cblacs_barrier(ctxt, "A");
//    }

    Matrix<double> local_matrix(nrows, ncols);

    // Distribute the global matrix
    int send_row = 0, send_col = 0, receive_row = 0, receive_col = 0;

    // Loop over the blocks in each column
    for (int c = 0; c < n; c += col_block_size) {
        int cols_to_send = std::min({n - c, col_block_size});

        receive_row = 0;
     
        for (int r = 0; r < m; r += row_block_size) {
            int rows_to_send = std::min({m - r, row_block_size});

            //if (mpiroot) {
                //std::cout << r << ' ' << c << ": " << rows_to_send << 'x' << cols_to_send << " to " << send_row << ' ' << send_col << ". " << std::flush;
            //}

            if (mpiroot) {
                Cdgesd2d(ctxt, rows_to_send, cols_to_send, &full_matrix->operator()(r, c), full_matrix->size(0), send_row, send_col);
            }
     
            // Receive
            if (myrow == send_row && mycol == send_col) {
                Cdgerv2d(ctxt, rows_to_send, cols_to_send, &local_matrix(receive_row, receive_col), local_matrix.size(0), 0, 0);
                receive_row += rows_to_send;
            }

            send_row = (send_row + 1) % proc_rows;
        }

        // At the end of the column, increment the col index if it's your current
        if (mycol == send_col)
            receive_col += cols_to_send;

        send_col = (send_col + 1) % proc_cols;
    }

    Cblacs_barrier(ctxt, "A");

    for (int p = 0; p < 4; ++p) {
        if (mpirank == p)
            std::cout << "Proc " << p << '\n' << local_matrix << std::endl;
        Cblacs_barrier(ctxt, "A");
    }

    MPI_Finalize();
}

