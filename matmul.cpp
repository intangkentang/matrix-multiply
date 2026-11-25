#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>

using namespace std;

// Sequential Matrix Multiplication
void multiplySequential(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Parallel Matrix Multiplication using OpenMP
void multiplyParallel(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: ./matmul <N_or_input_file> <num_threads> [print_matrix: 0 or 1]\n";
        return 1;
    }

    string source = argv[1];
    int num_threads = stoi(argv[2]);
    bool print_matrix = (argc >= 4 && stoi(argv[3]) == 1);

    int N = 0;
    vector<double> A, B;

    // Check if input is a text file or a matrix dimension N
    ifstream infile(source);
    if (infile.good()) {
        infile >> N;
        A.resize(N * N);
        B.resize(N * N);
        for (int i = 0; i < N * N; ++i) infile >> A[i];
        for (int i = 0; i < N * N; ++i) infile >> B[i];
        infile.close();
    } else {
        N = stoi(source);
        A.resize(N * N);
        B.resize(N * N);
        srand(42); // Fixed seed ensures identical input matrices across runs
        for (int i = 0; i < N * N; ++i) {
            A[i] = (rand() % 10) + 1;
            B[i] = (rand() % 10) + 1;
        }
    }

    vector<double> C_seq(N * N, 0.0);
    vector<double> C_par(N * N, 0.0);

    // 1. Run & Time Sequential
    double start_seq = omp_get_wtime();
    multiplySequential(A, B, C_seq, N);
    double time_seq = omp_get_wtime() - start_seq;

    // 2. Run & Time Parallel
    double start_par = omp_get_wtime();
    multiplyParallel(A, B, C_par, N, num_threads);
    double time_par = omp_get_wtime() - start_par;

    // 3. Verify Correctness
    bool valid = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C_seq[i] - C_par[i]) > 1e-6) {
            valid = false;
            break;
        }
    }

    // 4. Print Output
    cout << "Matrix Dimension (N x N) : " << N << " x " << N << "\n";
    cout << "OpenMP Threads Used      : " << num_threads << "\n";
    cout << fixed << setprecision(6);
    cout << "Sequential Execution Time: " << time_seq << " seconds\n";
    cout << "Parallel Execution Time  : " << time_par << " seconds\n";
    cout << "Result Verification      : " << (valid ? "PASSED (Outputs Match)" : "FAILED") << "\n";

    if (print_matrix && N <= 10) {
        cout << "\nResultant Matrix C (Parallel Output):\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << setw(6) << (int)C_par[i * N + j] << " ";
            }
            cout << "\n";
        }
    }

    return 0;
}