//
//  main.cpp
//  best-rotation-scake
//
//  Created by Wayne Cochran on 6/22/24.
//

#include <iostream>
#include <random>
#include <set>
#include "Best3x3RotationScale.hpp"

using namespace Eigen;
using namespace std;

Matrix3d generateRandomMatrix() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);

    Matrix3d randomMatrix;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            randomMatrix(i, j) = dis(gen);

    return randomMatrix;
}

Matrix3d generateRandomOrthogonalMatrix() {
    Matrix3d randomMatrix = generateRandomMatrix();
    HouseholderQR<Matrix3d> qr(randomMatrix);
    Matrix3d Q = qr.householderQ();
    return Q;
}

Matrix3d generateRandomDiagonalMatrix() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);

    Matrix3d diagonalMatrix = Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
        diagonalMatrix(i, i) = dis(gen);

    return diagonalMatrix;
}

bool isUnitary(const Matrix3d& matrix) {
    Matrix3d identity = Matrix3d::Identity();
    Matrix3d product = matrix.transpose() * matrix;
    return product.isApprox(identity, 1e-6);
}

void randomlyAlterMatrix(Matrix3d& matrix, int n) {
    // Random number generators
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distPos(0, 8); // To select matrix positions
    uniform_real_distribution<> distMultiplier(0.95, 1.05); // To select multipliers close to 1

    // Set to store unique positions
    set<int> positions;

    // Select n unique positions
    while (positions.size() < n)
        positions.insert(distPos(gen));

    // Alter the selected positions
    for (int pos : positions) {
        int row = pos / 3;
        int col = pos % 3;
        matrix(row, col) *= distMultiplier(gen);
    }
}


int main(int argc, const char * argv[]) {

    const Eigen::Matrix3d R_truth = generateRandomOrthogonalMatrix();
    const Eigen::Matrix3d B_truth = generateRandomDiagonalMatrix();
    const Eigen::Matrix3d A = R_truth * B_truth;

    std::cout << "A =\n" << A << "\n";
    std::cout << "R_truth =\n" << R_truth << "\n";
    std::cout << "B_truth =\n" << B_truth << "\n";

    Eigen::Matrix3d R, B;
    const int iters = best3x3RotationScale(A, R, B);

    std::cout << "iterations = " << iters << "\n";
    std::cout << "R =\n" << R << "\n";
    std::cout << "B =\n" << B << "\n";

    const double err = (A - R*B).norm();
    std::cout << "error = " << err << "\n";

    constexpr int numTests = 10;
    constexpr bool verbose = false;

    std::cout << "===== RANDOM MATRIX TESTS =====\n";
    int totalIters = 0;
    double totalError = 0;
    for (int t = 0; t < numTests; t++) {
        const Eigen::Matrix3d A = generateRandomMatrix();
        Eigen::Matrix3d R, B;
        const int iters = best3x3RotationScale(A, R, B);
        totalIters += iters;
        const double err = (A - R*B).norm();
        totalError += err;

        if (!isUnitary(R))
            std::cout << "R is NOT unitary";
        if (verbose) {
            std::cout << "=============\n";
            std::cout << "iterations = " << iters << "\n";
            std::cout << "R =\n" << R << "\n";
            std::cout << "B =\n" << B << "\n";
            std::cout << "error = " << err << "\n";
        }
    }
    std::cout << "average iterations = " << double(totalIters)/numTests << "\n";
    std::cout << "average error = " << totalError/numTests << "\n";

    std::cout << "\n===== MUTATED MATRIX TESTS =====\n";
    totalIters = 0;
    totalError = 0;
    for (int t = 0; t < numTests; t++) {
        Eigen::Matrix3d R_approx = generateRandomOrthogonalMatrix();
        Eigen::Matrix3d B_approx = generateRandomDiagonalMatrix();
        randomlyAlterMatrix(R_approx, 3);
        const Eigen::Matrix3d A = R_approx * B_approx;

        Eigen::Matrix3d R, B;
        const int iters = best3x3RotationScale(A, R, B);
        totalIters += iters;
        const double err = (A - R*B).norm();
        totalError += err;

        if (!isUnitary(R))
            std::cout << "R is NOT unitary";
        if (verbose) {
            std::cout << "=============\n";
            std::cout << "A =\n" << A << "\n";
            std::cout << "R_approx =\n" << R_approx << "\n";
            std::cout << "B_approx =\n" << B_approx << "\n";
            std::cout << "iterations = " << iters << "\n";
            std::cout << "R =\n" << R << "\n";
            std::cout << "B =\n" << B << "\n";
            std::cout << "error = " << err << "\n";
        }
    }
    std::cout << "average iterations = " << double(totalIters)/numTests << "\n";
    std::cout << "average error = " << totalError/numTests << "\n";


    return 0;
}
