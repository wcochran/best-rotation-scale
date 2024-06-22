//
//  Best3x3RotationScale.cpp
//  best-rotation-scake
//
//  Created by Wayne Cochran on 6/22/24.
//

#include "Best3x3RotationScale.hpp"

int best3x3RotationScale(const Eigen::Matrix3d& A,
                         Eigen::Matrix3d& R,
                         Eigen::Matrix3d& B) {
    const double epsilon = 1e-10; // XXX std::numeric_limits<double>::epsilon();
    R = B = Eigen::Matrix3d::Identity();
    double norm = (R*B - A).norm();  // Frobenius norm
    constexpr int maxIters = 10;
    for (int k = 0; k < maxIters; k++) {
        Eigen::BDCSVD<Eigen::Matrix3d> svd(A * B.transpose(),
                                           Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        const Eigen::Matrix3d RtA = R.transpose() * A;
        const Eigen::Vector3d d(RtA(0,0), RtA(1,1), RtA(2,2));
        B = d.asDiagonal();
        const double n = (R*B - A).norm();
        const bool noError = n < epsilon;
        const bool noImprovement = fabs(n - norm) < n * epsilon;
        if (noError || noImprovement)
            return k+1;
        norm = n;
    }
    return maxIters;
}
