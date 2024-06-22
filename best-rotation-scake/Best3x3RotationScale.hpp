//
//  Best3x3RotationScale.hpp
//  best-rotation-scake
//
//  Created by Wayne Cochran on 6/22/24.
//

#ifndef Best3x3RotationScale_hpp
#define Best3x3RotationScale_hpp

#include <Eigen/Dense>

int best3x3RotationScale(const Eigen::Matrix3d& A,
                         Eigen::Matrix3d& R,
                         Eigen::Matrix3d& B);

#endif /* Best3x3RotationScale_hpp */
