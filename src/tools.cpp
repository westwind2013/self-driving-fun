#include "tools.h"
#include <iostream>
#include <cassert>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Root mean square error
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.empty())
  {
    std::cout << "Empty input" << std::endl;
    return rmse;
  }

  if (estimations.size() != ground_truth.size())
  {
    std::cout << "Inconsistent input size" << std::endl;
    return rmse;
  }

  for (size_t i = 0; i < estimations.size(); i++)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  
  rmse = rmse / estimations.size();
  return rmse.array().sqrt();
  //return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
  // The size of state vector must be exactly 4
  assert(x_state.size() == 4);
 
  // Jacobian matrix
  MatrixXd Hj(3,4);
 
  // Calculate the intermediate terms for computing Jacobi matrix 
  const double px = x_state(0);
  const double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);
  const double c1 = (px * px) + (py * py);
  const double c2 = sqrt(c1);
  const double c3 = c1 * c2;

  // Check division by zero
  if (fabs(c1) < 0.0001) {
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return Hj;
  }

  // Compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj; 
}
