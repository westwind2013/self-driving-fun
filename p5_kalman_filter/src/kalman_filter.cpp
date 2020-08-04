#include "kalman_filter.h"
#include <cmath>
#include <iostream>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

namespace
{
  void debug(const MatrixXd& m, const std::string& name)
  {
    std::cout << "Matrix " << name << ": " << m.rows() << "x" << m.cols() << std::endl;
    fflush(stdout);
  }

  void debug(const VectorXd& v, const std::string& name)
  {
    std::cout << "Vector " << name << ": " << v.size() << std::endl;
    fflush(stdout);
  }
} // namespace

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_ ;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
  /*
  debug(H_, "H");
  debug(P_, "P");
  debug(R_, "R");
  fflush(stdout);
  */

  // Calculate the intermediate terms 
  const VectorXd y = z - H_ * x_;
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K =  P_ * Ht * S.inverse();
  
  /*
  debug(K, "K");
  debug(y, "y");
  fflush(stdout);
  */

  // Update
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;  
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
  /*
  debug(H_, "H");
  debug(P_, "P");
  debug(R_, "R");
  fflush(stdout);
  */
  
  // Convert cartesian coordinate to polar coordinate 
  const double px = x_(0);
  const double py = x_(1);
  const double vx = x_(2);
  const double vy = x_(3);
  const double rho = sqrt(px*px + py*py);
  const double theta = atan2(py, px);
  const double rho_dot = (px*vx + py*vy) / rho;
  VectorXd x_polar = VectorXd(3);
  x_polar << rho, theta, rho_dot;
  
  // Obtain the difference
  VectorXd y = z - x_polar;
  while (y(1) > M_PI)
  {
    y(1) -= M_PI;
  }
  while (y(1) < -M_PI)
  {
    y(1) += M_PI;
  }

  // Calculate the intermediate terms 
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K =  P_ * Ht * S.inverse();
  
  // Update
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;  
}
