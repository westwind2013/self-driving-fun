#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  {
    // Initializing matrices
    //
    // Measurement covariance matrix for laser
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    // Measurement covariance matrix for radar
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    // Measurement matrix for laser 
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0; 
  }

  // Process covariance matrix
  // 
  // The variance of position in x/y dimension is set to 1 as it indicates
  // the position captured at the begining is accurate. But the variance of
  // velocity in x/y dimension is unknown, so we use a large variance 1000
  // to indicate our significant lack of confidence.  
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  
  if (!is_initialized_) {
    // Set the first measurement
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      const double rho = measurement_pack.raw_measurements_[0];
      const double phi = measurement_pack.raw_measurements_[1];
      const double rho_dot = measurement_pack.raw_measurements_[2];
      double x = rho * cos(phi);
      if ( x < 0.0001 )
      {
        x = 0.0001;
      }
      double y = rho * sin(phi);
      if ( y < 0.0001 )
      {
        y = 0.0001;
      }
      const double vx = rho_dot * cos(phi);
      const double vy = rho_dot * sin(phi);
      ekf_.x_ << x, y, vx , vy;
    }
    else {
      // velocity == 0
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_ ;
    return;
  }

  ///////////// Predict //////////////
  
  const double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // State transition matrix update
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  { // Noise covariance matrix computation
    
    // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    const double noise_ax = 9.0;
    const double noise_ay = 9.0;
    const double dt_2 = dt * dt; 
    const double dt_3 = dt_2 * dt;
    const double dt_4 = dt_3 * dt;
    const double dt_4_4 = dt_4 / 4;
    const double dt_3_2 = dt_3 / 2;
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
               0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
               dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
               0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;
  }

  ekf_.Predict();

  ///////////// Update /////////////
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
  { // Radar
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else 
  { // Laser
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // Print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
