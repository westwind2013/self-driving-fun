#include "PID.h"

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
}

void PID::UpdateError(double cte) {
  /**
   * Update PID errors based on cte.
   */
  p_error = cte;
  d_error = p_error - prev_p_error;
  i_error += cte;

  prev_p_error = p_error;
}

double PID::TotalError() {
  /**
   * Calculate and return the total error
   */
  return p_error * Kp + i_error * Ki + d_error * Kd;
}
