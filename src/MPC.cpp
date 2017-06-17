#include "MPC.h"
#include <math.h>
#include <time.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

using CppAD::AD;

const double Lf = 2.67;


// Will be filled in by the MPC class
size_t x_start;
size_t y_start;
size_t psi_start;
size_t v_start;
size_t cte_start;
size_t epsi_start;
size_t delta_start;
size_t a_start;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
class FG_eval {
private:
  int N_;
  double dt_;
  Configuration& config_;
public:
  Eigen::VectorXd coeffs_;


  // Coefficients of the fitted polynomial.
  FG_eval(Eigen::VectorXd& coeffs, Configuration& config) : config_(config), coeffs_(coeffs){
    coeffs_ = coeffs;

    N_ = config.solver_N;
    dt_ = config.solver_dt;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  // `fg` is a vector containing the cost and constraints.
  // `vars` is a vector containing the variable values (state & actuators).
  void operator()(ADvector& fg, const ADvector& vars) {
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;


    // The part of the cost based on the reference state.
    // std::cout<<"Ref v : "<<config_.ref_v<<std::endl;
    for (int i = 0; i < N_; i++) {
      fg[0] += config_.w_cte*CppAD::pow(vars[cte_start + i] - config_.ref_cte, 2);
      fg[0] += config_.w_epsi*CppAD::pow(vars[epsi_start + i] - config_.ref_epsi, 2);
      fg[0] += config_.w_v*CppAD::pow(vars[v_start + i] - config_.ref_v, 2);
      // cout<<"Vel error : "<<vars[v_start + i] - ref_v<<endl;
    }

    // Minimize the use of actuators.
    for (int i = 0; i < N_ - 1; i++) {
      fg[0] += config_.w_delta*CppAD::pow(vars[delta_start + i], 2);
      fg[0] += config_.w_a*CppAD::pow(vars[a_start + i], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int i = 0; i < N_ - 2; i++) {
      fg[0] += config_.w_deltadot*CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[0] += config_.w_adot*CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
    }

    //
    // Setup Constraints
    //
    // Initial constraints
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int i = 0; i < N_ - 1; i++) {
      // The state at time t+1 .
      AD<double> x1 = vars[x_start + i + 1];
      AD<double> y1 = vars[y_start + i + 1];
      AD<double> psi1 = vars[psi_start + i + 1];
      AD<double> v1 = vars[v_start + i + 1];
      AD<double> cte1 = vars[cte_start + i + 1];
      AD<double> epsi1 = vars[epsi_start + i + 1];

      // The state at time t.
      AD<double> x0 = vars[x_start + i];
      AD<double> y0 = vars[y_start + i];
      AD<double> psi0 = vars[psi_start + i];
      AD<double> v0 = vars[v_start + i];
      AD<double> cte0 = vars[cte_start + i];
      AD<double> epsi0 = vars[epsi_start + i];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + i];
      AD<double> a0 = vars[a_start + i];

      AD<double> f0 = polyeval_cppad(coeffs_, x0);
      AD<double> psides0 = CppAD::atan(polyeval_slope_cppad(coeffs_, x0));

      // The idea here is to constraint this value to be 0.
      fg[2 + x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt_);
      fg[2 + y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt_);
      fg[2 + psi_start + i] = psi1 - (psi0 + v0 * delta0 / Lf * dt_);
      fg[2 + v_start + i] = v1 - (v0 + a0 * dt_);
      fg[2 + cte_start + i] =
          cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt_));
      fg[2 + epsi_start + i] =
          epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt_);
    }
  }
};

//
// MPC class definition
//

MPC::MPC(Configuration& config): config_(config) {}
MPC::~MPC() {}

void MPC::Init(Eigen::VectorXd x0)
{
  size_t i;
  cout<<"Initializing optimizer..."<<endl;
  double x = x0[0];
  double y = x0[1];
  double psi = x0[2];
  double v = x0[3];
  double cte = x0[4];
  double epsi = x0[5];

  N_ = config_.solver_N;
  dt_ = config_.solver_dt;

  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = N_ * 6 + (N_ - 1) * 2;
  // Number of constraints
  size_t n_constraints = N_ * 6;

  // options
  stringstream option_maker;


  option_maker << "Integer print_level  0\n";
  option_maker << "Sparse  true        forward\n";
  option_maker << "Sparse  true        reverse\n";
  option_maker << "Numeric max_cpu_time          ";
  option_maker << fixed << setprecision(2) << config_.solver_timeout;
  option_maker << "\n";
  options_ = option_maker.str();

  x_start = 0;
  y_start = x_start + N_;
  psi_start = y_start + N_;
  v_start = psi_start + N_;
  cte_start = v_start + N_;
  epsi_start = cte_start + N_;
  delta_start = epsi_start + N_;
  a_start = delta_start + N_ - 1;

  // Initial value of the independent variables.
  // Should be 0 except for the initial values.
  vars_ = Dvector(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars_[i] = 0.0;
  }

  // Lower and upper limits for x
  vars_lowerbound_ = Dvector(n_vars);
  vars_upperbound_ = Dvector(n_vars);

  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound_[i] = -1.0e19;
    vars_upperbound_[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  const double STEERING_MAG = config_.p_steering_limit;
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound_[i] = -STEERING_MAG;
    vars_upperbound_[i] = +STEERING_MAG;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound_[i] = -1.0;
    vars_upperbound_[i] = 1.0;
  }

  // Lower and upper limits for constraints
  // All of these should be 0 except the initial
  // state indices.
  constraints_lowerbound_ = Dvector(n_constraints);
  constraints_upperbound_ = Dvector(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound_[i] = 0;
    constraints_upperbound_[i] = 0;
  }

  // Initialize saved control vector
  last_control_ = Dvector(2);
  last_control_[0] = 0;
  last_control_[1] = 0;

  cout<<"Initialization complete."<<endl;
  is_initialized_ = true;
}

MPC_OUTPUT MPC::Solve(Eigen::VectorXd x0, Eigen::VectorXd coeffs) {
  if(!is_initialized_)
  {
    Init(x0);
  }

  double x = x0[0];
  double y = x0[1];
  double psi = x0[2];
  double v = x0[3];
  double cte = x0[4];
  double epsi = x0[5];

  // Apply Lag compensation
  double dt = config_.p_lag;
  double delta = last_control_[0];
  double a = last_control_[1];

  x = x + v*cos(psi)*dt;
  y = y + v*sin(psi)*dt;
  psi = psi + v*delta/Lf *dt;
  v = v + a*dt;
  cte = cte + (v * sin(epsi) * dt);
  epsi = epsi + v * delta / Lf * dt;

  vars_[x_start] = x;
  vars_[y_start] = y;
  vars_[psi_start] = psi;
  vars_[v_start] = v;
  vars_[cte_start] = cte;
  vars_[epsi_start] = epsi;

  constraints_lowerbound_[x_start] = x;
  constraints_lowerbound_[y_start] = y;
  constraints_lowerbound_[psi_start] = psi;
  constraints_lowerbound_[v_start] = v;
  constraints_lowerbound_[cte_start] = cte;
  constraints_lowerbound_[epsi_start] = epsi;

  constraints_upperbound_[x_start] = x;
  constraints_upperbound_[y_start] = y;
  constraints_upperbound_[psi_start] = psi;
  constraints_upperbound_[v_start] = v;
  constraints_upperbound_[cte_start] = cte;
  constraints_upperbound_[epsi_start] = epsi;

  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs, config_);

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options_, vars_, vars_lowerbound_, vars_upperbound_, constraints_lowerbound_,
      constraints_upperbound_, fg_eval, solution);

  //
  // Check some of the solution values
  //
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  auto cost = solution.obj_value;
  // std::cout << "Cost " << cost << std::endl;

  time_ctr = 0;
  auto sol_x = solution.x;
  vector<double> next_x_vals(sol_x.data()+x_start+1, sol_x.data()+y_start);
  vector<double> next_y_vals(sol_x.data()+y_start+1, sol_x.data()+psi_start);

  double steering = sol_x[delta_start];
  double throttle = sol_x[a_start];

  auto output = MPC_OUTPUT(steering, throttle, next_x_vals, next_y_vals);

  last_sol_ = solution;
  last_control_[0] = steering;
  last_control_[1] = throttle;

  return output;
}
