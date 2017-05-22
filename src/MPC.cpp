#include "MPC.h"
#include <math.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

using CppAD::AD;

// We set the number of timesteps to 25
// and the timestep evaluation frequency or evaluation
// period to 0.05.
size_t N = 20;
double dt = 0.05;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Both the reference cross track and orientation errors are 0.
// The reference velocity is set to 40 mph.
double ref_cte = 0;
double ref_epsi = 0;
double ref_v = 20*.447;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;


class FG_eval {
 public:
  Eigen::VectorXd coeffs;
  // Coefficients of the fitted polynomial.
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  // `fg` is a vector containing the cost and constraints.
  // `vars` is a vector containing the variable values (state & actuators).
  void operator()(ADvector& fg, const ADvector& vars) {
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // The part of the cost based on the reference state.
    for (int i = 0; i < N; i++) {
      fg[0] += CppAD::pow(vars[cte_start + i] - ref_cte, 2);
      fg[0] += CppAD::pow(vars[epsi_start + i] - ref_epsi, 2);
      fg[0] += CppAD::pow(vars[v_start + i] - ref_v, 2);
      // cout<<"Vel error : "<<vars[v_start + i] - ref_v<<endl;
    }

    // Minimize the use of actuators.
    for (int i = 0; i < N - 1; i++) {
      fg[0] += CppAD::pow(vars[delta_start + i], 2);
      fg[0] += CppAD::pow(vars[a_start + i], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int i = 0; i < N - 2; i++) {
      fg[0] += CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[0] += CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
    }

    //
    // Setup Constraints
    //
    // NOTE: In this section you'll setup the model constraints.

    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // // Used to find relative coordinates for cte
    // ADvector vehicle(3);
    // vehicle.push_back(vars[x_start]);
    // vehicle.push_back(vars[y_start]);
    // vehicle.push_back(vars[psi_start]);
    // The rest of the constraints
    for (int i = 0; i < N - 1; i++) {
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

      // AD<double> rel_x0, rel_y0, rel_x1, rel_y1;
      // std::tie(rel_x0, rel_y0) = local_transform<AD<double>, ADvector>(make_tuple(x0, y0), vehicle);
      // std::tie(rel_x1, rel_y1) = local_transform<AD<double>, ADvector>(make_tuple(x1, y1), vehicle);

      // AD<double> f0 = coeffs[0] + coeffs[1] * rel_x0;
      // AD<double> psides0 = CppAD::atan(coeffs[1]);
      AD<double> f0 = polyeval_cppad(coeffs, x0);
      AD<double> psides0 = CppAD::atan(polyeval_slope_cppad(coeffs, x0));

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      fg[2 + x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[2 + y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[2 + psi_start + i] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[2 + v_start + i] = v1 - (v0 + a0 * dt);
      fg[2 + cte_start + i] =
          cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[2 + epsi_start + i] =
          epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};
//
// MPC class definition
//

MPC::MPC() {}
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

  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // Should be 0 except for the initial values.
  vars_ = Dvector(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars_[i] = 0.0;
  }
  // // Set the initial variable values

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
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound_[i] = -0.436332;
    vars_upperbound_[i] = 0.436332;
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
  FG_eval fg_eval(coeffs);

  // options
  std::string options;
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  options += "Numeric max_cpu_time          0.05\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars_, vars_lowerbound_, vars_upperbound_, constraints_lowerbound_,
      constraints_upperbound_, fg_eval, solution);

  //
  // Check some of the solution values
  //
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;
  // if(time_ctr > -1 && cost == 0)
  // {
  //   solution = last_sol_;
  //
  //   if(time_ctr+4<N-1){
  //     time_ctr++;
  //   }
  // }
  time_ctr = 0;
  auto sol_x = solution.x;
  vector<double> next_x_vals(sol_x.data()+x_start+1, sol_x.data()+x_start+6);
  vector<double> next_y_vals(sol_x.data()+y_start+1, sol_x.data()+y_start+6);

  double steering = sol_x[delta_start+4];
  double throttle = sol_x[a_start+4];

  // if(cost > 20.0)
  // {
  //   throttle = 0.01;
  //   steering = 0.0;
  // }
  auto output = MPC_OUTPUT(steering, throttle, next_x_vals, next_y_vals);

  last_sol_ = solution;

  return output;
}
