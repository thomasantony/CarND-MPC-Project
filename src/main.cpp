#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <tuple>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
#include "INIReader.h"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Transforms waypoints into local coordinates
tuple<vector<double>, vector<double>>
transform_points(vector<double> ptsx, vector<double> ptsy, vector<double> vehicle) {
  vector<double> out_x;
  vector<double> out_y;
  double x, y;
  for(auto i=0; i < ptsx.size(); i++)
  {
    std::tie(x, y) = local_transform(std::make_tuple(ptsx[i], ptsy[i]), vehicle);
    out_x.push_back(x);
    out_y.push_back(y);
  }
  return std::make_tuple(out_x, out_y);
}


double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Evaluate a polynomial slope.
double polyeval_slope(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i-1);
  }
  return result;
}
// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(vector<double>& xvals, vector<double>& yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  // Convert to Eigen format for math
  Eigen::VectorXd yvals_ = Eigen::VectorXd::Map(yvals.data(), yvals.size());

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals[j];
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals_);
  return result;
}

Configuration load_config(std::string filename)
{
  Configuration cfg;
  // Initialize config file parser
  INIReader reader(filename);
  if (reader.ParseError() < 0) {
    std::cout << "Can't load "<<filename<<std::endl;
    throw 1;
  }
  cfg.v_max = reader.GetReal("ref","v_mph", 25)*.447;
  cfg.ref_v = cfg.v_max;
  cfg.w_cte = reader.GetReal("weights", "w_cte", 1.0);
  cfg.w_epsi = reader.GetReal("weights", "w_epsi", 1.0);
  cfg.w_v = reader.GetReal("weights", "w_v", 1.0);

  cfg.w_delta = reader.GetReal("weights", "w_delta", 1.0);
  cfg.w_a = reader.GetReal("weights", "w_a", 1.0);
  cfg.w_deltadot = reader.GetReal("weights", "w_deltadot", 1.0);
  cfg.w_adot = reader.GetReal("weights", "w_adot", 1.0);

  cfg.solver_N = reader.GetInteger("solver", "N", 15);
  cfg.solver_dt = reader.GetReal("solver", "dt", 0.1);
  cfg.solver_timeout = reader.GetReal("solver", "timeout", 0.5);
  cfg.p_lag = reader.GetReal("parameters","lag",0.1);
  cfg.p_steering_limit = reader.GetReal("parameters","steering_limit_deg",20)*M_PI/180;
  return cfg;
}
int main(int argc, char *argv[]) {
  uWS::Hub h;

  std::string config_file;
  if(argc > 1)
  {
    config_file = argv[1];
  }else{
    config_file = "mpc_config.ini";
  }
  // Load solver configuration
  Configuration cfg = load_config(config_file);

  // MPC is initialized here!
  MPC mpc(cfg);

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    // cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          v = v*0.447; // convert to m/s

          vector<double> ptsx_rel, ptsy_rel;
          std::tie(ptsx_rel, ptsy_rel) = transform_points(ptsx, ptsy, {px, py, psi});
          auto coeffs = polyfit(ptsx_rel, ptsy_rel, 3);
          // The cross track error is calculated by evaluating at polynomial at x, f(x)
          // and subtracting y.
          double cte = polyeval(coeffs, 0.0) - 0;
          // Due to the sign starting at 0, the orientation error is -f'(x).
          // derivative of coeffs[0] + coeffs[1] * x -> coeffs[1]
          double epsi = -atan(polyeval_slope(coeffs, 0.0));
          /*
          * Calculate steeering angle and throttle using MPC.
          */
          Eigen::VectorXd state(6);
          state << 0, 0, 0, v, cte, epsi;

          auto mpc_output = mpc.Solve(state, coeffs);

          double steer_value;
          double throttle_value;
          //Display the MPC predicted trajectory
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          std::tie(steer_value, throttle_value, mpc_x_vals, mpc_y_vals) = mpc_output;

          json msgJson;
          msgJson["steering_angle"] = -steer_value/0.436332; // Normalize steering value
          msgJson["throttle"] = throttle_value;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals(ptsx_rel.data()+1, ptsx_rel.data() + ptsx_rel.size());
          vector<double> next_y_vals(ptsy_rel.data()+1, ptsy_rel.data() + ptsy_rel.size());

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";

          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
