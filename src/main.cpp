#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

double get_accel(const bool too_close, const double ref_vel)
{
  const double ACC_MAX = .224;
  if(too_close)
  {
    return -ACC_MAX;
  }
  else if(ref_vel < 49)
  {
    return ACC_MAX;
  }
  
  return 0.0;
}


int main() 
{
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Start from lane 1
  int lane = 1;

  // Have a reference velocity to target
  double ref_vel = 0.0; // mph

  h.onMessage(
      [&map_waypoints_x, &map_waypoints_y, &map_waypoints_s, 
       &map_waypoints_dx, &map_waypoints_dy, &lane, &ref_vel]
      (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
       uWS::OpCode opCode) 
      {
        if (length <= 2 || data[0] != '4' || data[1] != '2') 
        {
          // The message is not a websocket message event if it does 
          // not starts with "42".
          // The 4 signifies a websocket message
          // The 2 signifies a websocket event
          return; 
        }

        auto s = hasData(data); 
        if (s != "") 
        {
          auto j = json::parse(s);

          string event = j[0].get<string>();
          if (event != "telemetry") 
          {
            return;
          }

          // j[1] is the data JSON object
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];

          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the 
          // same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          ///////////////////////////////////////////////////////////
          // Define a path made up of (x,y) points that the car will 
          // visit sequentially every .02 seconds
          ///////////////////////////////////////////////////////////

          const auto prev_size = previous_path_x.size();
          
          // Prevent collision with other vehicles
          if (prev_size > 0) {
            car_s = end_path_s;
          }

          // The minimum safe distance our car should stay away from 
          // other cars.
          const int MIN_DIST = 20;
          bool car_same_lane = false, car_left_lane = false, car_right_lane = false;
          bool too_close = false;

          for (int i = 0; i < sensor_fusion.size(); i++) 
          { // Go through the surrounding vehicles
            float check_d = sensor_fusion[i][6];
            double check_vx = sensor_fusion[i][3];
            double check_vy = sensor_fusion[i][4];
            double check_speed = sqrt(check_vx * check_vx + check_vy * check_vy);
            double check_car_s = sensor_fusion[i][5];

            int check_lane = -1;
            if (check_d > 0 && check_d < 4) 
            {
              check_lane = 0;
            } 
            else if (check_d > 4 && check_d < 8) 
            {
              check_lane = 1;
            } 
            else if (check_d > 8 && check_d < 12) {
              check_lane = 2;
            }

            if (check_lane == lane && check_car_s > car_s && check_car_s - car_s< MIN_DIST)
            { // The car in the same lane and in front of us is too close
              car_same_lane = true;
            }
            else if (check_lane == lane - 1 && abs(check_car_s - car_s) < MIN_DIST)
            { // The car on the left makes us unable to change to the left lane
              car_left_lane = true;
            }
            else if (check_lane == lane + 1 && abs(check_car_s - car_s) < MIN_DIST)
            { // The car on the right makes us unable to change to the right lane
              car_right_lane = true;
            }
          }

          if (car_same_lane)
          { // Change lane if a car is in front of us and another lane is free
            if(!car_left_lane && lane > 0)
            {
              lane--;
            }
            else if(!car_right_lane && lane < 2)
            {
              lane++;
            }
            else
            {
              too_close = true;
            }
          }

          // Adjust velocity based on the fact if we are too close to
          // the vehicle in front of us and at the same time we could
          // not change lane
          ref_vel += get_accel(too_close, ref_vel);

          // Create a list of widely spaced (x,y) waypoints, evenly 
          // spaced at 30m. Later we will interpolate these waypoints 
          // with a spline with more points that control needs.
          vector<double> ptsx;
          vector<double> ptsy;

          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          if (prev_size < 2) 
          { // if previous size is almost empty, use the car as starting reference.
            // Use two points that makes the path tangent to the car.
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);

            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
          }

          else 
          { // Use the previous path's endpoint as starting reference
            ref_x = previous_path_x[prev_size - 1];
            ref_y = previous_path_y[prev_size - 1];

            double ref_x_prev = previous_path_x[prev_size - 2];
            double ref_y_prev = previous_path_y[prev_size - 2];
            ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);

            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }

          // In Frenet add evenly 30m spaced points ahead of the starting reference
          const auto next_d = 2 + 4 * lane;
          vector<double> next_wp0 = getXY(car_s + 30, next_d, 
                                          map_waypoints_s, map_waypoints_x, 
                                          map_waypoints_y);
          vector<double> next_wp1 = getXY(car_s + 60, next_d, 
                                          map_waypoints_s, map_waypoints_x,
                                          map_waypoints_y);
          vector<double> next_wp2 = getXY(car_s + 90, next_d, 
                                          map_waypoints_s, map_waypoints_x, 
                                          map_waypoints_y);

          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);

          for (int i = 0; i < ptsx.size(); i++) 
          { // Shift car reference angle to 0 degrees
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;

            ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
            ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
          }

          // Create a spline.
          tk::spline spline;

          // Set (x,y) points to the spline
          spline.set_points(ptsx, ptsy);

          json msgJson;
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for ( int i = 0; i < prev_size; i++ ) 
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate how to break up spline points so that we 
          // travel at our desired reference velocity
          double target_x = 30.0;
          double target_y = spline(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);

          double x_add_on = 0;

          // Fill up the rest of our path planner after filling it with previous points, here we will always outputs 50 points
          for( int i = 1; i < 50 - previous_path_x.size(); i++ ) 
          {
            ref_vel += get_accel(too_close, ref_vel);

            double N = target_dist / (0.02 * ref_vel / 2.24);
            double x_point = x_add_on + target_x / N;
            double y_point = spline(x_point);

            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            // rotate back to normal after rotating it earlier
            x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
            y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;
          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        } 
        else 
        {
          // Manual driving
          std::string msg = "42[\"manual\",{}]";
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      }); // end h.onMessage

  h.onConnection([&h] (uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req)
      {std::cout << "Connected!!!" << std::endl;});

  h.onDisconnection([&h] (uWS::WebSocket<uWS::SERVER> ws, int code, 
      char *message, size_t length) 
      {
        ws.close();
        std::cout << "Disconnected" << std::endl;
      });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  } 
  else 
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}
