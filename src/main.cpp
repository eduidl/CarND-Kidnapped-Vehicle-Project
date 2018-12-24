#include <algorithm> // copy
#include <array>
#include <iostream> // cout, endl
#include <iterator> // istream_iterator, back_inserter
#include <numeric>  // accumulate
#include <sstream>  // istringstream
#include <string>
#include <vector>

#include "json.hpp"
#include "particle_filter.h"
#include <uWS/uWS.h>

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  const auto found_null = s.find("null");
  const auto b1 = s.find_first_of("[");
  const auto b2 = s.find_first_of("]");
  if (found_null != std::string::npos) {
    return "";
  } else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }

  return "";
}

int main() {
  using namespace std::string_literals;

  uWS::Hub h;

  // Set up parameters here
  const double delta_t = 0.1;     // Time elapsed between measurements [sec]
  const double sensor_range = 50; // Sensor range [m]

  // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  const std::array<double, 3> sigma_pos{0.3, 0.3, 0.01};
  // Landmark measurement uncertainty [x [m], y [m]]
  const std::array<double, 2> sigma_landmark{0.3, 0.3};

  // Read map data
  Map map;
  if (!read_map_data("../data/map_data.txt", map)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }

  // Create particle filter
  ParticleFilter pf;

  h.onMessage([&pf, &map, &delta_t, &sensor_range, &sigma_pos, &sigma_landmark](
      uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    if (length <= 2 || data[0] != '4' || data[1] != '2')
      return;

    auto s = hasData(std::string(data));
    if (s == "") {
      const auto msg = "42[\"manual\",{}]"s;
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      return;
    }

    const auto j = json::parse(s);
    // event
    if (j[0].get<std::string>() != "telemetry")
      return;

    // j[1] is the data JSON object
    if (!pf.initialized()) {
      // Sense noisy position data from the simulator
      const auto sense_x = std::stod(j[1]["sense_x"].get<std::string>());
      const auto sense_y = std::stod(j[1]["sense_y"].get<std::string>());
      const auto sense_theta =
          std::stod(j[1]["sense_theta"].get<std::string>());

      pf.init(sense_x, sense_y, sense_theta, sigma_pos);
    } else {
      // Predict the vehicle's next state from previous (noiseless
      // control) data.
      const auto previous_velocity =
          std::stod(j[1]["previous_velocity"].get<std::string>());
      const auto previous_yawrate =
          std::stod(j[1]["previous_yawrate"].get<std::string>());

      pf.prediction(delta_t, sigma_pos, previous_velocity, previous_yawrate);
    }

    // receive noisy observation data from the simulator
    // sense_observations in JSON format
    // [{obs_x,obs_y},{obs_x,obs_y},...{obs_x,obs_y}]
    std::vector<LandmarkObs> noisy_observations;
    const auto sense_observations_x =
        j[1]["sense_observations_x"].get<std::string>();
    const auto sense_observations_y =
        j[1]["sense_observations_y"].get<std::string>();

    std::vector<float> x_sense;
    std::istringstream iss_x(sense_observations_x);

    std::copy(std::istream_iterator<float>(iss_x),
              std::istream_iterator<float>(), std::back_inserter(x_sense));

    std::vector<float> y_sense;
    std::istringstream iss_y(sense_observations_y);

    std::copy(std::istream_iterator<float>(iss_y),
              std::istream_iterator<float>(), std::back_inserter(y_sense));

    for (auto i = decltype(x_sense.size())(0); i < x_sense.size(); i++) {
      noisy_observations.emplace_back(LandmarkObs{0, x_sense[i], y_sense[i]});
    }

    // Update the weights and resample
    pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);
    pf.resample();

    // Calculate and output the average weighted error of the particle
    // filter over all time steps so far.
    const double weight_sum = std::accumulate(
        pf.particles_.begin(), pf.particles_.end(), 0,
        [](double init, Particle p) { return init + p.weight_; });
    // Particle best_particle;
    const auto best_particle_it = std::max_element(
        pf.particles_.begin(), pf.particles_.end(),
        [](Particle a, Particle b) { return a.weight_ < b.weight_; });

    std::cout << "highest w " << best_particle_it->weight_ << std::endl;
    std::cout << "average w " << weight_sum / pf.particles_.size() << std::endl;

    json msgJson;
    msgJson["best_particle_x"] = best_particle_it->x_;
    msgJson["best_particle_y"] = best_particle_it->y_;
    msgJson["best_particle_theta"] = best_particle_it->theta_;

    // Optional message data used for debugging particle's sensing and
    // associations
    msgJson["best_particle_associations"] =
        pf.getAssociations(*best_particle_it);
    msgJson["best_particle_sense_x"] = pf.getSenseX(*best_particle_it);
    msgJson["best_particle_sense_y"] = pf.getSenseY(*best_particle_it);

    const auto msg = "42[\"best_particle\"," + msgJson.dump() + "]";
    // std::cout << msg << std::endl;
    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program doesn't compile :-(
  h.onHttpRequest(
      [](uWS::HttpResponse *res, uWS::HttpRequest req, char *, size_t, size_t) {
        const auto s = "<h1>Hello world!</h1>"s;
        if (req.getUrl().valueLength == 1) {
          res->end(s.data(), s.length());
        } else {
          // i guess this should be done more gracefully?
          res->end(nullptr, 0);
        }
      });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER>, uWS::HttpRequest) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int, char *, size_t) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  constexpr auto port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}
