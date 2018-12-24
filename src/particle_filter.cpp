/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm> // copy
#include <cmath>     // pow, abs, exp, log ...
#include <iterator>  // ostream_iterator
#include <limits>    // numeric_limits
#include <sstream>   // stringstream
#include <string>    // string
#include <utility>   // swap

#include "particle_filter.h"

void Particle::vectorClearReserve(const size_t size) {
  associations_.clear();
  associations_.reserve(size);
  sense_x_.clear();
  sense_x_.reserve(size);
  sense_y_.clear();
  sense_y_.reserve(size);
}

void Particle::updateFromObservation(
    const std::vector<LandmarkObs> &observations,
    const std::array<double, 2> &std_landmark, const Map &map_landmarks) {
  // take the log
  auto minus_log_weight =
      std::log(2.0 * M_PI * std_landmark[0] * std_landmark[1]) *
      observations.size();
  auto std_x_2 = 2.0 * std::pow(std_landmark[0], 2);
  auto std_y_2 = 2.0 * std::pow(std_landmark[1], 2);

  vectorClearReserve(observations.size());

  for (const auto &p_obs : observations) {
    const auto landmark = map_landmarks.landmark_list.at(p_obs.id - 1);
    minus_log_weight += std::pow(p_obs.x - landmark.x_f, 2) / std_x_2 +
                        std::pow(p_obs.y - landmark.y_f, 2) / std_y_2;

    associations_.push_back(p_obs.id);
    sense_x_.push_back(p_obs.x);
    sense_y_.push_back(p_obs.y);
  }

  weight_ = std::exp(-minus_log_weight);
}

void ParticleFilter::init(const double x, const double y, const double theta,
                          const std::array<double, 3> &std) {

  // Set the number of particles. Initialize all particles to first position
  // (based on estimates of x, y, theta and their uncertainties from GPS) and
  // all weights to 1. Add random Gaussian noise to each particle.
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  num_particles_ = 100;

  // initialize particles
  particles_.reserve(num_particles_);
  for (auto id = decltype(num_particles_)(0); id < num_particles_; ++id) {
    particles_.emplace_back(Particle{
        dist_x(engine_), dist_y(engine_), dist_theta(engine_),
        1, // weight
    });
  }

  is_initialized_ = true;
}

void ParticleFilter::prediction(const double delta_t,
                                const std::array<double, 3> &std_pos,
                                const double velocity, const double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  // pre-calculation
  const auto distance = velocity * delta_t;
  const auto v_over_yaw_rate = velocity / yaw_rate;

  // update each paricle
  for (auto &&p : particles_) {
    if (std::abs(yaw_rate) == 0.0) {
      p.x_ += distance * cos(p.theta_);
      p.y_ += distance * sin(p.theta_);
    } else {
      const auto new_theta = p.theta_ + yaw_rate * delta_t;
      p.x_ += v_over_yaw_rate * (sin(new_theta) - sin(p.theta_));
      p.y_ += v_over_yaw_rate * (cos(p.theta_) - cos(new_theta));
      p.theta_ = new_theta;
    }

    // add noise
    p.x_ += dist_x(engine_);
    p.y_ += dist_y(engine_);
    p.theta_ += dist_theta(engine_);
  }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> &predicted,
                                     std::vector<LandmarkObs> &observations) {
  // Find the predicted measurement that is closest to each observed measurement
  // and assign the observed measurement to this particular landmark.

  for (auto &&observation : observations) {
    auto min = std::numeric_limits<double>::max();
    for (auto &&predicted_i : predicted) {
      const auto distance = dist_squared(observation.x, observation.y,
                                         predicted_i.x, predicted_i.y);
      if (distance < min) {
        min = distance;
        observation.id = predicted_i.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(const double sensor_range,
                                   const std::array<double, 2> &std_landmark,
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  const auto sensor_range_squared = std::pow(sensor_range, 2);
  weights_.clear();

  for (auto &&p : particles_) {
    // choose landmarks which are within sensor range
    std::vector<LandmarkObs> predicted;
    for (const auto &landmark : map_landmarks.landmark_list) {
      const auto distance =
          dist_squared(landmark.x_f, landmark.y_f, p.x_, p.y_);
      if (distance < sensor_range_squared) {
        predicted.emplace_back(
            LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // pre-calculation
    const double cos_theta = std::cos(p.theta_);
    const double sin_theta = std::sin(p.theta_);

    // calculate (particle coordinates) position of landmark
    std::vector<LandmarkObs> observations_from_particle;
    observations_from_particle.reserve(observations.size());
    for (auto &obs : observations) {
      observations_from_particle.emplace_back(
          LandmarkObs{obs.id, // 0: invalid
                      obs.x * cos_theta - obs.y * sin_theta + p.x_,
                      obs.x * sin_theta + obs.y * cos_theta + p.y_});
    }

    dataAssociation(predicted, observations_from_particle);
    p.updateFromObservation(observations_from_particle, std_landmark,
                            map_landmarks);
    weights_.push_back(p.weight_);
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their
  // weight.
  decltype(particles_) new_particles;
  new_particles.reserve(num_particles_);

  std::discrete_distribution<> dist(weights_.begin(), weights_.end());

  for (auto i = decltype(num_particles_)(0); i < num_particles_; i++) {
    new_particles.push_back(particles_[dist(engine_)]);
  }

  std::swap(particles_, new_particles);
  weights_.clear();
}

std::string ParticleFilter::getAssociations(const Particle &best) {
  const auto &v = best.associations_;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  auto s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseX(const Particle &best) {
  const auto &v = best.sense_x_;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  auto s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseY(const Particle &best) {
  const auto &v = best.sense_y_;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  auto s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
