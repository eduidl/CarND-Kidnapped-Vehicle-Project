/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#pragma once

#include <array>
#include <random> // mt19937, random_device, discrete_distribution
#include <vector>

#include "helper_functions.h"

class Particle {
public:
  double x_;
  double y_;
  double theta_;
  double weight_;
  std::vector<int> associations_;
  std::vector<double> sense_x_;
  std::vector<double> sense_y_;

  Particle(double x, double y, double theta, double weight)
      : x_(x), y_(y), theta_(theta), weight_(weight), associations_(0),
        sense_x_(0), sense_y_(0) {}

  ~Particle() = default;

  /**
   * vectorClearReserve Call clear() and reserve() from associations_, sense_x_,
   * sense_y
   * @param size Capacity of vector to reserve
   */
  void vectorClearReserve(const size_t size);

  /**
   * updateFromObservation Updates the weights for each particle based on the
   * likelihood of the observed measurements.
   * @param observations Vector of landmark observations from this particle
   * @param std_landmark Array of dimension 2 [Landmark measurement
   * uncertainty [x [m], y [m]]]
   * @param map Map class containing map landmarks
   */
  void updateFromObservation(const std::vector<LandmarkObs> &observations,
                             const std::array<double, 2> &std_landmark,
                             const Map &map_landmarks);
};

class ParticleFilter {
  int num_particles_;           // Number of particles to draw
  bool is_initialized_;         // Flag, if filter is initialized
  std::vector<double> weights_; // Vector of weights of all particles
  std::mt19937 engine_;         // Random engine

public:
  // Set of current particles
  std::vector<Particle> particles_;

  // Constructor
  // @param num_particles Number of particles
  ParticleFilter()
      : num_particles_(0), is_initialized_(false),
        engine_(std::random_device{}()) {}

  // Destructor
  ~ParticleFilter() {}

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std Array of dimension 3 [standard deviation of x [m], standard
   * deviation of y [m] standard deviation of yaw [rad]]
   */
  void init(const double x, const double y, const double theta,
            const std::array<double, 3> &std);

  /**
   * prediction Predicts the state for the next time step
   *   using the process model.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param std_pos Array of dimension 3 [standard deviation of x [m],
   * standard deviation of y [m] standard deviation of yaw [rad]]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void prediction(const double delta_t, const std::array<double, 3> &std_pos,
                  const double velocity, const double yaw_rate);

  /**
   * dataAssociation Finds which observations correspond to which landmarks
   * (likely by using a nearest-neighbors data association).
   * @param predicted Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  void dataAssociation(const std::vector<LandmarkObs> &predicted,
                       std::vector<LandmarkObs> &observations);

  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   * of the observed measurements.
   * @param sensor_range Range [m] of sensor
   * @param std_landmark Array of dimension 2 [Landmark measurement
   * uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  void updateWeights(const double sensor_range,
                     const std::array<double, 2> &std_landmark,
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks);

  /**
   * resample Resamples from the updated set of particles to form
   *   the new set of particles.
   */
  void resample();

  std::string getAssociations(const Particle &best);
  std::string getSenseX(const Particle &best);
  std::string getSenseY(const Particle &best);

  /**
  * initialized Returns whether particle filter is initialized yet or not.
  */
  bool initialized() const { return is_initialized_; }
};
