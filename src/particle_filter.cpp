/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen(0);
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);

	// Create normal distributions for y and theta.
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		// Sample normal distrubtions
		Particle p;

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);

		particles.push_back(p);
		weights.push_back(1.0);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen(0);
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);


	if (yaw_rate < 0.0001)
	{
		for (size_t i = 0; i < particles.size(); i++)
		{
			Particle p = particles[i];
			p.x += velocity * delta_t * cos(p.theta) + noise_x(gen); // + noise
			p.y += velocity * delta_t * sin(p.theta) + noise_y(gen); // + noise
			particles[i] = p;
		}
	}
	else
	{
		for (size_t i = 0; i < particles.size(); i++)
		{
			Particle p = particles[i];
			// TODO: Find out if "* delta_t" is missing after velocity
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + noise_x(gen); // + noise
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + noise_y(gen); // + noise
			p.theta += yaw_rate * delta_t;
			particles[i] = p;
		}
	}
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(const std::vector<LandmarkObs> predicted, const std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	std::vector<LandmarkObs> obs;
	for (size_t o_i = 0; o_i < observations.size(); o_i++)
	{
		double smallest_dist = numeric_limits<double>::max();
		LandmarkObs closest;
		for (size_t p_i = 0; p_i < predicted.size(); p_i++)
		{
			double d = dist(predicted[p_i].x, predicted[p_i].y, observations[o_i].x, observations[o_i].y);
			if (d < smallest_dist)
			{
				smallest_dist = d;
				closest = observations[o_i];
			}
		}
		obs.push_back(closest);
	}
	return obs;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double weights_sum = 0.0;
	for (size_t prt_idx = 0; prt_idx < particles.size(); prt_idx++)
	{
		Particle p = particles[prt_idx];
		// transform map landmarks into particle CS
		std::vector<LandmarkObs> predicted;
		for (size_t obs_i = 0; obs_i < map_landmarks.landmark_list.size(); obs_i++)
		{
			Map::single_landmark_s lm = map_landmarks.landmark_list[obs_i];
			LandmarkObs obs_t;
			obs_t.id = lm.id_i;
			// TODO: check if this needs to be the inverse transformation
			obs_t.x = lm.x_f * cos(p.theta) - lm.y_f * sin(p.theta) + p.x;
			obs_t.y = lm.x_f * sin(p.theta) + lm.y_f * cos(p.theta) + p.y;
			predicted.push_back(obs_t);
		}
		// find closest landmark for each observation and calculate weight based on distance.
		vector<LandmarkObs> mappedObservations = dataAssociation(predicted, observations);
		weights[prt_idx] = 1.0f;
		double scaleFactor = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		for (size_t obs_idx = 0; obs_idx < mappedObservations.size(); obs_idx++)
		{
			double diff_x = mappedObservations[obs_idx].x - observations[obs_idx].x;
			double diff_y = mappedObservations[obs_idx].y - observations[obs_idx].y;
			double exponent = -diff_x * diff_x / (2 * std_landmark[0]) - diff_y * diff_y / (2 * std_landmark[1]);
			weights[prt_idx] *= scaleFactor * exp(exponent);
		}
		weights_sum += weights[prt_idx];
	}
	for (size_t wgt_idx = 0; wgt_idx < weights.size(); wgt_idx++)
	{
		weights[wgt_idx] /= weights_sum;
	}
}

void ParticleFilter::resample() 
{
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;     
	std::mt19937 rng(rd());    
	std::uniform_int_distribution<int> uni(0, num_particles - 1); // guaranteed unbiased

	double w_max = 0.0;
	for (size_t i = 0; i < weights.size(); i++)
	{
		if (weights[i] > w_max)
		{
			w_max = weights[i];
		}
	}
	std::uniform_real_distribution<double> unif(0, w_max);
	std::default_random_engine re;

	std::vector<Particle> newParticles;
	double beta = 0;
	int index = uni(rng);
	for (int i = 0; i < num_particles; i++)
	{
		beta += unif(re);

		while (weights[index] < beta)
		{
			beta -= weights[index];
			index += 1;
			index = index % num_particles;
		}
		newParticles.push_back(particles[index]);
	}
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
