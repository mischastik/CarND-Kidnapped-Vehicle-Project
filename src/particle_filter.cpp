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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	num_particles = 100;
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
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);

		particles.push_back(p);
		weights.push_back(1.0);
	}

	std::cout << "INIT:" << std::endl;
	std::cout << "P1: (" << particles[0].x << ", " << particles[0].y << ", " << particles[0].theta << ")" << std::endl;
	std::cout << "P11: (" << particles[10].x << ", " << particles[10].y << ", " << particles[10].theta << ")" << std::endl;
	std::cout << "P100: (" << particles[99].x << ", " << particles[99].y << ", " << particles[99].theta << ")" << std::endl;

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen(0);
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);


	if (fabs(yaw_rate) < 0.0001)
	{
		for (size_t i = 0; i < particles.size(); i++)
		{
			Particle p = particles[i];
			p.x += velocity * delta_t * cos(p.theta) + noise_x(gen); 
			p.y += velocity * delta_t * sin(p.theta) + noise_y(gen); 
			p.theta += noise_theta(gen);
			particles[i] = p;
		}
	}
	else
	{
		for (size_t i = 0; i < particles.size(); i++)
		{
			Particle p = particles[i];
			// TODO: Find out if "* delta_t" is missing after velocity
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + noise_x(gen); 
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + noise_y(gen); 
			p.theta += yaw_rate * delta_t + noise_theta(gen);
			particles[i] = p;
		}
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	//std::cout << "Enter ASOC...";
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	std::vector<LandmarkObs> obs;
	for (size_t o_i = 0; o_i < observations.size(); o_i++)
	{
		double smallest_dist = numeric_limits<double>::max();
		int closest_idx;
		for (size_t p_i = 0; p_i < predicted.size(); p_i++)
		{
			double d = dist(predicted[p_i].x, predicted[p_i].y, observations[o_i].x, observations[o_i].y);
			if (d < smallest_dist)
			{
				smallest_dist = d;
				closest_idx = p_i;
			}
		}
		observations[o_i].id = closest_idx;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
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
	double min_diff_x = numeric_limits<double>::max();

	for (size_t prt_idx = 0; prt_idx < num_particles; prt_idx++)
	{
		Particle p = particles[prt_idx];
		// transform observations into map CS
		std::vector<LandmarkObs> observations_t;
		for (size_t obs_idx = 0; obs_idx < observations.size(); obs_idx++)
		{
			LandmarkObs obs = observations[obs_idx];

			LandmarkObs obs_t;
			obs_t.id = -1;
			
			obs_t.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
			obs_t.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;

			observations_t.push_back(obs_t);
		}
		std::vector<LandmarkObs> predicted;
		for (size_t lm_idx = 0; lm_idx < map_landmarks.landmark_list.size(); lm_idx++)
		{
			Map::single_landmark_s lm = map_landmarks.landmark_list[lm_idx];
			if (dist(p.x, p.y, lm.x_f, lm.y_f) > sensor_range)
			{
				continue;
			}
			LandmarkObs pred;
			pred.id = lm.id_i;
			pred.x = lm.x_f;
			pred.y = lm.y_f;
			predicted.push_back(pred);
		}

		// find closest landmark for each observation and calculate weight based on distance.
		dataAssociation(predicted, observations_t);
		weights[prt_idx] = 1.0f;
		double scaleFactor = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		double den_x = (2 * std_landmark[0] * std_landmark[0]);
		double den_y = (2 * std_landmark[1] * std_landmark[1]);
		for (size_t obs_idx = 0; obs_idx <observations_t.size(); obs_idx++)
		{
			LandmarkObs observation = observations_t[obs_idx];
			LandmarkObs mappedObservation = predicted[observation.id];
			double diff_x = mappedObservation.x - observation.x;
			if (diff_x < min_diff_x)
			{
				min_diff_x = diff_x;
			}
			double diff_y = mappedObservation.y - observation.y;
			double exponent = -(diff_x * diff_x) / den_x - (diff_y * diff_y) / den_y;
			weights[prt_idx] *= scaleFactor * exp(exponent);
		}
	}
	std::cout << min_diff_x << std::endl;
}

void ParticleFilter::resample() 
{
	double weights_sum = 0;
	for (size_t wgt_idx = 0; wgt_idx < weights.size(); wgt_idx++)
	{
		weights_sum += weights[wgt_idx];
	}
	for (size_t wgt_idx = 0; wgt_idx < weights.size(); wgt_idx++)
	{
		weights[wgt_idx] /= weights_sum;
	}
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
