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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
		
	num_particles = 200;
	weights.resize(num_particles, 1.0);

	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[3]);

	double sample_x, sample_y, sample_theta;
	sample_x = dist_x(gen);
	sample_y = dist_y(gen);
	sample_theta = dist_theta(gen);

	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1.;
		particles.push_back(p);
	}
	
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	normal_distribution<double> dist_x_pred(0, std_pos[0]);
	normal_distribution<double> dist_y_pred(0, std_pos[1]);
	normal_distribution<double> dist_theta_pred(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++)
	{
		if(fabs(yaw_rate) < 0.0001)
		{
			yaw_rate = 0.0001;
		}
		particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
		particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
		particles[i].theta += yaw_rate * delta_t;

		particles[i].x += dist_x_pred(gen);
		particles[i].y += dist_y_pred(gen);
		particles[i].theta += dist_theta_pred(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (int i = 0; i < observations.size(); i++)
	{
		double min_distance = 2000.0;

		for (int j = 0; j < predicted.size(); j++)
		{
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < min_distance)
			{
				min_distance = distance;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	std::cout << "Starting updating weights..." << std::endl;

	const double std_x = std_landmark[0]; // Bug in documentation (range?)
	const double std_y = std_landmark[1]; // Bug in documentation (bearing?)
	//double total_weight = 0;

	const double c1 = 1./(2.*M_PI*std_x*std_y);
	const double c2 = 2*std_x*std_x;
	const double c3 = 2*std_y*std_y;
	
	// Check the range for each particle and landmark
	for (int i = 0; i < particles.size(); i++)
	{	
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		particles[i].weight = 1./num_particles;

		vector<LandmarkObs> valid_landmarks; // Coordinates of the nearest landmark
		vector<LandmarkObs> map_observation; // Observations in map coordinates

		for (int j=0; j < map_landmarks.landmark_list.size(); j++)
		{
			int m_id = map_landmarks.landmark_list[j].id_i;
			float m_x = map_landmarks.landmark_list[j].x_f;
			float m_y = map_landmarks.landmark_list[j].y_f;

			double distance = dist(p_x, p_y, m_x, m_y);
			if(distance <= sensor_range)
			{
				valid_landmarks.push_back(LandmarkObs{m_id,m_x,m_y}); // Coordinates of the nearest landmark
			}
		}
	
		for (int k = 0; k < observations.size(); k++)
		{
			double px_m = p_x + (cos(p_theta) * observations[k].x) - (sin(p_theta) * observations[k].y); 
			double py_m = p_y + (sin(p_theta) * observations[k].x) + (cos(p_theta) * observations[k].y);
			map_observation.push_back(LandmarkObs{observations[k].id,px_m,py_m}); // Observations in map coordinates
		}
	
		dataAssociation(valid_landmarks, map_observation);
		double weight_prob = 1.;

		for (int n = 0; n < map_observation.size(); n++)
		{
			int m_obs_id = map_observation[n].id;
			double m_obs_x = map_observation[n].x;
			double m_obs_y = map_observation[n].y;

			double v_lm_x = valid_landmarks[m_obs_id].x;
			double v_lm_y = valid_landmarks[m_obs_id].y;

			// Multivariate-Gaussian Probabolity Density
			double x_mu = pow((m_obs_x - v_lm_x),2);
			double y_mu = pow((m_obs_y - v_lm_y),2);

			double prob = c1 * exp(-((x_mu / c2) + (y_mu / c3)));

			weight_prob *= prob;
		}

		particles[i].weight = weight_prob;
		weights[i] = weight_prob;
	}
}



void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> dist(weights.begin(), weights.end());

	vector<Particle> sampled_particles;
	vector<double> sampled_weights;

	for (int i = 0; i < num_particles; i++)
	{
		int idx = dist(gen);
		sampled_particles.push_back(particles[idx]);
		sampled_weights.push_back(particles[idx].weight);
	}

	particles = sampled_particles;
	weights = sampled_weights;

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
