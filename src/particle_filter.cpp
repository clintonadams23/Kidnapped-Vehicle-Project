#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <sstream>
#include <string>
#include <iterator>
#include <math.h>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double stdev[]) {	
	default_random_engine gen;
	// Initialize outside of the loop instead of using the AddGaussianNoise() function for efficiency 
	normal_distribution<double> dist_x(x, stdev[0]);
	normal_distribution<double> dist_y(y, stdev[1]);
	normal_distribution<double> dist_theta(theta, stdev[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
		
		weights.push_back(particle.weight);
		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;
	for (auto &particle : particles) {		
		double x = PredictMeasurementX(particle.x, particle.theta, delta_t, velocity, yaw_rate);
		double y = PredictMeasurementY(particle.y, particle.theta, delta_t, velocity, yaw_rate);
		double theta = PredictMeasurementTheta(particle.theta, delta_t, yaw_rate);
		
		particle.x = AddGaussianNoise(x, std_pos[0], gen);
		particle.y = AddGaussianNoise(y, std_pos[1], gen);
		particle.theta = AddGaussianNoise(theta, std_pos[2], gen);
	}
}

void ParticleFilter::dataAssociation(std::map<int, LandmarkObs> &landmarks , std::vector<LandmarkObs> &observations) {
	for (auto &observation : observations) {
		double smallest_euclidian_distance = dist(landmarks[0].x, landmarks[0].y, observation.x, observation.y);

		for (auto &map_landmark : landmarks) {
			LandmarkObs landmark = map_landmark.second;
			double euclidian_distance = dist(landmark.x, landmark.y, observation.x, observation.y);
			
			if (euclidian_distance < smallest_euclidian_distance) {
				observation.id = landmark.id;
				smallest_euclidian_distance = euclidian_distance;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	weights.clear();

	for (auto &particle: particles) {
		std::vector<LandmarkObs> observations_map;

		for (auto &observation : observations) {
			LandmarkObs observation_map = TransformToMapSpace(observation, particle);
			observations_map.push_back(observation_map);
		}

		std::map<int, LandmarkObs> landmarks = GetFilteredLandmarks(map_landmarks, particle, sensor_range);
		dataAssociation(landmarks, observations_map);
		
		particle.weight = GetParticleWeight(observations_map, landmarks, std_landmark);
		weights.push_back(particle.weight);
	}
}

void ParticleFilter::resample() {
	default_random_engine gen;
	discrete_distribution<int> weighted_distribution(weights.begin(), weights.end());

	std::vector<Particle> particles_resampled;

	for (int i = 0; i < num_particles; i++) {
		int sampled_index = weighted_distribution(gen);
		particles_resampled.push_back(particles[sampled_index]);
	}

	particles = particles_resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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

double ParticleFilter::PredictMeasurementX(double x_0, double theta, double delta_t, double velocity, double yaw_rate) {
	if (fabs(yaw_rate) < 0.0001) 
		return x_0 + velocity * delta_t * cos(theta);
	
	return x_0 + velocity/yaw_rate * ( sin(theta + yaw_rate*delta_t) - sin(theta) );
}

double ParticleFilter::PredictMeasurementY(double y_0, double theta, double delta_t, double velocity, double yaw_rate) {
	if (fabs(yaw_rate) < 0.0001)
		return y_0 + velocity * delta_t * sin(theta);
	
	return y_0 + velocity/yaw_rate * ( cos(theta) - cos(theta + yaw_rate*delta_t) );
}

double ParticleFilter::PredictMeasurementTheta(double theta_0, double delta_t, double yaw_rate) {
	return theta_0 + yaw_rate * delta_t;
}

double ParticleFilter::AddGaussianNoise(double measurement, double stdev, default_random_engine &gen) {
	normal_distribution<double> distribution(measurement, stdev);
	return distribution(gen);
}

std::map<int, LandmarkObs> ParticleFilter::GetFilteredLandmarks(Map &map, Particle &particle, double sensor_range) {
	std::map<int, LandmarkObs> landmark_lookup;
	for (auto &landmark : map.landmark_list) {
		if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) > sensor_range)
			continue;

		LandmarkObs filtered_landmark;
		filtered_landmark.x = landmark.x_f;
		filtered_landmark.y = landmark.y_f;
		filtered_landmark.id = landmark.id_i;

		landmark_lookup[landmark.id_i] = filtered_landmark;
	}

	return landmark_lookup;
}

LandmarkObs ParticleFilter::TransformToMapSpace(LandmarkObs &landmark, Particle &particle) {
	LandmarkObs landmark_global_coord;
	landmark_global_coord.x = particle.x + cos(particle.theta)*landmark.x - sin(particle.theta) * landmark.y;
	landmark_global_coord.y = particle.y + sin(particle.theta)*landmark.x + cos(particle.theta) * landmark.y;

	return landmark_global_coord;
}

double ParticleFilter::GetParticleWeight(std::vector<LandmarkObs> &observations, std::map<int, LandmarkObs> &landmarks, double std_landmark[]) {
	double weight = 1;
	for (auto &observation : observations) {
		LandmarkObs cooresponding_landmark = landmarks[observation.id];
		weight *= GaussianProbability(observation.x, observation.y, cooresponding_landmark.x, cooresponding_landmark.y, std_landmark);
	}
	return weight;
}

double ParticleFilter::GaussianProbability(double x, double y, double mu_x, double mu_y, double std_landmark[]) {
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	double denom = 2 * M_PI * sigma_x * sigma_y;
	double exponent = pow(x - mu_x, 2) / (2 * pow(sigma_x, 2)) + pow(y - mu_y, 2) / (2 * pow(sigma_y, 2));
	return exp(-exponent) / denom;
}