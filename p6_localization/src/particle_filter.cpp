/*
 * particle_filter.cpp
 *
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <map>

#include "particle_filter.h"
using namespace std;

// Setup random value generation engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
   /**
    * Initialize particles based on the given state of the car, i.e., coordinate
    * (x, y) in 2d space and heading of the vehicle (theta). Note Gaussian noises
    * are also generated in the creation of particles. Weights of all particles
    * are set to 1.0.
    */
    
    // Set number of particles
    num_particles = 100;
    weights.resize(num_particles);
    
    // Create normal distribution for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Generate random particles
    for (size_t i = 0; i < num_particles; ++i) 
    {
        particles.emplace_back(i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0);   
    }

    // Mark as initialized
    is_initialized = true;
}

void ParticleFilter::predict(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
    /**
     * Add (1) measurements and (2) Gaussian noise to each particle
     */
    
    // Setup Gaussian noise for x, y, and theta
    normal_distribution<double> jitter_x(0, std_pos[0]);
    normal_distribution<double> jitter_y(0, std_pos[1]);
    normal_distribution<double> jitter_theta(0, std_pos[2]);

    // Make prediction 
    for (size_t i = 0; i < num_particles; ++i) 
    {
        if (abs(yaw_rate) != 0)
        {
            particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;

        } 
        else 
        {
            // theta does not change
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }

        // Add Gaussian noise 
        particles[i].x += jitter_x(gen);
        particles[i].y += jitter_y(gen);
        particles[i].theta += jitter_theta(gen);
    }
}

void ParticleFilter::associate(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) 
{
    /**
     * Find the predicted measurement that is closest to each observed
     * measurement and assign the observed measurement to this 
     * particular landmark.
     */

    for (size_t i = 0; i < observations.size(); i++) 
    {
        // Initialization
        double min_dist = numeric_limits<double>::max();
        int map_id = -1;

        // Find the minimum distance
        for (size_t j = 0; j < predicted.size(); j++) 
        {
            const double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (cur_dist < min_dist) 
            {
                min_dist = cur_dist;
                map_id = predicted[j].id;
            }
        }

        // Update the id to the found map id of the minimum distance 
        observations[i].id = map_id;
    }
}

void ParticleFilter::update_weights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
    /**
     * Updates the weights of each particle using a multi-variate Gaussian 
     * distribution.
     */
    auto transform = [&](const int i) -> vector<LandmarkObs>
    {
        // Transform the observations from vehicle coordinate to map 
        // coordinate given the particle state

        vector<LandmarkObs> observations_t; 
        for (unsigned int j = 0; j < observations.size(); j++) 
        {
            double x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
            double y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
            observations_t.push_back(LandmarkObs{ observations[j].id, x, y });
        }
        return observations_t;
    };

    auto get_predictions = [&](const int i) -> vector<LandmarkObs> 
    {
        // Get the landmarks within the sensor coverage given the 
        // particle state

        vector<LandmarkObs> predictions;
        for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++) 
        {
            // Get id and x,y coordinates
            const float x = map_landmarks.landmark_list[j].x_f;
            const float y = map_landmarks.landmark_list[j].y_f;
            const int id = map_landmarks.landmark_list[j].id_i;
            // Only consider landmarks within the sensor coverage 
            if (fabs(x - particles[i].x) <= sensor_range && fabs(y - particles[i].y) <= sensor_range) 
            {
                predictions.push_back(LandmarkObs{id, x, y });
            }
        }
        return predictions;
    };

    const auto fixed1 = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    const auto fixed2 = 2 * std_landmark[0] * std_landmark[0];
    const auto fixed3 = 2 * std_landmark[1] * std_landmark[1];

    for (size_t i = 0; i < num_particles; ++i) 
    {
        // Get the predicted landmarks and the transformed observations
        auto predictions = get_predictions(i);
        auto observations_t = transform(i); 

        // Associate the predicted landmarks to the transformed observations 
        associate(predictions, observations_t);

        // Update weight
        particles[i].weight = 1.0;
        for (size_t j = 0; j < observations_t.size(); j++) 
        {
            double ox, oy, px, py;
            ox = observations_t[j].x;
            oy = observations_t[j].y;

            // Find the prediction associated with this observation
            for (size_t k = 0; k < predictions.size(); k++) 
            {
                if (predictions[k].id == observations_t[j].id) 
                {
                    px = predictions[k].x;
                    py = predictions[k].y;
                    break;
                }
            }

            // Calculate weight for this observation with multivariate Gaussian
            double obs_w = fixed1 * exp(-(pow(px - ox, 2) / fixed2 + pow(py - oy, 2) / fixed3));

            // Update the weight
            particles[i].weight *= obs_w;
        }
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample()
{
    /**
     * Resample particles with replacement with probability proportional 
     * to their weight. 
     */

    vector<Particle> new_particles;
    
    // Select an random index as the starting point for resampling
    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);

    // Uniform random distribution [0.0, max_weight)
    const double max_weight = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    // Resample based on the spinning wheel
    double beta = 0.0;
    for (size_t i = 0; i < num_particles; i++) 
    {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) 
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;
}

string ParticleFilter::get_associations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::get_sense_coord(Particle best, string coord) 
{
    vector<double> v;

    if (coord == "X") 
    {
        v = best.sense_x;
    } 
    else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
