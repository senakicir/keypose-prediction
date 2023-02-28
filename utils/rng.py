import torch
import numpy as np 
import random
from torch.distributions.multivariate_normal import MultivariateNormal


class Random_Number_Generator(object):
    def __init__(self, seed, body_noise_magn=5, input_noise_magn=0.05):
        self.seed = seed
        self.body_noise_magn = body_noise_magn
        self.input_noise_magn = input_noise_magn
        self.ss_prob = 1.0
        self.ss_cc_prob = 1.0
        self.shuffle_prob = 0.0

        torch.random.manual_seed(seed)
        self.teacher_forcing_state = torch.random.get_rng_state()
        self.teacher_forcing_cc_state = torch.random.get_rng_state()
        self.cluster_sample_state = torch.random.get_rng_state()

        self.kd_t_state = torch.random.get_rng_state()
        self.random_shuffle_state = torch.random.get_rng_state()
        self.input_noise_state = torch.random.get_rng_state()
        self.body_noise_state = torch.random.get_rng_state()

        self.augmentation_state = torch.random.get_rng_state()
        self.body_state = torch.random.get_rng_state()
        self.refinement_gt_prob_state = torch.random.get_rng_state()
        self.sample_label_state =  torch.random.get_rng_state()
        self.sample_duration_state =  torch.random.get_rng_state()

    def scheduled_sampling(self):
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.teacher_forcing_state)

        #generate the number
        random_prob = torch.rand(1)[0]

        #save the state of the tf to the rng
        self.teacher_forcing_state = torch.random.get_rng_state()
    
        return random_prob

    def scheduled_sampling_cc(self):
         #set the state of the tf to the rng
        torch.random.set_rng_state(self.teacher_forcing_cc_state)

        #generate the number
        random_prob = torch.rand(1)[0]

        #save the state of the tf to the rng
        self.teacher_forcing_cc_state = torch.random.get_rng_state()
    
        return random_prob
    
    def refinement_gt_prob(self):
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.refinement_gt_prob_state)

        #generate the number
        random_prob = torch.rand(1)[0]

        #save the state of the tf to the rng
        self.refinement_gt_prob_state = torch.random.get_rng_state()
    
        return random_prob

    def generate_body_noise(self, body_shape):
        N,C,T,V,M = body_shape
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.body_state)

        #generate the number
        random_prob1 = torch.rand(1)[0]
        random_prob2 = torch.rand(1)[0]

        body_noise = torch.zeros((N,C,T,V,M)).cuda()
        if random_prob1 < 0.50:
            body_noise += torch.normal(mean=0.0, std=30, size=(N,C,1,V,M)).cuda()
        if random_prob2 < 0.50:
            body_noise += torch.normal(mean=0.0, std=20, size=(N,C,T,V,M)).cuda()

        #save the state of the tf to the rng
        self.body_state = torch.random.get_rng_state()
    
        return body_noise

    def generate_body_noise_v2(self, body_shape):
        B,C,N = body_shape
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.body_state)

        #generate the number
        random_prob1 = torch.rand(1)[0]
        random_prob2 = torch.rand(1)[0]

        body_noise = torch.zeros(body_shape).cuda()
        if random_prob1 < 0.50:
            body_noise += torch.normal(mean=0.0, std=30, size=(B,C,1)).cuda()
        if random_prob2 < 0.50:
            body_noise += torch.normal(mean=0.0, std=20, size=(B,C,N)).cuda()

        #save the state of the tf to the rng
        self.body_state = torch.random.get_rng_state()
    
        return body_noise

    def random_shuffle(self):
        torch.random.set_rng_state(self.random_shuffle_state)

        random_prob = torch.rand(1)[0]

        self.random_shuffle_state = torch.random.get_rng_state()

        return random_prob


    def kd_t(self):
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.kd_t_state)

        #generate the number
        kd_t = torch.rand(1)[0]*0.9+0.1

        #save the state of the tf to the rng
        self.kd_t_state = torch.random.get_rng_state()
    
        return kd_t

    def input_noise(self, noise_shape):
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.input_noise_state)

        #generate the number
        input_noise =torch.normal(mean=0, std=self.input_noise_magn, size=noise_shape).cuda()

        #save the state of the tf to the rng
        self.input_noise_state = torch.random.get_rng_state()
    
        return input_noise


    def body_noise(self, noise_shape):
        #set the state of the tf to the rng
        torch.random.set_rng_state(self.body_noise_state)

        #generate the number
        input_noise =torch.normal(mean=0, std=self.body_noise_magn, size=noise_shape).cuda()

        #save the state of the tf to the rng
        self.body_noise_state = torch.random.get_rng_state()
    
        return input_noise

    def update_prob(self, ss_prob, shuffle_prob):
        self.ss_prob = ss_prob
        self.ss_cc_prob = ss_prob
        self.shuffle_prob = shuffle_prob

    def set_body_noise(self,body_noise_magn):
        self.body_noise_magn = body_noise_magn

    def set_input_noise(self,input_noise_magn):
        self.input_noise_magn = input_noise_magn

    def sample_label(self, dist):
        torch.random.set_rng_state(self.sample_label_state)

        generated_label = torch.multinomial(dist, 1)
        assert generated_label.shape == (dist.shape[0],1)
        
        self.sample_label_state = torch.random.get_rng_state()
        return generated_label[:,0]

    def sample_duration(self, dist):
        torch.random.set_rng_state(self.sample_duration_state)

        generated_label = torch.multinomial(dist, 1)
        assert generated_label.shape == (dist.shape[0],1)
        
        self.sample_duration_state = torch.random.get_rng_state()
        return generated_label[:,0]
        

    # def sample_from_cluster(self, cluster_means, cluster_covs):
    #     torch.random.set_rng_state(self.cluster_sample_state)

    #     #cluster means size (batch size, 66)
    #     batch_size, dims = cluster_means.shape
    #     sampled_vals = MultivariateNormal(cluster_means, cluster_covs).sample().cuda()
    #     assert sampled_vals.shape == cluster_means.shape

    #     self.cluster_sample_state = torch.random.get_rng_state()

    #     return sampled_vals

    def sample_from_cluster(self, cluster_contents, probabilites):
        batch_size = cluster_contents.shape[0]
        assert cluster_contents.shape == (batch_size, 10, 66)
        assert probabilites.shape == (batch_size, 10)

        torch.random.set_rng_state(self.cluster_sample_state)
        indices = torch.distributions.Categorical(probabilites).sample()
        sampled_vals = cluster_contents[torch.arange(batch_size), indices, :]
        self.cluster_sample_state = torch.random.get_rng_state()

        assert sampled_vals.shape == (batch_size, 66)
        return sampled_vals