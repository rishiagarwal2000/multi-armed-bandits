import argparse
import numpy as np
# from random import seed
# from random import random

parser = argparse.ArgumentParser(description="Multi armed bandit")
parser.add_argument('--instance', type=str, help='Input file to bandit instance')
parser.add_argument('--algorithm', type=str, help='Name of algorithm to be used')
parser.add_argument('--randomSeed', type=int, help='Random seed')
parser.add_argument('--epsilon', type=float, help='epsilon for greedy algorithm')
parser.add_argument('--horizon', type=int, help='T')

args = parser.parse_args()
infile = args.instance
algo = args.algorithm
rs = args.randomSeed
ep = args.epsilon
hz = args.horizon

# Verify argparse
# print('{}, {}, {}, {}, {}'.format(infile, algo, rs, ep, hz))

# seed(rs)

def pull(I, arm):
	p = I[arm]
	r = 1 if np.random.uniform() < p else 0
	return r

# Verify seed functionality
# for i in range(10):
# 	print(pull(I, i))

def ep_greedy(I, n, ep, hz):
	assert ep is not None
	sum_rewards = np.zeros(n)
	num_pulls = np.zeros(n)
	total_reward = 0
	arm = -1
	for i in range(min(n, hz)):
		r = pull(I, i)
		num_pulls[i] += 1
		sum_rewards[i] += r
		total_reward += r

	for i in range(hz-n):
		if np.random.uniform() < ep:
			# explore
			arm = np.random.choice(n)
		else:
			# exploit
			means = sum_rewards / num_pulls
			arm = np.argmax(means)
		r = pull(I, arm)
		sum_rewards[arm] += r
		num_pulls[arm] += 1
		total_reward += r
	reg = max(I) * hz - total_reward
	return reg

def ucb(I, n, hz):
	sum_rewards = np.zeros(n)
	num_pulls = np.zeros(n)
	total_reward = 0
	arm = -1
	for i in range(min(n, hz)):
		r = pull(I, i)
		num_pulls[i] += 1
		sum_rewards[i] += r
		total_reward += r

	for i in range(hz-n):
		# exploit
		means = sum_rewards / num_pulls
		means = means + np.sqrt(2 * np.log(i+n) / num_pulls)
		arm = np.argmax(means)
		r = pull(I, arm)
		sum_rewards[arm] += r
		num_pulls[arm] += 1
		total_reward += r
	reg = max(I) * hz - total_reward
	return reg


def kl_ucb(I, n, hz):
	sum_rewards = np.zeros(n)
	num_pulls = np.zeros(n)
	total_reward = 0
	arm = -1
	c = 3
	for i in range(min(n, hz)):
		r = pull(I, i)
		num_pulls[i] += 1
		sum_rewards[i] += r
		total_reward += r

	for i in range(hz-n):
		# exploit
		means = sum_rewards / num_pulls
		ucb_metric = np.zeros(n)
		thres = np.log(i+n) + c * np.log(np.log(i+n))  # assume n >= 2
		for j in range(n):
			l = means[j]
			h = 1.0
			eps = 1e-3
			while l < h - eps:
				m = (l + h) / 2
				# if abs(m) < 1e-8 or abs(m-1)<1e-8:
				# 	print("Error ",m)
				KL = means[j] * np.log(means[j]/(m+1e-9)) + (1-means[j]) * np.log((1-means[j])/(1-m+1e-9))
				if num_pulls[j] * KL > thres:
					h = m
				else:
					l = m
			ucb_metric[j] = (l+h)/2 
		arm = np.argmax(ucb_metric)
		r = pull(I, arm)
		sum_rewards[arm] += r
		num_pulls[arm] += 1
		total_reward += r
	reg = max(I) * hz - total_reward
	return reg

def thompson(I, n, hz):
	s = np.zeros(n)
	f = np.zeros(n)
	total_reward = 0
	arm = -1

	for i in range(hz):
		# exploit
		samples = np.random.beta(s+1, f+1)
		arm = np.argmax(samples)
		r = pull(I, arm)
		s[arm] += r
		f[arm] += 1-r
		total_reward += r
	reg = max(I) * hz - total_reward
	return reg

def thompson_hint(I, n, hz, hint):
	belief = np.ones((n, n)) / n
	p_max_id = np.argmax(hint)
	total_reward = 0
	for i in range(hz):
		arm = np.argmax(belief[:, p_max_id])
		r = pull(I, arm)
		normalizing_factor = 0
		for j in range(n):
			belief[arm, j] = (hint[j]**r) * ((1-hint[j])**(1-r)) * belief[arm, j]
			normalizing_factor += belief[arm, j]
		belief[arm, :] = belief[arm, :] / normalizing_factor
		total_reward += r
	reg = max(I) * hz - total_reward
	return reg

def main(infile, algo, rs, ep, hz):
	reg = 0
	np.random.seed(rs)
	I = np.loadtxt(infile)
	n = len(I)
	if algo == 'epsilon-greedy':
		reg = ep_greedy(I, n, ep, hz)
	elif algo == 'ucb':
		reg = ucb(I, n, hz)
	elif algo == 'kl-ucb':
		reg = kl_ucb(I, n, hz)
	elif algo == 'thompson-sampling':
		reg = thompson(I, n, hz)
	elif algo == 'thompson-sampling-with-hint':
		hint = np.sort(I)
		reg = thompson_hint(I, n, hz, hint)	
	else:
		print('Algo not found')
	if ep is None:
		ep = 0.1
	print('{}, {}, {}, {}, {}, {}'.format(infile, algo, rs, ep, hz, reg))
	return reg

main(infile, algo, rs, ep, hz)