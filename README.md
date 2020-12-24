<!-- Header -->
# Multi Armed Bandits

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#adding-a-new-algorithm">Adding a new algorithm</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Implemented different algorithms for sampling the arms of a stochastic multi-armed bandit.

### Algorithms
The algorithms implemented are:
*  Epsilon Greedy exploration
* UCB
* KL-UCB
* Thompson Sampling
* A variation of Thompson Sampling
### Built With

Code is in python. It uses the following libraries:
* [NumPy](https://numpy.org/)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* NumPy

  ```sh
  $ pip3 install numpy
  ```

### Installation

1. Clone the repo
   
   ```sh
   $ git clone https://github.com/rishiagarwal2000/multi-armed-bandits.git 
   ```

<!-- USAGE EXAMPLES -->
## Usage

```sh
$ python3 bandit.py --instance INSTANCE --algorithm ALGORITHM --randomSeed RANDOMSEED --epsilon EPSILON --horizon HORIZON 
```
Here is an example:
```sh
$ python3 bandity.py --instance ../instances/i-1.txt --algorithm ucb --randomSeed 0 --epsilon 0.1 --horizon 200
```
Refer [Algorithms](#algorithms) to see the list of implemented algorithms.

Note: epsilon does not make sense for all the algorithms (it is dummy for these algorithms)

<!-- Adding a new algorithm -->
## Adding a new algorithm
Add your algorithm in the following section:
```python
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
```
### I/O specification:

* Inputs of the algorithm:
    * _I_ is the bandit instance (it is a numpy array)
    * _n_ is the number of arms in the instance (it is passed separately as we do not want the algorithm to use _I_ directly)
    * _ep_ is an optional argument
    * _hz_ is the horizon

* The algorithm should output the cumulative regret.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Rishi Agarwal - rishiapril2000@gmail.com

Project Link: [https://github.com/rishiagarwal2000/multi-armed-bandits.git](https://github.com/rishiagarwal2000/multi-armed-bandits.git)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Professor Shivaram Kalyanakrishnan, IIT Bombay](https://www.cse.iitb.ac.in/~shivaram/)
* Santhosh Kumar G., IIT Bombay