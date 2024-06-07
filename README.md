# Stanford Open Autonomous Racing (SOAR)
## SafeGP MPC for Autonomous Racing
AA203/AA273 Final Project

https://github.com/AndrewHWang1605/SOAR/assets/70298553/78e2f56e-88a3-4983-b8ae-2de7cf37f215

## About
SafeGP MPC is a safe intention-aware MPC controller for autonomous racing on a track with adversarial agents. Implemented are modules for autonomous racing simulation/data collection, offline global laptime optimization, Gaussian-process regression for opponent intent inference, and a safe MPC controller for safe autonomous racing. Additional documentation, demonstrations, and presentation materials can be found [here](https://drive.google.com/drive/folders/1O63d1-YUX6T9cePRiIDy8zhiHLa1naHj?usp=sharing).

## How to Use
All dynamics parameters, controller gains, simulation settings, etc can be found in `config.py`. To set up a racing scenario, simply initialize agents, corresponding controllers, and starting states, select options for visualizing the simulation results in `run_simulation.py`, and then run `python3 run_simulation.py`. 

## Acknowledgements
Special thanks to Tim Chen for his advice on Gaussian process implementation and to Daniele Gammelli for his guidance on optimization and prediction schemes. We’d also like to thank Mac Schwager and Marco Pavone for their help throughout the project and the course.



