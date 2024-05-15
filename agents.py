"""
MIT License

Copyright (c) 2024 Andrew Wang, Bryan Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
Implement various agents 
"""
class Agent:
    def __init__(self, veh_config, scen_config, x0, controller):
        self.veh_config = veh_config
        self.scen_config = scen_config
        self.x = [x0]
        self.controller = controller

    # Implement dynamics and update state one timestep later
    def step(self, oppo_states):
        raise NotImplementedError("Inheritance not implemented correctly")

    def getLastState(self):
        return self.x[-1]

    @property
    def ID(self):
        return self.config["ID"] 

   
class BicycleVehicle(Agent):
    def __init__(self, veh_config, scen_config, x0, controller):
        super().__init__(veh_config, scen_config, x0, controller)

    def step(self, oppo_states):

