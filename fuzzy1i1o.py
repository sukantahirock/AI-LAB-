import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Define the fuzzy sets for temperature and fan speed
temperature = ctrl.Antecedent(np.arange(40, 81, 1), 'temperature')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# Step 2: Create membership functions
temperature['Cold'] = fuzz.trimf(temperature.universe, [40, 40, 50])
temperature['Cool'] = fuzz.trimf(temperature.universe, [45, 55, 65])
temperature['Warm'] = fuzz.trimf(temperature.universe, [60, 70, 75])
temperature['Hot'] = fuzz.trimf(temperature.universe, [70, 80, 80])

fan_speed['High'] = fuzz.trimf(fan_speed.universe, [80, 100, 100])
fan_speed['Medium'] = fuzz.trimf(fan_speed.universe, [50, 65, 80])
fan_speed['Low'] = fuzz.trimf(fan_speed.universe, [20, 35, 50])
fan_speed['Zero'] = fuzz.trimf(fan_speed.universe, [0, 0, 20])

# Step 3: Define the fuzzy rules
rule1 = ctrl.Rule(temperature['Cold'], fan_speed['High'])
rule2 = ctrl.Rule(temperature['Cool'], fan_speed['Medium'])
rule3 = ctrl.Rule(temperature['Warm'], fan_speed['Low'])
rule4 = ctrl.Rule(temperature['Hot'], fan_speed['Zero'])

# Step 4: Create a control system and simulation
fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

# Step 5: Input the temperature value to the system and compute the result
fan_sim.input['temperature'] = 70
fan_sim.compute()

# Get the result
fan_speed_result = fan_sim.output['fan_speed']
print(f"The fan speed for a temperature of 70 is: {fan_speed_result:.2f}")

# Visualize the result
temperature.view(sim=fan_sim)
fan_speed.view(sim=fan_sim)
