import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define input variables
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
fan_speed = ctrl.Consequent(np.arange(0, 401, 1), 'fan_speed')

# Define temperature ranges
temperature['cool'] = fuzz.trimf(temperature.universe, [0, 15, 30])
temperature['warm'] = fuzz.trimf(temperature.universe, [20, 40, 60])
temperature['hot'] = fuzz.trimf(temperature.universe, [50, 75, 100])

# Define humidity ranges
humidity['dry'] = fuzz.trimf(humidity.universe, [0, 25, 50])
humidity['moderate'] = fuzz.trimf(humidity.universe, [40, 60, 80])
humidity['wet'] = fuzz.trimf(humidity.universe, [70, 85, 100])

# Define fan speed ranges
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 100, 200])
fan_speed['moderate'] = fuzz.trimf(fan_speed.universe, [150, 225, 300])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [250, 325, 400])

# Define rules
rule1 = ctrl.Rule(temperature['cool'] & humidity['dry'], fan_speed['low'])
rule2 = ctrl.Rule(temperature['cool'] & humidity['moderate'], fan_speed['low'])
rule3 = ctrl.Rule(temperature['warm'] & humidity['dry'], fan_speed['low'])
rule4 = ctrl.Rule(temperature['warm'] & humidity['moderate'], fan_speed['moderate'])
rule5 = ctrl.Rule(temperature['warm'] & humidity['wet'], fan_speed['moderate'])
rule6 = ctrl.Rule(temperature['hot'] & humidity['moderate'], fan_speed['high'])
rule7 = ctrl.Rule(temperature['hot'] & humidity['wet'], fan_speed['high'])

# Create control system
fan_speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
fan_speed_simulation = ctrl.ControlSystemSimulation(fan_speed_ctrl)

# Take user inputs
temp_input = float(input("Enter temperature (0-100): "))
humidity_input = float(input("Enter humidity (0-100): "))

# Compute fan speed
fan_speed_simulation.input['temperature'] = temp_input
fan_speed_simulation.input['humidity'] = humidity_input
fan_speed_simulation.compute()

# Output fan speed
print("Fan Speed:", fan_speed_simulation.output['fan_speed'])

# Show the fan speed graph
fan_speed.view(sim=fan_speed_simulation)