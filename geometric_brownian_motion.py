##############################################################################################################

import pandas as pd
import numpy as np

##############################################################################################################

def geometric_brownian_motion(number_of_years = 10, number_of_scenarios = 1000, mu = 0.07, 
                              sigma = 0.15, steps_per_year = 12, initial_value = 100, prices = True):
    """
    Geometric Brownian Motion trajectories through Monte Carlo:
    
    1- "number_of_years" is about the number of years to generate data for.

    2- "mu" is annualized drift on market return.

    3- "sigma" is annualized volatility on market return.

    4- "steps_per_year" is granularity of the simulation for a year's steps.
    
    5- "initial_value" is initial value for amount of investing.

    """
    dt = 1 / steps_per_year
    number_of_steps = int(number_of_years * steps_per_year) + 1
    returns = np.random.normal(loc = (1 + mu) ** dt, scale = (sigma * np.sqrt(dt)), 
                               size = (number_of_steps, number_of_scenarios))
    returns[0] = 1
    returns_values = initial_value * pd.DataFrame(returns).cumprod() if prices else returns - 1
    return returns_values

##############################################################################################################