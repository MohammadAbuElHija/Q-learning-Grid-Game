# Stochastic Wizard Game - Homework 3

## Project Overview

This project extends the previous wizard control problem by introducing **stochastic elements**, such as unpredictable movement of Death Eaters and random relocation of Horcruxes. The objective is to implement two intelligent agents (**WizardAgent** and **OptimalWizardAgent**) that maximize points while navigating this uncertain environment.

## Environment Details

- **Grid-based world** with wizards, horcruxes, and death eaters.
- **Stochastic Death Eaters:** Move randomly within predefined paths.
- **Stochastic Horcruxes:** Can change location randomly each turn.
- **Limited number of turns** to accumulate as many points as possible.

## Scoring System

- **+2 points** for destroying a horcrux.
- **-2 points** for resetting the environment.
- **-1 point per wizard** encountering a Death Eater.

## Actions

- **Reset:** Resets the game environment but does not reset turns or points.
- **Terminate:** Ends the game early if resetting is not beneficial.
- **Destroy Horcrux:** Requires specifying the wizard and horcrux name.

## Implementation Details

- Implement two agents:
  - **WizardAgent:** A baseline agent aiming for a positive score.
  - **OptimalWizardAgent:** Must solve all test cases optimally.
- Implement `__init__(self, initial)` (constructor) and `act(self, state)` (decision-making function).
- The **constructor must run within 300 seconds** and **act function must execute in 5 seconds**.

## Provided Files

- `ex3.py` - The only file to modify (implement the agents).
- `check.py` - Runs the environment simulation.
- `utils.py` - Contains useful helper functions.
- `inputs.py` - Provides input scenarios for testing.

##

