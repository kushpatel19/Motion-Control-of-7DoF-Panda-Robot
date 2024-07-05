# Motion-Control-of-7DoF-Panda-Robot

## Project Overview
The "Motion-Control-of-7DoF-Panda-Robot" project focuses on executing a real-world manipulation task using the 7 Degrees of Freedom (7DoF) Franka Emika Panda robotic manipulator in simulation. This project is divided into two parts: analysis and simulation of the manipulator, and the development of an environment and implementation of a manipulation task.

https://github.com/kushpatel19/Motion-Control-of-7DoF-Panda-Robot/assets/97977579/6f32e9ee-9606-4d35-a0e1-cc8858abc337

## Table of Contents
1. [Introduction](#introduction)
2. [Project Parts](#project-parts)
   - [Part 1: Analysis and Simulation](#part-1-analysis-and-simulation)
   - [Part 2: Environment Development and Task Implementation](#part-2-environment-development-and-task-implementation)
3. [Features](#features)
4. [Installation Instructions](#installation-instructions)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction
The main focus of this project is to perform a detailed analysis and simulation of a robotic manipulator, and then develop an environment to execute a manipulation task using the Franka Emika Panda robot. We have used Coppelia Sim for simulation and MATLAB for analysis and task implementation.

## Project Parts

### Part 1: Analysis and Simulation
1. **Manipulator Selection**: We selected the 7DoF Franka Emika Panda robotic manipulator, known for its dexterity and versatility.
2. **Literature Survey**: A comprehensive literature survey was conducted to establish the motivation and significance of the chosen manipulator.
3. **Analytical Analysis**:
   - **Task Representation**
   - **Position Analysis**
   - **Velocity and Statics Analysis**
   - **Redundancy Resolution**
   - **Stiffness Analysis**
   - **Dynamics Analysis**
   - Tools used: MATLAB for symbolic computation and plotting.
4. **Simulation Development**: An existed model of the manipulator was used in Coppelia Sim. Various simulations were performed to validate the manipulator's capabilities.

### Part 2: Environment Development and Task Implementation
1. **Task Definition**: Defined the manipulation task for the Panda robot.
2. **Motion Control**: Developed a motion control strategy for the robot using MATLAB.
3. **Simulation and Analysis**: Detailed formulation and analysis of the robot’s dynamics were performed.
4. **Report and Media**: A PDF report elaborating the task and its execution was created. 

## Features
- **Robust Analysis**: Comprehensive position, velocity, statics, and dynamics analysis.
- **Motion Control Implementation**: Effective motion control strategy implemented using MATLAB.
- **Documentation**: Well-documented reports and code files for easy understanding and replication.

## Installation Instructions
To run the simulations and analyses, follow these steps:

1. **Clone the Repository**:
   ```sh
   $ git clone https://github.com/kushpatel19/Motion-Control-of-7DoF-Panda-Robot.git
   $ cd Motion-Control-of-7DoF-Panda-Robot
   ```
   
2. **Software Requirements**:
  - Install [Coppelia Sim](https://www.coppeliarobotics.com/)
  - Install [MATLAB](https://www.mathworks.com/products/matlab.html)

3. **Running Simulations**:
  - Open Coppelia Sim and load the simulation files located in the `simulation` folder.
  - Run the simulations to observe the manipulator's behavior.

4. **Running MATLAB Scripts**:
  - Open MATLAB and navigate to the `code` folder.
  - Run the scripts to perform the analysis and motion control tasks.

## Usage
To use the project, follow the detailed instructions provided in the reports:

- **Part 1 Report**: `AK47_Project-I.pdf`
- **Part 2 Report**: `AK47_Project-II.pdf`

These reports provide step-by-step guidance on performing the analyses and executing the manipulation task.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
