# Cross-Modal Stereo Sonar Geometry: Projections for Sidescan and Forward-Looking Sonar

This repository contains helpfer functions and example code for the processes outlined in the paper "Cross-Modal Stereo Sonar Geometry: Projections for Sidescan and Forward-Looking Sonar", which was presented at IEEE/MTS OCEANS 2025 Great Lakes. 

All of the code can be found in the `src` folder, and a brief description of the files is: 

- `example_fl2ss.py`: Contains a simple example of projecting a feature observed by a forward-looking sonar to a sidescan sonar measurement. The values found in the `config` variable can be easily adjusted for a desired scenario.
- `example_ss2fl.py`: Contains a simple example of projecting a feature observed by a sidescan sonar to a forward-looking sonar image. The values found in the `config` variable can be easily adjusted for a desired scenario.
- `utils.py`: Contains all of the helper functions used in the provided examples. 
- `input_checking.py`: Each function in all of the above files has a function in this file that is used to check input parameters to ensure that they are valid and of the correct type. This is intended to prevent issues with accidental invalid inputs, but may not catch everything, and the functionality can easily be disabled if you intend to make changes to existing functions in this repo. 

With these functions and examples we hope to provide a resource for simple simulations of the work presented in our paper, and we encourage the use of the geometric feature projections in your own work! 

We note that this is not necessarily a "library", but rather a collection of openly accesssible code that, in functionality, is identical to what was used to generate the results for our paper, but without the actual scripts that we used for the figures. 