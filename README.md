# FUNTF-Sampler

This repository contains a Python implementation for sampling finite unit norm tight frames (FUNTFs). The code constructs frames in the complex vector space ℂ<sup>d</sup> with prescribed eigensteps. It employs a rejection sampling routine over the polytope of independent eigensteps, uses a Hamiltonian torus action to explore the fiber of frames with fixed eigensteps, and computes the coherence distribution of the resulting frames.

## Overview  

The implementation is based on algorithms described in the papers:  

> **Sampling FUNTFs by way of a Hamiltonian toric action**  
> Faldet and Shonkwiler


> **Constructing Finite Frames of a Given Spectrum and Set of Lengths**  
> Cahill, Fickus, Mixon, Poteet, and Strawn.   

See `Results` folder for the ouput of several runs of the `coherence_distribution` function with various values of d and N. Key components of the implementation include:  

- **Frame Class:**   
Represents a frame (a spanning set of vectors in ℂ <sup>d</sup>) and provides methods to check properties (e.g., tightness, unit norm, coherence) and compute related operators (analysis, synthesis, frame operator).  

- **Eigensteps and EigenstepSystem Classes:**  
These classes encode the eigensteps (the eigenvalues of the partial frame operators) and their geometric structure. The eigensteps are generated from a symbolic table whose entries are either constants or symbolic variables representing the independent eigensteps. The `EigenstepSystem` class also extracts the system of inequalities (in H-representation) defining the polytope of independent eigensteps.    

- **StandardFormProgram Class:**  
Sets up the archetypal eigenstep table and extracts the defining inequalities, then parses these inequalities into a standard form (A*x ≤ b) for the polytope.   

- **Sampling Routines:**  

  - `random_FUNTF_eigensteps_bb(d, N, n_samples=1)`: Uses rejection sampling on the bounding box of the eigenstep (moment) polytope to generate a random point corresponding to a set of random eigensteps
  - `random_FUNTF(d, N, eigensteps=DEFAULT)`: Constructs a random finite unit norm tight frame with prescribed random eigensteps.
  - `torus_action(frame, theta=DEFAULT)`: Applies a Hamiltonian torus action on a the random frame, as defined in Faldet and Shonkwiler, to generate a random frame on the same fiber (i.e., with the same eigensteps).

- **Coherence Distribution:**  
    The `coherence_distribution` function generates a specified number of random frames, computes their coherence (the maximum absolute off-diagonal entry of the Gram matrix), plots a normalized histogram of these coherence values, and saves both the histogram image and the raw coherence values (one per line) to the Results folder. This is the proof of concept expirement referenced in Faldet and Shonkwiler.   

## Dependencies  

The implementation requires the following Python packages:
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [polytope](https://github.com/cvxgrp/polytope) (for polytope operations)

Install these dependencies using pip:

```bash
pip install numpy scipy matplotlib tqdm polytope
