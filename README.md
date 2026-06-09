# RimuRealTime

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rimuqmc.github.io/RimuRealTime.jl/dev/)

[![Build Status](https://github.com/RimuQMC/RimuRealTime.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RimuQMC/RimuRealTime.jl/actions/workflows/CI.yml?query=branch%3Amain)

Real time evolution with [Rimu.jl](https://github.com/RimuQMC/Rimu.jl).

## Installation

RimuRealTime.jl is not yet registered. To install it, run

```julia
import Pkg; Pkg.add("https://github.com/RimuQMC/RimuRealTime.jl")
```

## Usage Guide

```julia
using Rimu
using RimuRealTime
```

First we set up the model Hamiltonian, here a one-dimensional Fröhlich polaron.

```julia
l = 2
kc = 2
coupling = 0.5
modes = l*kc + 1
address = OccupationNumberFS{modes}()
hamiltonian = FroehlichPolaron(address; v=sqrt(2*coupling/l), l)
```

We construct a `QuantumDynamicsProblem` to carry out the evolution, using various keyword
arguments.

The starting state of the evolution can be chosen by passing an address or DVec with
keyword argument `start_at`. By default the starting state is the `starting_address` of the
hamiltonian.

We set size of the time step, along with the endpoint of the evolution in real time.

```julia
time_step = 0.001
maximum_time = 10
```

Optionally the time evolution can be carried out at an angle in the complex plane, by
setting non-zero `alpha`. By default `alpha=0` and the evolution is purely real;
`alpha=pi/2` corresponds to pure imaginary evolution.

```julia
alpha = 0.01
```

The `evolution_strategy` defines how the evolution occurs at each step. A first-order step
can be chosen with `Euler()`. At second order, the most efficient method is `PEC()`, which
only requires one operator application per step.

```julia
evolution_strategy = PEC()
```

The vector is initialised with the chosen `style` and number of `initial_walkers`.
The walker number can be kept constant as shown using the `scaling_strategy`.
We can also use initiators, here with a threshold of `3`.

```julia
style = IsDynamicSemistochastic{ComplexF64}()
initial_walkers = 1000
scaling_strategy = DynamicScaling(initial_walkers)
initiator = Initiator(3)
```

A `post_step_strategy` can be defined and observables can be calculated with replicas
using `AllOverlaps` as in Rimu. Here these are used to calculate the 2-norm at each time
step, and the density of phonons in each mode.

```julia
post_step_strategy = Projector(:norm, Norm2Projector())
operator = ((DensityMatrixDiagonal(i) for i in 1:modes)...,)
n_replicas = 3
replica_strategy = AllOverlaps(n_replicas; operator)
```

We can now set up the problem and solve it.

```julia
problem = QuantumDynamicsProblem(
    hamiltonian;
    time_step,
    maximum_time,
    alpha,
    evolution_strategy,
    style,
    initial_walkers,
    scaling_strategy,
    initiator,
    post_step_strategy,
    replica_strategy
)
simulation = solve(problem)
```

The results can be obtained as a DataFrame.

```julia
df = DataFrame(simulation)
```
