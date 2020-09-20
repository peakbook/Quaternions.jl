# Quaternions

[![Build Status](https://travis-ci.org/peakbook/Quaternions.jl.svg?branch=master)](https://travis-ci.org/peakbook/Quaternions.jl)
[![Coverage Status](https://coveralls.io/repos/peakbook/Quaternions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/peakbook/Quaternions.jl?branch=master)
[![codecov](https://codecov.io/gh/peakbook/Quaternions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/peakbook/Quaternions.jl)

Quaternion package for Julia

A Julia module for Quaternions.

```julia
struct Quaternion{T<:Real} <: Number
    q0::T
    q1::T
    q2::T
    q3::T
end

exp(Quaternion)::Quaternion
log(Quaternion)::Quaternion
cos(Quaternion)::Quaternion
sin(Quaternion)::Quaternion
tan(Quaternion)::Quaternion
cosh(Quaternion)::Quaternion
sinh(Quaternion)::Quaternion
tanh(Quaternion)::Quaternion
sqrt(Quaternion)::Quaternion
^(Quaternion, AbstractFloat)::Quaternion
^(Quaternion, Quaternion)::Quaternion
acos(Quaternion)::Quaternion
asin(Quaternion)::Quaternion
atan(Quaternion)::Quaternion
```
