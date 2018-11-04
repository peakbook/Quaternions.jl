# Quaternions

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
