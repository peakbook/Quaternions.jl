"""
    Quaternions

A package for quaternion arithmetic.

"""
module Quaternions
import Base: convert, promote_rule, show, real, imag, conj, abs, abs2, inv, rand, randn
import Base: +, -, /, *, &, ⊻, |
import Base: inv, float, isreal, isinteger, isfinite, isnan, isinf, iszero, isone
import Base: widen

import Random
import LinearAlgebra: pinv, norm

export Quaternion, QuaternionF64, QuaternionF32, QuaternionF16
export quaternion, imagi, imagj, imagk, jm, km, ispure
export wedge, antiwedge, perp, para

"""
    Quaternion{T<:Real} <: Number

Quaternion number type with real and imaginary parts of type `T`.

`QuaternionF16`, `QuaternionF32` and `QuaternionF64` are aliases for
`Quaternion{Float16}`, `Quaternion{Float32}` and `Quaternion{Float64}` respectively.
"""
struct Quaternion{T<:Real} <: Number
    q0::T
    q1::T
    q2::T
    q3::T
end

include("qmath.jl")

Quaternion(q0::Real,q1::Real,q2::Real,q3::Real) = Quaternion(promote(q0,q1,q2,q3)...)
Quaternion(x::Real) = Quaternion(x,zero(x),zero(x),zero(x))
Quaternion(x::Irrational) = Quaternion(float(x))
Quaternion(z::Complex) = Quaternion(real(z),imag(z),zero(real(z)),zero(real(z)))

const QuaternionF64 = Quaternion{Float64}
const QuaternionF32 = Quaternion{Float32}
const QuaternionF16 = Quaternion{Float16}

convert(::Type{Quaternion}, x::Real) = Quaternion(x)
convert(::Type{Quaternion}, z::Complex) = Quaternion(z)
convert(::Type{Quaternion{T}}, x::Real) where T<:Real = Quaternion(convert(T,x))
convert(::Type{Quaternion{T}}, z::Complex) where T<:Real = Quaternion(convert(Complex{T},z))
convert(::Type{Quaternion{T}}, q::Quaternion) where T<:Real  = Quaternion{T}(convert(T,q.q0), convert(T,q.q1), convert(T,q.q2), convert(T,q.q3))
convert(::Type{Quaternion{T}}, q::Quaternion{T}) where T<:Real = q
convert(::Type{T}, q::Quaternion) where T<:Real = (iszero(q.q1) && iszero(q.q2) && iszero(q.q3)) ? convert(T,q.q0) : throw(InexactError())

promote_rule(::Type{Quaternion{T}}, ::Type{T}) where T<:Real = Quaternion{T}
promote_rule(::Type{Quaternion}, ::Type{T}) where T<:Real = Quaternion
promote_rule(::Type{Quaternion{T}}, ::Type{Complex{T}}) where T<:Real = Quaternion{T}
promote_rule(::Type{Quaternion}, ::Type{Complex{T}}) where T<:Real = Quaternion
promote_rule(::Type{Quaternion{T}}, ::Type{S}) where {T<:Real, S<:Real} = Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Complex{S}}) where {T<:Real, S<:Real} = Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{S}}) where {T<:Real, S<:Real} = Quaternion{promote_type(T,S)}

quaternion(q0,q1,q2,q3) = Quaternion(q0,q1,q2,q3)
quaternion(x) = Quaternion(x)
quaternion(z::Complex) = Quaternion(z)
quaternion(q::Quaternion) = q

widen(::Type{Quaternion{T}}) where {T} = Quaternion{widen(T)}
function show(io::IO, q::Quaternion)
    compact = get(io, :compact, false)
    show(io, q.q0)
    p(i, unit) = begin
        if signbit(i) && !isnan(i)
            print(io, compact ? "-" : " - ")
            if isa(i,Signed) && !isa(i,BigInt) && i == typemin(typeof(i))
                show(io, -widen(i))
            else
                show(io, -i)
            end
        else
            print(io, compact ? "+" : " + ")
            show(io, i)
        end
        if !(isa(i,Integer) && !isa(i,Bool) || isa(i,AbstractFloat) && isfinite(i))
            print(io, "*")
        end
        print(io, unit)
    end
    p(q.q1, "im")
    p(q.q2, "jm")
    p(q.q3, "km")
end

show(io::IO, q::Quaternion{Bool}) =
    print(io, q == im ? "im" :
              q == jm ? "jm" :
              q == km ? "km" :
          "Quaternion($(q.q0),$(q.q1),$(q.q2),$(q.q3))")

function quaternion(A::Array{S}, B::Array{T}, C::Array{U}, D::Array{V}) where {S<:Real, T<:Real, U<:Real, V<:Real}
    if !(size(A)==size(B)==size(C)==size(D)); error("argument dimensions must match"); end
    F = similar(A, typeof(quaternion(zero(S),zero(T),zero(U),zero(V))))
    for i=1:length(A)
        @inbounds F[i] = quaternion(A[i], B[i], C[i], D[i])
    end
    return F
end

quaternion(x::AbstractArray{T}) where T<:Quaternion = x
quaternion(x::AbstractArray{T}) where T<:Complex = copy!(similar(x, Quaternion{real(eltype(x))}), x)
quaternion(x::AbstractArray{T}) where T<:Real = copy!(similar(x, Quaternion{eltype(x)}), x)

real(z::Quaternion) = z.q0
imagi(z::Quaternion) = z.q1
imagj(z::Quaternion) = z.q2
imagk(z::Quaternion) = z.q3
imagi(z::Real) = zero(z)
imagj(z::Real) = zero(z)
imagk(z::Real) = zero(z)
imag(z::Quaternion) = Quaternion(zero(z.q0),z.q1,z.q2,z.q3)

imagi(x::AbstractVector{T}) where T<:Real = zero(x)
imagj(x::AbstractVector{T}) where T<:Real = zero(x)
imagk(x::AbstractVector{T}) where T<:Real = zero(x)

ispure(z::Quaternion)::Bool = iszero(z.q0)

for fn in (:imagi,:imagj,:imagk)
    @eval begin
        ($fn)(A::AbstractArray) = map(($fn),A)
    end
end

conj(z::Quaternion) = Quaternion(z.q0, -z.q1, -z.q2, -z.q3)
abs(z::Quaternion) = sqrt(z.q0*z.q0 + z.q1*z.q1 + z.q2*z.q2 + z.q3*z.q3)
abs2(z::Quaternion) = z.q0*z.q0 + z.q1*z.q1 + z.q2*z.q2 + z.q3*z.q3
inv(z::Quaternion) = conj(z)/abs2(z)

(+)(z::Quaternion) = Quaternion(+z.q0, +z.q1, +z.q2, +z.q3)
(-)(z::Quaternion) = Quaternion(-z.q0, -z.q1, -z.q2, -z.q3)
(/)(z::Quaternion, x::Real) = Quaternion(z.q0/x, z.q1/x, z.q2/x, z.q3/x)
(+)(z::Quaternion, w::Quaternion) = Quaternion(z.q0 + w.q0, z.q1 + w.q1,
                                                           z.q2 + w.q2, z.q3 + w.q3)
(-)(z::Quaternion, w::Quaternion) = Quaternion(z.q0 - w.q0, z.q1 - w.q1,
                                                           z.q2 - w.q2, z.q3 - w.q3)
(*)(z::Quaternion, w::Quaternion) = Quaternion(z.q0*w.q0 - z.q1*w.q1 - z.q2*w.q2 - z.q3*w.q3,
                                                           z.q0*w.q1 + z.q1*w.q0 + z.q2*w.q3 - z.q3*w.q2,
                                                           z.q0*w.q2 - z.q1*w.q3 + z.q2*w.q0 + z.q3*w.q1,
                                                           z.q0*w.q3 + z.q1*w.q2 - z.q2*w.q1 + z.q3*w.q0)
(/)(z::Quaternion, w::Quaternion) = z*inv(w)

# element wise multiplication
mul_ew(z1::Quaternion, z2::Quaternion)::Quaternion = Quaternion(z1.q0*z2.q0,z1.q1*z2.q1,z1.q2*z2.q2,z1.q3*z2.q3)
(&)(z1::Quaternion, z2::Quaternion)::Quaternion = mul_ew(z1,z2)
(&)(z::Quaternion, x::Real) = x*z
(&)(z::Real, x::Quaternion) = x*z

wedge(p::Quaternion,q::Quaternion) = (p*q-q*p)/2
antiwedge(p::Quaternion,q::Quaternion) = (p*q+q*p)/2
(⊻)(z1::Quaternion, z2::Quaternion) = wedge(z1,z2)
(|)(z1::Quaternion, z2::Quaternion) = antiwedge(z1,z2)

para(p, q::Quaternion) = (p - q*p*q)/2
perp(p, q::Quaternion) = (p + q*p*q)/2

#rand(r::Random.AbstractRNG, ::Random.SamplerType{Quaternion{T}}) where {T<:Real} = Quaternion(rand(r, T), rand(r, T), rand(r,T), rand(r,T))
#randn(r::Random.AbstractRNG, ::Type{Quaternion{T}}) where {T<:AbstractFloat} = Quaternion(0.5*randn(r, T), 0.5*randn(r, T), 0.5*randn(r,T), 0.5*randn(r,T))
function Base.rand(r::Random.AbstractRNG, ::Random.SamplerType{Quaternion{T}}) where {T <: Number}
    # http://planning.cs.uiuc.edu/node198.html
    # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L260
    tau = 2.0 * pi
    r1 = rand(r, T)
    r2 = rand(r, T)
    r3 = rand(r, T)
    q1 = sqrt(one(T) - r1) * (sin(tau * r2))
    q2 = sqrt(one(T) - r1) * (cos(tau * r2))
    q3 = sqrt(r1)       * (sin(tau * r3))
    q4 = sqrt(r1)       * (cos(tau * r3))
    Quaternion(q1, q2, q3, q4)
end

function Base.randn(r::Random.AbstractRNG, ::Type{Quaternion{T}}) where {T <: Number}
     Quaternion(randn(r, T), randn(r, T), randn(r, T), randn(r, T))*0.5
end

real(::Type{Quaternion{T}}) where T<:Real = T
quaternion(::Type{T}) where T<:Real = Quaternion{T}
quaternion(::Type{Quaternion{T}}) where T<:Real = Quaternion{T}

"""
    jm

The quaternion imaginary unit `j`.

# Examples
```jldoctest
julia> jm * jm
-1 + 0im + 0jm + 0km
```
"""
const jm = Quaternion(false,false,true,false)

"""
    km

The quaternion imaginary unit `k`.

# Examples
```jldoctest
julia> km * km
-1 + 0im + 0jm + 0km
```
"""
const km = Quaternion(false,false,false,true)

function equiv!(Y::AbstractArray{Complex{T}}, X::AbstractArray{Quaternion{T}}) where T<:Real
    @assert size(X,1)<<1 == size(Y,1)
    @assert size(X,2)<<1 == size(Y,2)

    dtype = real(eltype(X))
    M = size(X,1)
    N = size(X,2)
    Xr = reshape(reinterpret(dtype, X), (M<<2,N))
    Yr = reshape(reinterpret(dtype, Y), (M<<2,N<<1))

    Yr[1:2:end>>1,1:end>>1] = Xr[1:4:end,:]
    Yr[2:2:end>>1,1:end>>1] = Xr[2:4:end,:]
    Yr[1+end>>1:2:end,1+end>>1:end] = Xr[1:4:end,:]
    Yr[2+end>>1:2:end,1+end>>1:end] = -Xr[2:4:end,:]
    Yr[1:2:end>>1,1+end>>1:end] = Xr[3:4:end,:]
    Yr[2:2:end>>1,1+end>>1:end] = Xr[4:4:end,:]
    Yr[1+end>>1:2:end,1:end>>1] = -Xr[3:4:end,:]
    Yr[2+end>>1:2:end,1:end>>1] = Xr[4:4:end,:]

    return Y
end

function equiv(X::AbstractArray{T}) where T<:Quaternion
    Y = Array{Complex{real(eltype(X))}}(undef, size(X,1)<<1, size(X,2)<<1)
    return equiv!(Y, X)
end

function equiv!(Y::AbstractArray{Quaternion{T}}, X::AbstractArray{Complex{T}}) where T<:Real
    @assert size(X,1)>>1 == size(Y,1)
    @assert size(X,2)>>1 == size(Y,2)

    dtype = real(eltype(X))
    M = size(X,1)
    N = size(X,2)
    Xr = reshape(reinterpret(dtype, X), (M<<1,N))
    Yr = reshape(reinterpret(dtype, Y), (M<<1,N>>1))

    Yr[1:4:end,:] = Xr[1:2:end>>1,1:end>>1]
    Yr[2:4:end,:] = Xr[2:2:end>>1,1:end>>1]
    Yr[3:4:end,:] = Xr[1:2:end>>1,1+end>>1:end]
    Yr[4:4:end,:] = Xr[2:2:end>>1,1+end>>1:end]

    return Y
end

function equiv(X::AbstractArray{T}) where T<:Complex
    Y = Array{Quaternion{real(eltype(X))}}(undef, size(X,1)>>1, size(X,2)>>1)
    return equiv!(Y, X)
end

for fn in (:pinv, :inv)
    @eval begin
        function ($fn)(A::Array{Quaternion{T},2}) where T<:Real
            cmat = equiv(A)
            ret = ($fn)(cmat)
            return equiv(ret)
        end
    end
end

isreal(q::Quaternion) = iszero(imagi(q)) & iszero(imagj(q)) & iszero(imagk(q))
isinteger(q::Quaternion) = isreal(q) & isinteger(real(q))
isfinite(q::Quaternion) = isfinite(real(q)) & isfinite(imagi(q)) & isfinite(imagj(q)) & isfinite(imagk(q))
isnan(q::Quaternion) = isnan(real(q)) | isnan(imagi(q)) | isnan(imagj(q)) | isnan(imagk(q))
isinf(q::Quaternion) = isinf(real(q)) | isinf(imagi(q)) | isinf(imagj(q)) | isinf(imagk(q))
iszero(q::Quaternion) = iszero(real(q)) & iszero(imagi(q)) & iszero(imagj(q)) & iszero(imagk(q))
isone(q::Quaternion) = isone(real(q)) & iszero(imagi(q)) & iszero(imagj(q)) & iszero(imagk(q))

end
