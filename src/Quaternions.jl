VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Quaternions
using Compat
import Base: int, convert, promote_rule, show, real, imag, conj, abs, abs2, inv, rand, randn
import Base: +, -, /, *, &, $, |
import Base: inv, pinv, float

export Quaternion, Quaternion256, Quaternion128, Quaternion64
export quaternion, imagi, imagj, imagk, jm, km
export wedge, antiwedge, perp, para

immutable Quaternion{T<:Real} <: Number
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

typealias Quaternion256 Quaternion{Float64}
typealias Quaternion128 Quaternion{Float32}
typealias Quaternion64 Quaternion{Float16}

convert(::Type{Quaternion}, x::Real) = Quaternion(x)
convert(::Type{Quaternion}, z::Complex) = Quaternion(z)
convert(::Type{Quaternion}, z::Complex{Bool}) = Quaternion(real(z)+z)
convert{T<:Real}(::Type{Quaternion{T}}, x::Real) = Quaternion(x)
convert{T<:Real}(::Type{Quaternion{T}}, z::Complex) = Quaternion(z)
convert{T<:Real}(::Type{Quaternion{T}}, z::Complex{Bool}) = Quaternion(real(z)+z)
convert{T<:Real}(::Type{Quaternion{T}}, q::Quaternion) = Quaternion{T}(convert(T,q.q0), convert(T,q.q1), convert(T,q.q2), convert(T,q.q3))
convert{T<:Real}(::Type{Quaternion{T}}, q::Quaternion{T}) = q
convert{T<:Real}(::Type{T}, q::Quaternion) = (q.q1==0 && q.q2 == 0 && q.q3 == 0 ? convert(T,q.q0) : throw(InexactError()))

promote_rule{T<:Real}(::Type{Quaternion{T}}, ::Type{T}) = Quaternion{T}
promote_rule{T<:Real}(::Type{Quaternion}, ::Type{T}) = Quaternion
promote_rule{T<:Real}(::Type{Quaternion{T}}, ::Type{Complex{T}}) = Quaternion{T}
promote_rule{T<:Real}(::Type{Quaternion}, ::Type{Complex{T}}) = Quaternion
promote_rule{T<:Real,S<:Real}(::Type{Quaternion{T}}, ::Type{S}) = Quaternion{promote_type(T,S)}
promote_rule{T<:Real,S<:Real}(::Type{Quaternion{T}}, ::Type{Complex{S}}) = Quaternion{promote_type(T,S)}
promote_rule{T<:Real,S<:Real}(::Type{Quaternion{T}}, ::Type{Quaternion{S}}) = Quaternion{promote_type(T,S)}

quaternion(q0,q1,q2,q3) = Quaternion(q0,q1,q2,q3)
quaternion(x) = Quaternion(x)
quaternion(z::Complex) = Quaternion(z)
quaternion(q::Quaternion) = q 

quaternion256(q0::Float64,q1::Float64,q2::Float64,q3::Float64) = Quaternion{Float64}(q0,q1,q2,q3)
quaternion256(q0::Real,q1::Real,q2::Real,q3::Real) = quaternion256(float64(q0),float64(q1),float64(q2),float64(q3))
quaternion128(q0::Float32,q1::Float32,q2::Float32,q3::Float32) = Quaternion{Float32}(q0,q1,q2,q3)
quaternion128(q0::Real,q1::Real,q2::Real,q3::Real) = quaternion128(float32(q0),float32(q1),float32(q2),float32(q3))
quaternion64(q0::Float16,q1::Float16,q2::Float16,q3::Float16) = Quaternion{Float16}(q0,q1,q2,q3)
quaternion64(q0::Real,q1::Real,q2::Real,q3::Real) = quaternion64(float16(q0),float16(q1),float16(q2),float16(q3))

#for fn in _numeric_conversion_func_names
for fn in (:int,:integer,:signed,:int8,:int16,:int32,:int64,:int128,
    :uint,:unsigned,:uint8,:uint16,:uint32,:uint64,:uint128,
    :float,:float16,:float32,:float64)
    @eval $fn(q::Quaternion) = Quaternion($fn(q.q0),$fn(q.q1),$fn(q.q2),$fn(q.q3))
end

function show(io::IO, z::Quaternion)
    pm(z) = z < 0 ? " - $(-z)" : " + $z"
    print(io, z.q0, pm(z.q1), "i", pm(z.q2), "j", pm(z.q3), "k")
end

function quaternion{S<:Real,T<:Real,U<:Real,V<:Real}(A::Array{S}, B::Array{T}, C::Array{U}, D::Array{V})
    if !(size(A)==size(B)==size(C)==size(D)); error("argument dimensions must match"); end
    F = similar(A, typeof(quaternion(zero(S),zero(T),zero(U),zero(V))))
    for i=1:length(A)
        @inbounds F[i] = quaternion(A[i], B[i], C[i], D[i])
    end
    return F
end

for (f,t) in ((:quaternion64, Quaternion64),
    (:quaternion128, Quaternion128),
    (:quaternion256, Quaternion256))
    @eval ($f)(x::AbstractArray{$t}) = x
    @eval ($f)(x::AbstractArray) = copy!(similar(x,$t), x)
end

quaternion{T<:Quaternion}(x::AbstractArray{T}) = x
quaternion(x::AbstractArray) = copy!(similar(x,typeof(quaternion(one(eltype(x))))), x)

real(z::Quaternion) = z.q0
imagi(z::Quaternion) = z.q1
imagj(z::Quaternion) = z.q2
imagk(z::Quaternion) = z.q3
imagi(z::Real) = zero(z)
imagj(z::Real) = zero(z)
imagk(z::Real) = zero(z)
imag(z::Quaternion) = Quaternion(zero(z.q0),z.q1,z.q2,z.q3)

imagi{T<:Real}(x::AbstractVector{T}) = zero(x)
imagj{T<:Real}(x::AbstractVector{T}) = zero(x)
imagk{T<:Real}(x::AbstractVector{T}) = zero(x)

for fn in (:imagi,:imagj,:imagk)
    @eval begin
        ($fn)(A::AbstractArray) = map(($fn),A)
    end
end

conj(z::Quaternion) = Quaternion(z.q0, -z.q1, -z.q2, -z.q3)
abs(z::Quaternion) = sqrt(z.q0*z.q0 + z.q1*z.q1 + z.q2*z.q2 + z.q3*z.q3)
abs2(z::Quaternion) = z.q0*z.q0 + z.q1*z.q1 + z.q2*z.q2 + z.q3*z.q3
inv(z::Quaternion) = conj(z)/abs2(z)

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
mul_ew(z1::Quaternion, z2::Quaternion) = Quaternion(z1.q0*z2.q0,z1.q1*z2.q1,z1.q2*z2.q2,z1.q3*z2.q3)
(&)(z1::Quaternion, z2::Quaternion) = mul_ew(z1,z2)
(&)(z::Quaternion, x::Real) = x*z
(&)(z::Real, x::Quaternion) = x*z

wedge(p::Quaternion,q::Quaternion) = (p*q-q*p)/2
antiwedge(p::Quaternion,q::Quaternion) = (p*q+q*p)/2
($)(z1::Quaternion, z2::Quaternion) = wedge(z1,z2)
(|)(z1::Quaternion, z2::Quaternion) = antiwedge(z1,z2)

para(p, q::Quaternion) = 0.5*(p - q*p*q)
perp(p, q::Quaternion) = 0.5*(p + q*p*q)

rand{T<:Real}(::Type{Quaternion{T}}) = quaternion(rand(T),rand(T),rand(T),rand(T))
randn{T<:Real}(::Type{Quaternion{T}}) = quaternion(randn(),randn(),randn(),randn())
rand{T<:Real}(r::AbstractRNG, ::Type{Quaternion{T}}) = quaternion(rand(r, T), rand(r, T), rand(r, T), rand(r, T))
randn{T<:Real}(r::AbstractRNG, ::Type{Quaternion{T}}) = quaternion(randn(r), randn(r), randn(r), randn(r))

const jm = Quaternion(false,false,true,false)
const km = Quaternion(false,false,false,true)

function q2c{T<:Real}(q::Quaternion{T})
    return complex(real(q),imagi(q))
end

function q2cj{T<:Real}(q::Quaternion{T})
    return complex(imagj(q),imagk(q))
end

function c2q{T<:Real}(q1::Complex{T},q2::Complex{T})
    return quaternion(real(q1),imag(q1),real(q2),imag(q2))
end

for fn in (:q2c,:q2cj)
    @eval begin
        ($fn)(A::AbstractArray) = map(($fn),A)
    end
end

c2q(A::AbstractArray, B::AbstractArray) = map(c2q,A,B)

function equiv{T<:Real}(qmat::AbstractArray{Quaternion{T}})
    A = q2c(qmat)
    B = q2cj(qmat)

    return hcat(vcat(A,-conj(B)),vcat(B,conj(A)))
end

function equiv{T<:Real}(cmat::AbstractArray{Complex{T}})
    A = cmat[1:end>>1,1:end>>1]
    B = cmat[1:end>>1,end>>1+1:end]

    return A+B*jm
end

for fn in (:pinv, :inv)
    @eval begin
        function ($fn){T<:Real}(A::Array{Quaternion{T},2})
            cmat = equiv(A)
            ret = ($fn)(cmat)
            return equiv(ret)
        end
    end
end

end # module

