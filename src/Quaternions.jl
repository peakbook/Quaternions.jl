__precompile__()

module Quaternions
import Base: convert, promote_rule, show, real, imag, conj, abs, abs2, inv, rand, randn
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
convert{T<:Real}(::Type{Quaternion{T}}, x::Real) = Quaternion(convert(T,x))
convert{T<:Real}(::Type{Quaternion{T}}, z::Complex) = Quaternion(convert(Complex{T},z))
convert{T<:Real}(::Type{Quaternion{T}}, q::Quaternion) = Quaternion{T}(convert(T,q.q0), convert(T,q.q1), convert(T,q.q2), convert(T,q.q3))
convert{T<:Real}(::Type{Quaternion{T}}, q::Quaternion{T}) = q
convert{T<:Real}(::Type{T}, q::Quaternion) = (q.q1 == zero(q.q1) && q.q2 == zero(q.q1) && q.q3 == zero(q.q1) ? convert(T,q.q0) : throw(InexactError()))

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

function show(io::IO, z::Quaternion)
    pm(z) = z < zero(z) ? " - $(-z)" : " + $z"
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

quaternion{T<:Quaternion}(x::AbstractArray{T}) = x
quaternion{T<:Complex}(x::AbstractArray{T}) = copy!(similar(x, Quaternion{real(eltype(x))}), x)
quaternion{T<:Real}(x::AbstractArray{T}) = copy!(similar(x, Quaternion{eltype(x)}), x)

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

para(p, q::Quaternion) = (p - q*p*q)/2
perp(p, q::Quaternion) = (p + q*p*q)/2

rand{T<:Real}(::Type{Quaternion{T}}) = quaternion(rand(T),rand(T),rand(T),rand(T))
randn{T<:Real}(::Type{Quaternion{T}}) = quaternion(randn(),randn(),randn(),randn())
rand{T<:Real}(r::AbstractRNG, ::Type{Quaternion{T}}) = quaternion(rand(r, T), rand(r, T), rand(r, T), rand(r, T))
randn{T<:Real}(r::AbstractRNG, ::Type{Quaternion{T}}) = quaternion(randn(r), randn(r), randn(r), randn(r))

real{T<:Real}(::Type{Quaternion{T}}) = T
quaternion{T<:Real}(::Type{T}) = Quaternion{T}
quaternion{T<:Real}(::Type{Quaternion{T}}) = Quaternion{T}

const jm = Quaternion(false,false,true,false)
const km = Quaternion(false,false,false,true)

function equiv!{T<:Real}(Y::AbstractArray{Complex{T}}, X::AbstractArray{Quaternion{T}})
    @assert size(X,1)<<1 == size(Y,1)
    @assert size(X,2)<<1 == size(Y,2)

    dtype = real(eltype(X))
    M = size(X,1)
    N = size(X,2)
    Xr = reinterpret(dtype, X, (M<<2,N))
    Yr = reinterpret(dtype, Y, (M<<2,N<<1))

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

function equiv{T<:Quaternion}(X::AbstractArray{T})
    Y = Array(Complex{real(eltype(X))}, (size(X,1)<<1, size(X,2)<<1))
    return equiv!(Y, X)
end

function equiv!{T<:Real}(Y::AbstractArray{Quaternion{T}}, X::AbstractArray{Complex{T}})
    @assert size(X,1)>>1 == size(Y,1)
    @assert size(X,2)>>1 == size(Y,2)

    dtype = real(eltype(X))
    M = size(X,1)
    N = size(X,2)
    Xr = reinterpret(dtype, X, (M<<1,N))
    Yr = reinterpret(dtype, Y, (M<<1,N>>1))

    Yr[1:4:end,:] = Xr[1:2:end>>1,1:end>>1] 
    Yr[2:4:end,:] = Xr[2:2:end>>1,1:end>>1] 
    Yr[3:4:end,:] = Xr[1:2:end>>1,1+end>>1:end] 
    Yr[4:4:end,:] = Xr[2:2:end>>1,1+end>>1:end] 

    return Y
end

function equiv{T<:Complex}(X::AbstractArray{T})
    Y = Array(Quaternion{real(eltype(X))}, (size(X,1)>>1, size(X,2)>>1))
    return equiv!(Y, X)
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

