import Base: exp, log, sqrt, ^
import Base: cos, sin, tan, cosh, sinh, tanh, acos, asin, atan, acosh, asinh, atanh
# exp
function exp(q::Quaternion)
    z = sqrt(q.q1*q.q1 + q.q2*q.q2 + q.q3*q.q3)
    return exp(q.q0)*(cos(z) + imag(q)*(z != zero(q.q0) ? sin(z)/z : one(q.q0)))
end

function log(q::Quaternion)
    z = abs(imag(q))
    return log(abs(q)) + ((imag(q)!=zero(typeof(q))) ? imag(q)/z*atan2(z,q.q0) :
    q.q0 < zero(typeof(q.q0)) ? quaternion(pi*im) : zero(typeof(q)))
end

function cos(q::Quaternion)
    z = abs(imag(q))
    return cos(q.q0)*cosh(z)-imag(q)*sin(q.q0)*(z != zero(q.q0) ? sinh(z)/z : one(q.q0))
end

function sin(q::Quaternion)
    z = abs(imag(q))
    return sin(q.q0)*cosh(z)+imag(q)*cos(q.q0)*(z != zero(q.q0) ? sinh(z)/z : one(q.q0))
end

function tan(q::Quaternion)
    return sin(q)/cos(q)
end

function cosh(q::Quaternion)
    return (exp(q)+exp(-q))/2
end

function sinh(q::Quaternion)
    return (exp(q)-exp(-q))/2
end

function tanh(q::Quaternion)
    return sinh(q)/cosh(q)
end

function sqrt(q::Quaternion)
    return exp(0.5*log(q))
end

function (^)(q::Quaternion, p::AbstractFloat)
    return exp(p*log(q))
end

function (^)(q::Quaternion, p::Quaternion)
    return exp(p*log(q))
end

function acos(q::Quaternion)
    x = imag(q)
    y = abs(x)
    z = y!=zero(q.q0) ? x/y : quaternion(one(q.q0)*im)

    return -z*log(q + z*sqrt(one(q.q0) - q*q))
    # return -z*log(q - z*sqrt(one(q.q0) - q*q))
end

function asin(q::Quaternion)
    x = imag(q)
    y = abs(x)
    z = y!=zero(q.q0) ? x/y : quaternion(one(q.q0)*im)

    return -z*log(z*q + sqrt(one(q.q0) - q*q))
    #= return -z*log(z*q - sqrt(one(q.q0) - q*q)) =#
end

function atan(q::Quaternion)
    x = imag(q)
    y = abs(x)
    z = y!=zero(q.q0) ? x/y : quaternion(one(q.q0)*im)

    return -0.5*z*log((one(q.q0)+z*q)/(one(q.q0)-z*q))
end

