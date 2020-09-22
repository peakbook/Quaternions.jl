using Quaternions
using Test
using Random
using LinearAlgebra

@testset "Quaternions Tests" begin
  @testset "basics" begin
    for T in (Int64, Float64)
      @test real(Quaternion{T}) == T
      @test quaternion(T) == Quaternion{T}
      @test quaternion(Quaternion{T}) == Quaternion{T}
    end

    @test sprint(show, quaternion(1, 0, 0, 0)) == "1 + 0im + 0jm + 0km"
    @test sprint(show, Quaternion{Int8}(0, 0, 0, typemin(Int8))) == "0 + 0im + 0jm - 128km"

    @test real(Quaternion(1, 2, 3, 4)) == 1
    @test imagi(Quaternion(1, 2, 3, 4)) == 2
    @test imagj(Quaternion(1, 2, 3, 4)) == 3
    @test imagk(Quaternion(1, 2, 3, 4)) == 4
    @test imag(Quaternion(1, 2, 3, 4)) == Quaternion(0, 2, 3, 4)
    @test conj(Quaternion(1, 2, 3, 4)) == Quaternion(1, -2, -3, -4)
  end

  @testset "unary operator on quaternion boolean" begin
    p = [true, false]
    for components in Iterators.product(p, p, p, p)
      @test +Quaternion(components...) === Quaternion(map(+,components)...)
      @test -Quaternion(components...) === Quaternion(map(-,components)...)
    end
  end

  @testset "random values" begin
    for T in (Float16, Float32, Float64)
      @test typeof(rand(Quaternion{T})) == Quaternion{T}
      @test typeof(randn(Quaternion{T})) == Quaternion{T}
    end
    for T in (Int8, Int16, Int32, Int64, Int128)
      @test typeof(rand(Quaternion{T})) == Quaternion{T}
    end
  end

  @testset "equivalent complex matrix" begin
    x = rand(QuaternionF64, 100,100)
    @test isequal(Quaternions.equiv(Quaternions.equiv(x)), x)
  end

  @testset "linear algebra" begin
    N = 100
    M = 200
    x = rand(QuaternionF64, N, N)
    y = rand(QuaternionF64, M, N)
    atol = 1e-12

    x_inv = inv(x)
    @test x_inv isa typeof(x)
    x_mul = x_inv * x
    @test isapprox(real(x_mul), I(N); atol = atol)
    @test isapprox(imagi(x_mul), zeros(N, N); atol = atol)
    @test isapprox(imagj(x_mul), zeros(N, N); atol = atol)
    @test isapprox(imagk(x_mul), zeros(N, N); atol = atol)

    y_pinv = pinv(y)
    @test y_pinv isa typeof(y)
    y_mul = y_pinv * y
    @test isapprox(real(y_mul), I(N); atol = atol)
    @test isapprox(imagi(y_mul), zeros(N, N); atol = atol)
    @test isapprox(imagj(y_mul), zeros(N, N); atol = atol)
    @test isapprox(imagk(y_mul), zeros(N, N); atol = atol)
  end

  @testset "math functions" begin
    for T in (Float16, Float32, Float64)
      x = rand(Quaternion{T})
      @test exp(x) isa typeof(x)
      @test log(x) isa typeof(x)
      @test cos(x) isa typeof(x)
      @test sin(x) isa typeof(x)
      @test tan(x) isa typeof(x)
      @test cosh(x) isa typeof(x)
      @test sinh(x) isa typeof(x)
      @test tanh(x) isa typeof(x)
      @test sqrt(x) isa typeof(x)
      @test x^x isa typeof(x)
      @test x^2 isa typeof(x)
      @test acos(x) isa typeof(x)
      @test asin(x) isa typeof(x)
      @test atan(x) isa typeof(x)
      @test norm(x) isa T
    end
    @test round(Quaternion(0.1, 0.1, 0.1, 0.1)) == Quaternion(0.0)
    @test round(Quaternion(0.9, 0.9, 0.9, 0.9)) == Quaternion(1.0, 1.0, 1.0, 1.0)
  end

  @testset "type handling" begin
    for T in (Int8, Int16, Int32, Int64, Int128)
      @test float(Quaternion{T}) == Quaternion{float(T)}
    end
    for T in (QuaternionF16, QuaternionF32, QuaternionF64)
      @test float(T) == T
    end
  end
end
