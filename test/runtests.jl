using ZeroInflatedLikelihoods
using Distributions
using QuadGK
using Random
using Test

@testset "ZeroInflatedLikelihoods.jl" begin
    @testset "Link functions" begin
        p1 = 0
        p2 = 1
        # Logit-log link tests
        ll = LogitLogLink()
        @test encprob(ll, p1, p2) == 1//2
        @test posrate(ll, p1, p2) == Float64(ℯ)
        # Poisson link tests
        pl = PoissonLink()
        @test isone(pl.offset)
        @test_throws DomainError PoissonLink(-1)
        @test encprob(pl, p1, p2) == 0.6321205588285577
        @test posrate(pl, p1, p2) == 4.300258535328371
        # Identity link tests
        il = IdentityLink()
        @test_throws DomainError encprob(il, -1)
        @test_throws DomainError encprob(il, 1.1)
        @test encprob(il, 0.5) == 0.5
        @test posrate(il, 1.0) == 1
    end

    @testset "Zero-inflated likelihood constructors" begin
        p1 = 0
        p2 = 1
        disp = 0.5
        ll = LogitLogLink()
        pl = PoissonLink()
        # Test inner constructor
        zil0 = ZeroInflatedDistribution(Bernoulli(0.5), LogNormal(0.0, 1.0))
        @test loglikelihood(zil0, 0) == log(0.5)
        @test loglikelihood(zil0, 1) == logpdf(LogNormal(0.0, 1.0), 1) + log(0.5)

        # Without bias correction, posrate is the median of the log-normal
        zil1 = ZeroInflatedDistribution(ll, LogNormal, p1, p2, disp; biascorrect = false)
        @test median(zil1.posdist) == posrate(ll, p1, p2)

        # With bias correction (default) it is the mean
        zil2 = ZeroInflatedDistribution(ll, LogNormal, p1, p2, disp)
        @test mean(zil2.posdist) == posrate(ll, p1, p2)

        zil3 = ZeroInflatedDistribution(pl, Gamma, p1, p2, disp)
        @test mean(zil3.posdist) == posrate(pl, p1, p2)
        @test std(zil3.posdist) == disp

        zil4 = ZeroInflatedDistribution(pl, InverseGamma, p1, p2, disp)
        @test mean(zil4.posdist) == posrate(pl, p1, p2)
        @test std(zil4.posdist) == disp

        zil5 = ZeroInflatedDistribution(pl, InverseGaussian, p1, p2, disp)
        @test mean(zil5.posdist) == posrate(pl, p1, p2)
        @test shape(zil5.posdist) == disp
    end

    @testset "Zero-inflated log-likelihoods" begin
        zil = ZeroInflatedDistribution(Bernoulli(0.5), LogNormal(0.0, 1.0))
        @test loglikelihood(zil, 0) == log(0.5)
        @test loglikelihood(zil, 1) == logpdf(LogNormal(0.0, 1.0), 1) + log(0.5)
    end

    @testset "Random zero-inflated data generation" begin
        n = 1_000
        zil = ZeroInflatedDistribution(Bernoulli(0.5), LogNormal(-1/2, 1.0))
        x = rand(zil, n)
        @test any(x .== 0)
        @test all(x .≥ 0)
    end

    @testset "Zero-inflated distribution statistics" begin
        zil = ZeroInflatedDistribution(Bernoulli(0.25), LogNormal(-1/2, 1.0))
        zil0 = ZeroInflatedDistribution(Bernoulli(0), LogNormal(-1/2, 1.0))
        zil1 = ZeroInflatedDistribution(Bernoulli(1), LogNormal(-1/2, 1.0))
        zilgam = ZeroInflatedDistribution(Bernoulli(0.15), Gamma(2, 4))

        @testset "Zero-inflated distribution supports" begin
            @test !insupport(zil, -1)
            @test all(minimum.([zil, zil0, zil1]) .== 0)
            @test all(maximum.([zil, zil0, zil1]) .== Inf)
        end


        @testset "Test means and variances via simulation" begin
            Random.seed!(13579)
            n = 10_000

            x = rand(zil, n)
            x0 = rand(zil0, n)
            x1 = rand(zil1, n)

            sd = std(zil) / sqrt(n)
            sd1 = std(zil1) / sqrt(n)

            @test isapprox(mean(x), 0.25, atol = 2 * sd)
            @test all(x0 .== 0)
            @test isapprox(mean(x1), 1, atol = 2 * sd1)

            # @test isapprox(var(zil), var(x))
            @test var(zil0) == 0
            @test var(zil1) == var(zil1.posdist)
        end

        @testset "Test means and variances via quadrature" begin
            # 𝔼[X]
            quadex = quadgk(x -> x * pdf(zil, x), 0, Inf)
            quadexgam = quadgk(x -> x * pdf(zilgam, x), 0, Inf)
            # 𝔼[X²]
            quadex2 = quadgk(x -> x^2 * pdf(zil, x), 0, Inf)
            quadexgam2 = quadgk(x -> x^2 * pdf(zilgam, x), 0, Inf)
            # 𝕍[X] = 𝔼[X^2] - 𝔼[X]^2
            quadvar = quadex2[1] - mean(zil)^2
            quadvargam = quadexgam2[1] - mean(zilgam)^2

            @test isapprox(quadex[1], mean(zil), atol = quadex[2])
            @test isapprox(quadvar, var(zil), atol = quadex2[2])
            @test isapprox(quadexgam[1], mean(zilgam), atol = quadexgam[2])
            @test isapprox(quadvargam, var(zilgam), atol = quadexgam2[2])
        end
    end
end
