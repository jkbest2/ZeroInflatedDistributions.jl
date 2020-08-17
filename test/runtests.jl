using ZeroInflatedLikelihoods
using Distributions
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
end
