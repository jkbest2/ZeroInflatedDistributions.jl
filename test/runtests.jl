using ZeroInflatedLikelihoods
using Test

@testset "ZeroInflatedLikelihoods.jl" begin
    @testset "Link functions" begin
        p1 = 0
        p2 = 1
        # Logit-log link tests
        ll = LogitLogLink()
        @test encprob(ll, p1, p2) == 1//2
        @test posrate(ll, p1, p2) == Float64(ℯ)
        @test posrate(ll, p1, p2; bias = 1) == Float64(ℯ) - 1
        # Poisson link tests
        pl = PoissonLink()
        @test isone(pl.offset)
        @test_throws DomainError PoissonLink(-1)
        @test encprob(pl, p1, p2) == 0.6321205588285577
        @test posrate(pl, p1, p2) == 4.300258535328371
        @test posrate(pl, p1, p2; bias = 1) == 4.300258535328371 - 1
        # Identity link tests
        il = IdentityLink()
        @test_throws DomainError encprob(il, -1)
        @test_throws DomainError encprob(il, 1.1)
        @test encprob(il, 0.5) == 0.5
        @test posrate(il, 1.0) == 1
        @test posrate(il, 2.0; bias = 1) == 1
    end
    
    # Write your tests here.
end
