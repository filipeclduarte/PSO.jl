# testando PSO
using PSO, BenchmarkTools

f(X) = X.^2 |> sum
iterations = 1000

pso = Pso(30, 5, 0.75, 1.5, 1.5, -5.12, 5.12)

X, gBest, gBestValue = optimize(pso, f, iterations)

function testPSO(n::Int64, iterations::Int64)
    for i in 1:n
        pso = Pso(30, 5, 0.75, 1.5, 1.5, -5.12, 5.12)
        X, gBest, gBestValue = optimize(pso, f, iterations)
    end
end

function testPSOthreads(n::Int64, iterations::Int64)
    Threads.@threads for i in 1:n
        pso = Pso(30, 5, 0.75, 1.5, 1.5, -5.12, 5.12)
        X, gBest, gBestValue = optimize(pso, f, iterations)
    end
end

@btime testPSO(10, 100)
@btime testPSOthreads(10, 100)
