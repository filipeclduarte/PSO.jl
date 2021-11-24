module PSO

struct Pso
    n::Int64
    dim::Int64
    w::Float64
    c1::Float64
    c2::Float64
    mini::Float64
    maxi::Float64
end

function fitValuesCalc(f, X, fitValues)
    for i in 1:size(X, 1)
        fitValues[i] = f(X[i,:])
    end
    return fitValues
end

function PsoInit(pso::Pso, f)
    X = (pso.maxi - pso.mini) .* rand(pso.n, pso.dim) .+ pso.mini
    pBest = copy(X)
    V = zeros(pso.n, pso.dim)
    fitValues = zeros(pso.n)
    fitValues = fitValuesCalc(f, X, fitValues)
    pBestValues = copy(fitValues)
    gBest = argmin(fitValues)
    gBestValue = fitValues[gBest]
    return (X, pBest, V, fitValues, pBestValues, gBest, gBestValue)
end

function atualizaPart(pso::Pso, X, pBest, V, gBest)
    for i in 1:size(X, 1)
        r1 = rand()
        r2 = rand()    
        for j in 1:size(X, 2)
            V[i,j] = (pso.w * V[i,j]) + (pso.c1 * r1 * (pBest[i, j] - X[i, j])) +
                    (pso.c2 * r2 * (X[gBest, j] - X[i, j]))
            X[i,j] = X[i,j] + V[i,j]
        end
    end
    X .= clamp.(X, pso.mini, pso.maxi)
    return (X, V)
end

function evaluate(f, X, fitValues, pBest, pBestValues, gBest, gBestValue)
    fitValues = fitValuesCalc(f, X, fitValues)
    for (i, el) in enumerate(fitValues)
        if el < pBestValues[i]
            pBest[i,:] = X[i,:]
            pBestValues[i] = el
        end
        if el < gBestValue
            gBest = i
            gBestValue = el
        end
    end
    return (pBest, pBestValues, gBest, gBestValue)
end

function optimize(pso::Pso, f, iterations)
    iter = 0
    X, pBest, V, fitValues, pBestValues, gBest, gBestValue = PsoInit(pso, f)
    while iter < iterations
        X, V = atualizaPart(pso, X, pBest, V, gBest)
        pBest, pBestValues, gBest, gBestValue = evaluate(f, X, fitValues, pBest, pBestValues, gBest, gBestValue)
        if iter % 10 == 0
            println("iteration: $iter | gBestValue: $gBestValue")
        end
        iter += 1
    end
    return (X, gBest, gBestValue)
end

export Pso, fitValuesCalc, PsoInit, atualizaPart, evaluate, optimize

end # module
