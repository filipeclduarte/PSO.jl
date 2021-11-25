module PSO

export Pso, evaluate, optimize!
mutable struct Pso 
    # são parâmetros do construtor
    n::Int64
    dim::Int64
    w::Float64
    c1::Float64
    c2::Float64
    mini::Float64
    maxi::Float64
    # não são parâmetros do construtor
    X::Matrix{Float64}
    V::Matrix{Float64}
    pbest::Matrix{Float64}
    gbest::Vector{Float64}
    gbest_idx::Int64
    fit_values::Vector{Float64}
    pbest_values::Vector{Float64}
    gbest_value::Float64

    function Pso(n, dim, w, c1, c2, mini, maxi, f::Function)
        X = (maxi - mini) .* rand(n, dim) .+ mini
        pbest = copy(X)
        V = zeros(n, dim)
        fit_values = zeros(n)
        fit_values = fit_values_calc(f, X, fit_values)
        pbest_values = copy(fit_values)
        gbest_idx = argmin(fit_values)
        gbest = X[gbest_idx, :]
        gbest_value = fit_values[gbest_idx]
        new(n, dim, w, c1, c2, mini, maxi, X, V, pbest, gbest, gbest_idx, fit_values, pbest_values, gbest_value)
    end

end

function fit_values_calc(f, X, fit_values)
    @inbounds for i in axes(X, 1)
        fit_values[i] = f(X[i,:])
    end
    return fit_values
end

function fit_values_calc!(f, pso::Pso)
    @inbounds for i in axes(pso.X, 1)
        pso.fit_values[i] = f(pso.X[i,:])
    end
end

function update_particles!(pso::Pso)
    @inbounds for i in axes(pso.X, 1)
        r1 = rand()
        r2 = rand()    
        for j in axes(pso.X, 2)
            pso.V[i,j] = (pso.w * pso.V[i,j]) + (pso.c1 * r1 * (pso.pbest[i, j] - pso.X[i, j])) +
                    (pso.c2 * r2 * (pso.X[pso.gbest_idx, j] - pso.X[i, j]))
            pso.X[i,j] = pso.X[i,j] + pso.V[i,j]
        end
    end
    pso.X .= clamp.(pso.X, pso.mini, pso.maxi)
end

function evaluate!(f, pso::Pso)
    fit_values_calc!(f, pso)
    @inbounds for (i, el) in enumerate(pso.fit_values)
        if el < pso.pbest_values[i]
            pso.pbest[i,:] = pso.X[i,:]
            pso.pbest_values[i] = el
        end
        if el < pso.gbest_value
            pso.gbest_idx = i
            pso.gbest_value = el
        end
    end
end

function optimize!(pso::Pso, f, iterations)
    iter = 0
    while iter < iterations # todo: add avaliação da diferença entre iterações
        update_particles!(pso)
        evaluate!(f, pso)
        if iter % 10 == 0
            println("iteration: $iter | gbest_value: $(pso.gbest_value)")
        end
        iter += 1
    end
end

end # module
