assert_result(result) = @assert isa(result, Fit) "`OutlierDetectionInterface.fit` must return a tuple
 of `OutlierDetectionInterface.Model` and `OutlierDetectionInterface.Scores`"

assert_scores(scores) = @assert isa(scores, Scores) "`OutlierDetectionInterface.transform` must return  
`OutlierDetectionInterface.Scores`"

function MMI.fit(detector::UnsupervisedDetector, verbosity::Integer, X)
    result = fit(detector, X; verbosity=verbosity); assert_result(result);
    model, scores = result
    return model, nothing, (scores = scores,)
end

function MMI.fit(detector::SupervisedDetector, verbosity::Integer, X, y)
    result = fit(detector, X, y; verbosity=verbosity); assert_result(result);
    model, scores = result
    return model, nothing, (scores = scores,)
end

function MMI.transform(detector::Detector, model::DetectorModel, X)
    scores = transform(detector, model, X); assert_scores(scores);
    return scores
end

# specify scitypes
MMI.input_scitype(::Type{<:Detector}) = Union{AbstractMatrix{<:MMI.Continuous}, MMI.Table(MMI.Continuous)}
MMI.target_scitype(::Type{<:Detector}) = AbstractVector{<:Union{Missing,OrderedFactor{2}}}
MMI.output_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}
MMI.transform_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}

# helper function to generate colons according to the data dimensionality
ncolons(X::Data) = ntuple(_ -> Colon(), Val(ndims(X) - 1))

macro default_frontend(detector)
    detector = esc(detector)
    return quote
        MMI.reformat(::$detector, X::Data) = (X,)
        MMI.reformat(::$detector, X::Data, y) = (X, y)
        MMI.reformat(::$detector, X::Data, y, w) = (X, y, w) 
        MMI.reformat(::$detector, X) = (MMI.matrix(X, transpose=true),)
        MMI.reformat(::$detector, X, y) = (MMI.matrix(X, transpose=true), y)
        MMI.reformat(::$detector, X, y, w) = (MMI.matrix(X, transpose=true), y, w)
        MMI.selectrows(::$detector, I, Xmatrix) = (view(Xmatrix, ncolons(Xmatrix)..., I),)
        MMI.selectrows(::$detector, I, Xmatrix, y) = (view(Xmatrix, ncolons(Xmatrix)..., I), view(y, I))
        MMI.selectrows(::$detector, I, Xmatrix, y, w) = (view(Xmatrix, ncolons(Xmatrix)..., I), view(y, I), view(w, I))
    end
end

macro default_metadata(detector, uuid::String)
    detector = esc(detector)
    quote
        MMI.metadata_pkg($detector, package_name=string(@__MODULE__), package_uuid=$uuid,
                        package_url="https://github.com/OutlierDetectionJL/$(@__MODULE__).jl",
                        is_pure_julia=true, package_license="MIT", is_wrapper=false)
        MMI.load_path(::Type{$detector}) = string($detector)
    end
end
