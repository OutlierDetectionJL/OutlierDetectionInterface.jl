assert_result(result) = @assert isa(result, Fit) "`OutlierDetectionInterface.fit` must return a tuple
 of `OutlierDetectionInterface.Model` and `OutlierDetectionInterface.Scores`"

assert_scores(scores) = @assert isa(scores, Scores) "`OutlierDetectionInterface.transform` must return  
`OutlierDetectionInterface.Scores`"

function MMI.fit(detector::UnsupervisedDetector, verbosity, X)
    result = fit(detector, X; verbosity=verbosity)
    assert_result(result)
    model, scores = result
    return FitResult(model, scores), nothing, (;scores)
end

function MMI.fit(detector::SupervisedDetector, verbosity, X, y)
    result = fit(detector, X, y; verbosity=verbosity)
    assert_result(result)
    model, scores = result
    return FitResult(model, scores), nothing, (;scores)
end

function MMI.transform(detector::Detector, fitresult::FitResult, X)
    scores_test = transform(detector, fitresult.model, X)
    assert_scores(scores_test)
    return fitresult.scores, scores_test
end

# specify scitypes
MMI.input_scitype(::Type{<:Detector}) = Union{AbstractMatrix{<:MMI.Continuous},MMI.Table(MMI.Continuous)}
MMI.target_scitype(::Type{<:Detector}) = AbstractVector{<:Union{OrderedFactor{2},Missing}}
MMI.output_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}
MMI.transform_scitype(::Type{<:Detector}) = Tuple{AbstractVector{<:MMI.Continuous}, AbstractVector{<:MMI.Continuous}}

# helper function to generate colons according to the data dimensionality
ncolons(X::Data) = ntuple(_ -> Colon(), Val(ndims(X) - 1))
