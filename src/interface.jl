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

# data front-end for fit (supervised):
MMI.reformat(::SupervisedDetector, X::Data, y) = (X, y)
MMI.reformat(::SupervisedDetector, X::Data, y, w) = (X, y, w) 
MMI.reformat(::SupervisedDetector, X, y) = (MMI.matrix(X, transpose=true), y)
MMI.reformat(::SupervisedDetector, X, y, w) = (MMI.matrix(X, transpose=true), y, w) 
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

# data front-end for fit (unsupervised)/predict/transform
MMI.reformat(::Detector, X::Data) = (X,)
MMI.reformat(::Detector, X) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::Detector, I, Xmatrix) = (view(Xmatrix, :, I),)
