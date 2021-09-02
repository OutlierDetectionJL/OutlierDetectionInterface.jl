assert_result(result) = @assert isa(result, Tuple{Model, Scores}) "`OutlierDetectionInterface.fit` must return a tuple
 of `OutlierDetectionInterface.Model` and `OutlierDetectionInterface.Scores`"

assert_scores(scores) = @assert isa(scores, Scores) "`OutlierDetectionInterface.transform` must return  
`OutlierDetectionInterface.Scores`"

function MMI.fit(detector::UnsupervisedDetector, verbosity::Int, X)
    result = fit(detector, X); assert_result(result);
    model, scores = result
    return model, nothing, (scores = scores,)
end

function MMI.fit(detector::SupervisedDetector, verbosity::Int, X, y)
    result = fit(detector, X, y); assert_result(result);
    model, scores = result
    return model, nothing, (scores = scores,)
end

function MMI.transform(detector::Detector, model::Model, X)
    scores = transform(detector, model, X); assert_scores(scores);
    return scores
end

# helper to convert scores to univariate finite
function to_univariate_finite(scores::Scores)
    MMI.UnivariateFinite([CLASS_NORMAL, CLASS_OUTLIER], scores; augment=true, pool=missing, ordered=true)
end

# helper to convert classes to categorical values
function to_categorical(classes::Labels)
    # explicit cast to Vector{Union{String, Missing}} in case only missing values are passed
    MMI.categorical(Vector{Union{String, Missing}}(classes), ordered=true, levels=[CLASS_NORMAL, CLASS_OUTLIER])
end

# specify scitypes
MMI.input_scitype(::Type{<:Detector}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:Detector}) = AbstractVector{<:Union{Missing,OrderedFactor{2}}}
MMI.output_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}

# data front-end for fit (supervised):
MMI.reformat(::SupervisedDetector, X, y) = (MMI.matrix(X, transpose=true), y)
MMI.reformat(::SupervisedDetector, X, y, w) = (MMI.matrix(X, transpose=true), y, w) 
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

# data front-end for fit (unsupervised)/predict/transform
MMI.reformat(::Detector, X) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::Detector, I, Xmatrix) = (view(Xmatrix, :, I),)
