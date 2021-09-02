"""
    UnsupervisedDetector

This abstract type forms the basis for all implemented unsupervised outlier detection algorithms. To implement a new
`UnsupervisedDetector` yourself, you have to implement the `fit(detector, X)::Fit` and
`score(detector, model, X)::Scores` methods. 
"""
abstract type UnsupervisedDetector <: MMI.Unsupervised end

"""
    SupervisedDetector

This abstract type forms the basis for all implemented supervised outlier detection algorithms. To implement a new
`SupervisedDetector` yourself, you have to implement the `fit(detector, X, y)::Fit` and
`score(detector, model, X)::Scores` methods. 
"""
abstract type SupervisedDetector <: MMI.Probabilistic end

"""
    Detector::Union{<:SupervisedDetector, <:UnsupervisedDetector}

The union type of all implemented detectors, including supervised, semi-supervised and unsupervised detectors. *Note:* A
semi-supervised detector can be seen as a supervised detector with `missing` labels denoting unlabeled data.
"""
const Detector = Union{<:SupervisedDetector,<:UnsupervisedDetector}

"""
    Model

A `Model` represents the learned behaviour for specific [`Detector`](@ref). This might include parameters in parametric
models or other repesentations of the learned data in nonparametric models. In essence, it includes everything required
to transform an instance to an outlier score.
"""
abstract type Model end

"""
    Score::AbstractVector{<:Real}

Scores are continuous values, where the range depends on the specific detector yielding the scores. *Note:* All
detectors return increasing scores and higher scores are associated with higher outlierness.
"""
const Scores = AbstractVector{<:Real}

"""
    Label::AbstractVector{<:String}

Labels are used for supervision and evaluation and are defined as an `AbstractArray{<:AbstractString}`. The convention
for labels is that `"outlier"` indicates outliers, `"normal"` indicates inliers and `missing` indicates unlabeled data.
"""
# Labels may contain missing values, strings or categorical strings
const Labels = AbstractVector{<:Union{Missing, String, CategoricalValue{String, <:Integer}}}

"""
    Data::AbstractArray{<:Real}

The raw input data for every detector is defined as`AbstractArray{<:Real}` and should be a one observation per last axis
in an n-dimensional array. It represents the input data used to [`fit`](@ref) a [`Detector`](@ref) and 
[`score`](@ref) [`Data`](@ref).
"""
const Data = Union{AbstractArray{<:Real}, SubArray{<:Real}}

const DATA_ARGUMENT = """    X::AbstractArray{<:Real}
An array of real values with one observation per last axis."""

const LABEL_ARGUMENT = """    y::AbstractVector{String}
A vector of labels with `"outlier"` indicating an outlier and `"normal"` indicating an inlier."""

const DETECTOR_ARGUMENT = """    detector::Detector
Any [`UnsupervisedDetector`](@ref) or [`SupervisedDetector`](@ref) implementation."""

SCORE_UNSUPERVISED(name::String) = """
```julia
using OutlierDetection: $name, fit, score
detector = $name()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
```"""

SCORE_SUPERVISED(name::String) = """
```julia
using OutlierDetection: $name, fit, score
detector = $name()
X = rand(10, 100)
y = rand([-1,1], 100)
model = fit(detector, X, y)
train_scores, test_scores = score(detector, model, X)
```"""

const RESULT = """
```julia
using OutlierDetection: Class, KNNDetector, fit, score
detector = KNNDetector()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
ŷ = detect(Class(), train_scores, test_scores)
# or, if using multiple detectors
# ŷ = detect(Class(), (train1, test1), (train2, test2), ...)
```"""

NO_DETECTOR(method::String) = "`OutlierDetectionInterface.$method` called with no detector, make sure that your detector 
extends `OutlierDetectionInterface.$method`"

"""
    fit(detector,
        X,
        y)

Fit a specified unsupervised, supervised or semi-supervised outlier detector. That is, learn a `Model` from input data
`X` and, in the supervised and semi-supervised setting, labels `y`. In a supervised setting, the label `"outlier"`
represents outliers and `"normal"` inliers. In a semi-supervised setting, `missing` additionally represents unlabeled
data. *Note:* Unsupervised detectors can be fitted without specifying `y`.

Parameters
----------
$DETECTOR_ARGUMENT

$DATA_ARGUMENT

Returns
----------
    fit::Fit
The learned model of the given detector, which contains all the necessary information for later prediction and the
achieved outlier scores of the given input data `X`.

Examples
--------
$(SCORE_UNSUPERVISED("KNNDetector"))
""" # those definitions apply when the type of X (or y) is not matching
fit(_::UnsupervisedDetector, X::Data) = throw(DomainError(NO_DETECTOR("fit")))
fit(_::UnsupervisedDetector, X::Data, y::Labels) = throw(DomainError(NO_DETECTOR("fit")))
fit(_::UnsupervisedDetector, X) = throw(DomainError("Unsupervised detectors can only be fitted with array inputs 
with one observation per last dimension, found $(typeof(X))"))
fit(_::SupervisedDetector, X, y) = throw(DomainError("Supervised detectors can only be fitted with array inputs 
with one observation per last dimension and `Labels`, found X=$(typeof(X)), y=$(typeof(y))"))

"""
    score(detector,
          model,
          X)

Transform input data `X` to outlier scores using an [`UnsupervisedDetector`](@ref) or [`SupervisedDetector`](@ref) and
a corresponding [`Model`](@ref).

Parameters
----------
$DETECTOR_ARGUMENT

    model::Model
The model learned from using [`fit`](@ref) with a supervised or unsupervised [`Detector`](@ref)

$DATA_ARGUMENT

Returns
----------
    result::Scores
Tuple of the achieved outlier scores of the given train and test data.

Examples
--------
$(SCORE_UNSUPERVISED("KNNDetector"))
""" # definition applies when X is not already an abstract array
transform(_::Detector, _::Model, X::Data) = throw(DomainError(NO_DETECTOR("transform")))
transform(_::Detector, _::Model, X) = throw(DomainError("Detectors can only predict with array inputs with one observation 
per last dimension, found $(typeof(X))"))
