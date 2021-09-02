module OutlierDetectionInterface
    using Reexport:@reexport
    using CategoricalArrays:AbstractCategoricalVector,CategoricalValue

    const CLASS_NORMAL = "normal"
    const CLASS_OUTLIER = "outlier"
    const DEFAULT_THRESHOLD = 0.9

    @reexport using MLJModelInterface
    const MMI = MLJModelInterface

    # we might customize this macro later
    const var"@detector_model" = var"@mlj_model"

    export @detector_model,
           Detector,
           UnsupervisedDetector,
           SupervisedDetector,
           Model,
           Labels,
           Data,
           Result,
           Scores,
           CLASS_NORMAL,
           CLASS_OUTLIER,
           DEFAULT_THRESHOLD,
           to_categorical,
           to_univariate_finite,
           fit,
           transform

    # those do not get reexported from MLJModelInterface
    const _process_model_def = MMI._process_model_def
    const _model_constructor = MMI._model_constructor
    const _model_cleaner = MMI._model_cleaner

    export _process_model_def,
           _model_constructor,
           _model_cleaner

    include("base.jl")
    include("interface.jl")
end
